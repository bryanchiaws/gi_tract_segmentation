import torch.distributed as dist
import numpy as np
import ignite
import torch
from torch.nn.functional import sigmoid
from tqdm import tqdm

from . import detection
from . import classification
from . import segmentation


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        raise NotImplementedError(
            "[reset] method need to be implemented in child class.")

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        raise NotImplementedError(
            "[process] method need to be implemented in child class.")

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        """
        raise NotImplementedError(
            "[evaluate] method need to be implemented in child class.")

class SegmentationEvaluator(DatasetEvaluator):
    def __init__(self, threshold = 0.5):
        super().__init__()
        self.preds = []
        self.truth = []
        self.threshold = threshold

    def reset(self):
        self.preds = []
        self.truth = []
        return

    def process(self, inputs, logits_masks):
        images, masks = map(list, zip(*inputs))
        images = torch.stack(images)
        masks = torch.stack(masks)

        prob_masks = sigmoid(logits_masks)
        self.truth.append(masks.detach())
        self.preds.append(prob_masks >= self.threshold)
        return

    def evaluate(self):
        return tqdm(segmentation.get_metrics(self.preds, self.truth))

class SimCLREvaluator(DatasetEvaluator):
    def __init__(self, threshold = 0.5):
        super().__init__()
        self.preds = []
        self.truth = []
        self.threshold = threshold

    def reset(self):
        self.preds = []
        self.truth = []
        return

    def process(self, inputs, logits_masks):
        images, masks = map(list, zip(*inputs))
        images = torch.stack(images)
        masks = torch.stack(masks)

        prob_masks = sigmoid(logits_masks)
        self.truth.append(masks.detach())
        self.preds.append(prob_masks >= self.threshold)
        return

    def evaluate(self):
        return segmentation.get_metrics(self.preds, self.truth)

class BinaryClassificationEvaluator(
        ignite.metrics.EpochMetric,
        DatasetEvaluator):
    def __init__(self, threshold=None, check_compute_fn=True):
        self._threshold = threshold
        super().__init__(self.compute_fn)

    def compute_fn(self, prob, y):
        return classification.get_binary_metrics(
            prob, y, threshold=self._threshold)

    def evaluate(self):
        return self.compute()


class DetectionEvaluator(DatasetEvaluator):
    """
    Evaluator for detection task.
    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def __init__(self, iou_thresh=0.5):
        self._evaluator = detection.Evaluator()
        self._iou_thresh = iou_thresh
        self.reset()
        self._is_reduced = False

    def reset(self):
        self._bbox = detection.BoundingBoxes()
        self._is_reduced = True

    def process(self, groudtruths, predictions):
        """
        Inputs format:
        https://detectron2.readthedocs.io/en/latest/tutorials/models.html?highlight=input%20format#model-input-format
        Outputs format:
        https://detectron2.readthedocs.io/en/latest/tutorials/models.html?highlight=input%20format#model-output-format
        """
        for sample_input, sample_output in zip(groudtruths, predictions):
            image_id = sample_input['image_id']
            gt_instances = sample_input['instances']
            pred_instances = sample_output['instances']
            width = sample_input['width']
            height = sample_input['height']
            for i in range(len(gt_instances)):
                instance = gt_instances[i]
                class_id = instance.get(
                    'gt_classes').cpu().detach().numpy().item()
                boxes = instance.get('gt_boxes')
                for box in boxes:
                    box_np = box.cpu().detach().numpy()
                    bb = detection.BoundingBox(
                        image_id,
                        class_id,
                        box_np[0],
                        box_np[1],
                        box_np[2],
                        box_np[3],
                        detection.CoordinatesType.Absolute,
                        (width,
                         height),
                        detection.BBType.GroundTruth,
                        format=detection.BBFormat.XYX2Y2)
                    self._bbox.addBoundingBox(bb)
            for i in range(len(pred_instances)):
                instance = pred_instances[i]
                class_id = instance.get(
                    'pred_classes').cpu().detach().numpy().item()
                scores = instance.get('scores').cpu().detach().numpy().item()
                boxes = instance.get('pred_boxes')
                for box in boxes:
                    box_np = box.cpu().detach().numpy()
                    bb = detection.BoundingBox(
                        image_id,
                        class_id,
                        box_np[0],
                        box_np[1],
                        box_np[2],
                        box_np[3],
                        detection.CoordinatesType.Absolute,
                        (width,
                         height),
                        detection.BBType.Detected,
                        scores,
                        format=detection.BBFormat.XYX2Y2)
                    self._bbox.addBoundingBox(bb)
    
    @staticmethod
    def _parse_results(results):
        metrics = {}
        APs = []
        for result in results:
            score = np.array(result['score'])
            if len(score):
                precision = np.array(result['precision'])
                recall = np.array(result['recall'])
                f1 = 2 / (1 / precision + 1 / recall)
    
                metrics[f'F1_max_{result["class"]}'] = np.max(f1)
                ind = np.argmax(f1)
                metrics[f'F1_max_precision_{result["class"]}'] = precision[ind]
                metrics[f'F1_max_recall_{result["class"]}'] = recall[ind]
                metrics[f'F1_max_threshold_{result["class"]}'] = score[ind]
                metrics[f'AP_{result["class"]}'] = result['AP']
            APs.append(result['AP'])

        metrics['mAP'] = np.nanmean(APs)
        return metrics

    def evaluate(self, plot_save_dir=None):
        if dist.is_initialized() and not self._is_reduced:
            ws = dist.get_world_size()
            _bbox = [None for _ in range(ws)]
            dist.all_gather_object(_bbox, self._bbox._boundingBoxes)
            self._bbox._boundingBoxes = [
                box for boxes in _bbox for box in boxes]
            self._is_reduced = True
        else:
            self._is_reduced = True

        results = self._evaluator.GetPascalVOCMetrics(
            self._bbox, self._iou_thresh)
        if isinstance(results, dict):
            results = [results]

        metrics = self._parse_results(results)

        if plot_save_dir is not None:
            self._evaluator.PlotPrecisionRecallCurve(
                self._bbox, savePath=plot_save_dir, showGraphic=False)
        return metrics


class DatasetEvaluators(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results
