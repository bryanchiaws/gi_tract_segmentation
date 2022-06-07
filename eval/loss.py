import torch
import argparse
import segmentation_models_pytorch as smp
import torch.nn as nn


def sim(z_i, z_j):
    norm_dot_product = None
    norm_dot_product = torch.dot(z_i, z_j)/\
    (torch.linalg.norm(z_i) * torch.linalg.norm(z_j))
    return norm_dot_product

def sim_positive_pairs(out_left, out_right):
    """Normalized dot product between positive pairs.

    Inputs:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    Each row is a z-vector for an augmented sample in the batch.
    The same row in out_left and out_right form a positive pair.
    
    Returns:
    - A Nx1 tensor; each row k is the normalized dot product between out_left[k] and out_right[k].
    """
    pos_pairs = None
    
    pos_pairs = torch.sum(out_left * out_right, dim = 1) / \
    (torch.linalg.norm(out_left, dim = 1) * torch.linalg.norm(out_right, dim = 1))

    pos_pairs = pos_pairs.unsqueeze(1)
    return pos_pairs


def compute_sim_matrix(out):
    """Compute a 2N x 2N matrix of normalized dot products between all pairs of augmented examples in a batch.

    Inputs:
    - out: 2N x D tensor; each row is the z-vector (output of projection head) of a single augmented example.
    There are a total of 2N augmented examples in the batch.
    
    Returns:
    - sim_matrix: 2N x 2N tensor; each element i, j in the matrix is the normalized dot product between out[i] and out[j].
    """
    sim_matrix = None

    sim_matrix = torch.mm(out, out.T) 
    sim_matrix/= torch.linalg.norm(out, dim = 1).unsqueeze(1)
    sim_matrix/= torch.linalg.norm(out, dim = 1).unsqueeze(1).T

    return sim_matrix

def simclr_loss_vectorized(out_left, out_right, tau):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N = out_left.shape[0]
    
    # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    # Compute similarity matrix between all pairs of augmented examples in the batch.
    sim_matrix = compute_sim_matrix(out)  # [2*N, 2*N]
    
    # Step 1: Use sim_matrix to compute the denominator value for all augmented samples.
    # Hint: Compute e^{sim / tau} and store into exponential, which should have shape 2N x 2N.
    exponential = torch.exp(sim_matrix/tau)
    
    # This binary mask zeros out terms where k=i.
    mask = (torch.ones_like(exponential, device=device) - torch.eye(2 * N, device=device)).to(device).bool()
    
   #print((sim_matrix * mask).argmax(dim = 1))

    # We apply the binary mask.
    exponential = exponential.masked_select(mask).view(2 * N, -1)
    
    # Hint: Compute the denominator values for all augmented samples. This should be a 2N x 1 vector.
    denom = torch.sum(exponential, dim = 1).unsqueeze(1)
 
    # Step 2: Compute similarity between positive pairs.

    sim_pos = sim_positive_pairs(out_left, out_right)

    numerator = None

    numerator = torch.exp(sim_pos/tau)
    
    # Step 4: Now that you have the numerator and denominator for all augmented samples, compute the total loss.
    loss = None

    loss = torch.sum(-torch.log(numerator/denom[:N]) - torch.log(numerator/denom[N:])) / (2*N)

    #print(numerator/denom[:N])
    #print(numerator/denom[N:])
    
    return loss

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, truth, aux = None, aux_features = None):

        if pred is not None:
            DiceLoss = smp.losses.DiceLoss(mode = 'multilabel', from_logits = True)
            BCELoss = smp.losses.SoftBCEWithLogitsLoss()
            Loss = 0.4 * DiceLoss(pred, truth) + 0.6 * BCELoss(pred,truth)

        if aux == "simclr":
            N, D = aux_features.shape
            if (N % 2) == 1:
                aux_features = aux_features[:-1, :]
                N, D = aux_features.shape

            aux_features = aux_features.reshape(N//2, 2, D)

            SimClrLoss = simclr_loss_vectorized(aux_features[:, 0, :], aux_features[:, 1, :],  tau = 0.1)

            if pred is None:
                return SimClrLoss
            
            else:
                Loss = 0.9 * Loss + 0.1 * SimClrLoss
                #print('simclr_loss:', SimClrLoss)
                return Loss, SimClrLoss

        elif aux == "reg":
            #Add reg loss code here
            pass
        
        else:
            return Loss

def get_loss_fn(loss_args):
    loss_args_ = loss_args
    if isinstance(loss_args, argparse.Namespace):
        loss_args_ = vars(loss_args)
    loss_fn = loss_args_.get("loss_fn")

    if loss_fn == "BCE":
        return torch.nn.BCEWithLogitsLoss()
    elif loss_fn == "CE":
        return torch.nn.CrossEntropyLoss()
    elif loss_fn == 'DBE':
        return smp.losses.DiceLoss(mode = 'binary', from_logits = True)
    elif loss_fn == 'DLE':
        return smp.losses.DiceLoss(mode = 'multilabel', from_logits = True)
    elif loss_fn == 'Combined':
        return CombinedLoss()
    else:
        raise ValueError(f"loss_fn {loss_args.loss_fn} not supported.")
