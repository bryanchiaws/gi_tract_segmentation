"""Define constants to be used throughout the repository."""

# Main paths

# Image and mask sizes
IMAGE_SIZE = 224

# Dataset constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# US latitude/longitude boundaries
US_N = 49.4
US_S = 24.5
US_E = -66.93
US_W = -124.784

# Test image
TEST_IMG_PATH = [".circleci/images/test_image.png"] * 2

#Trial dataset path
TEST_DATASET_PATH = "./data/final_test.csv"
