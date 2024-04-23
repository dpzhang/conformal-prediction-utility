import numpy as np
import random
import torch
from utils_cp import *

# Fix randomness
np.random.seed(666)
torch.manual_seed(666)
torch.cuda.manual_seed(666)
random.seed(666)

if __name__ == "__main__":
    # Load the pre-trained wide resnet model trained on ImageNet
    wrn = torch.hub.load('pytorch/vision', 'wide_resnet101_2',
                         weights='Wide_ResNet101_2_Weights.IMAGENET1K_V2')

    # Load ImageNet validation data and human-readable label classes
    init = ImageNet_WRN(wrn)

    # Randomly split it the validation data into 50% calibration and 50% testing
    # Here, there are a total of 50k images, so we put 25k in the calibration
    vault = init.load_test_data(
        (0.5, 0.5)).load_class_labels().generate_subset_class_maps()

    cmodel_conditional = train_conformal(
        wrn=wrn, vault=vault,
        alpha=0.05, lamda_criterion='adaptiveness',
        randomized=True, allow_zero_sets=True)
