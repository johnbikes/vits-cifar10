import os
from dotenv import load_dotenv

load_dotenv()

hf_hub_cache = os.getenv("HF_HUB_CACHE")
hf_ds_cache = os.getenv("HF_DATASETS_CACHE")
hf_home = os.getenv("HF_HOME")

print(f"{hf_hub_cache = }. {hf_ds_cache = }, {hf_home = }")

if hf_home is not None:
    os.environ['HF_HOME'] = hf_home
    print('Using HF_HOME')
else:
    if hf_hub_cache is not None:
        os.environ['HF_HUB_CACHE'] = hf_hub_cache
        print('Using HF_HUB_CACHE')

    if hf_ds_cache is not None:
        os.environ['HF_DATASETS_CACHE'] = hf_ds_cache
        print('Using HF_DATASETS_CACHE')

# PyTorch
import torch
import torchvision
from torchvision.transforms import Normalize, Resize, ToTensor, Compose
# For dislaying images
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
# Loading dataset
from datasets import load_dataset
# Transformers
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import TrainingArguments, Trainer
# Matrix operations
import numpy as np
# Evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def main():
    print("\nall good!\n")

if __name__ == "__main__":
    main()