import torch
import numpy as np
from DataLoaderWithoutNormalization import data_loader

#   2. Dataset Parameters:
dataset_root_path = "../datasets"
#dataset = "ChestXRay"
dataset = "ISIC2016"
validation_split = 0.2
batch_size = 1024
INPUT_SIZE = (1, 3, 224, 224)

# Load dataset
TRAIN_LOADER, VALIDATION_LOADER, TEST_LOADER = data_loader(dataset_root_path, dataset, batch_size=batch_size)

pop_mean = []
pop_std0 = []
pop_std1 = []
for i, (image, label) in enumerate(TEST_LOADER, 0):
    # shape (batch_size, 3, height, width)
    numpy_image = image.numpy()
    
    # shape (3,)
    batch_mean = np.mean(numpy_image, axis=(0,2,3))
    batch_std0 = np.std(numpy_image, axis=(0,2,3))
    batch_std1 = np.std(numpy_image, axis=(0,2,3), ddof=1)
    
    pop_mean.append(batch_mean)
    pop_std0.append(batch_std0)
    pop_std1.append(batch_std1)

# shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
pop_mean = np.array(pop_mean).mean(axis=0)
pop_std0 = np.array(pop_std0).mean(axis=0)
pop_std1 = np.array(pop_std1).mean(axis=0)

print(pop_mean)
print(pop_std0)
print(pop_std1)
