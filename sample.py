import torch
from vdzh.data.visdrone import VisDroneDatasetInterface

visdrone = VisDroneDatasetInterface({"root": "../Datasets/VisDrone2023ZeroShot/raw"})
visdrone.build_data_loaders(train_batch_size=2, num_workers=0, pin_memory=False)

batch = next(iter(visdrone.train_loader))
x = 10
print(x)