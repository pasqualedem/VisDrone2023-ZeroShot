import torch

from tqdm import tqdm
from pprint import pprint
from .visdrone import VisDroneDatasetInterface


def show_statistics(root):
    visdrone = VisDroneDatasetInterface(dict(root=root))
    trainset = visdrone.trainset
    
    tot_boxes = 0
    different_labels_per_image = 0
    hist = torch.tensor([])
    max_labels = 0
    min_labels = 100
    
    for item in tqdm(trainset):
        unique_labels = len(item["labels"]["class_labels"].unique())
        different_labels_per_image += unique_labels
        
        num_labels = len(item["labels"]["class_labels"])
        tot_boxes += num_labels
        if num_labels > max_labels:
            max_labels = num_labels
        if num_labels < min_labels:
            min_labels = num_labels
        
        hist = torch.cat([hist, item["labels"]["class_labels"]])
    
    hist = hist.sort().values
    values, counts = hist.unique(return_counts=True)
    percentages = counts / counts.sum()
    
    hist = {trainset.id2label[k.item()]: v.item() for k, v in zip(values, percentages)}
    hist = {k: v for k, v in sorted(hist.items(), key=lambda item: item[1])}
    
    print(f"\
        Total number of images:                       {len(trainset)}\n\
        Total number of boxes:                        {tot_boxes}\n\
        Average number of boxes per image:            {tot_boxes / len(trainset)}\n\
        Average number of different labels per image: {different_labels_per_image / len(trainset)}\n\
        Max number of boxes per image:                {max_labels}\n\
        Min number of boxes per image:                {min_labels}\n\
    ")
    pprint(hist)
    
"""
Total number of images:                       8730
Total number of boxes:                        58226
Average number of boxes per image:            6.669644902634594
Average number of different labels per image: 1.429667812142039
Max number of boxes per image:                596
Min number of boxes per image:                1
    
{'Expressway-Service-area': 0.0183766707777977,
 'Expressway-toll-station': 0.009720743633806705,
 'airplane': 0.030879676342010498,
 'baseballfield': 0.030810978263616562,
 'bridge': 0.01835949532687664,
 'chimney': 0.010974478907883167,
 'dam': 0.008724624291062355,
 'golffield': 0.00707587692886591,
 'harbor': 0.04003366082906723,
 'overpass': 0.021450897678732872,
 'ship': 0.4685535728931427,
 'stadium': 0.006062583532184362,
 'storagetank': 0.049479614943265915,
 'tenniscourt': 0.05945797264575958,
 'trainstation': 0.008106344379484653,
 'vehicle': 0.21193280816078186}
"""