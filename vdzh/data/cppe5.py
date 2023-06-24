import itertools
import os
import albumentations
from typing import Any, Iterable


import numpy as np
import torch
from PIL import Image
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from datasets import load_dataset
from transformers import AutoProcessor, AutoImageProcessor, AutoModelForZeroShotObjectDetection, OwlViTProcessor

from ezdl.transforms import \
    PairRandomCrop, ToLong, FixValue, Denormalize, PairRandomFlip, squeeze0, \
    PairFlip, PairFourCrop, PairRandomRotation
from sklearn.model_selection import train_test_split

from super_gradients.training import utils as core_utils
from super_gradients.common.abstractions.abstract_logger import get_logger

from ezdl.data import DatasetInterface

logger = get_logger(__name__)

    
checkpoint = "google/owlvit-base-patch32"
processor = OwlViTProcessor.from_pretrained(checkpoint)
def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = processor(images=pixel_values, text=["mask", "gloves", "googles"], return_tensors="pt")
    labels = [item["labels"] for item in batch]
    # batch["pixel_mask"] = batch["pixel_mask"]
    return (encoding['input_ids'], encoding["pixel_values"], encoding["attention_mask"]), labels    
    

class CPPE5DatasetInterface(DatasetInterface):
    size = (10, 64, 64)

    def __init__(self, dataset_params):
        super(CPPE5DatasetInterface, self).__init__(dataset_params)
        
        image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
        image_processor.do_resize = False
        image_processor.do_center_crop = False
        image_processor.do_normalize = False
        image_processor.do_rescale = False
        def formatted_anns(image_id, category, area, bbox):
            annotations = []
            for i in range(len(category)):
                new_ann = {
                    "image_id": image_id,
                    "category_id": category[i],
                    "isCrowd": 0,
                    "area": area[i],
                    "bbox": list(bbox[i]),
                }
                annotations.append(new_ann)

            return annotations
    
        def transform_aug_ann(examples):
            image_ids = examples["image_id"]
            images, bboxes, area, categories = [], [], [], []
            for image, objects in zip(examples["image"], examples["objects"]):
                image = np.array(image.convert("RGB"))[:, :, ::-1]
                out = transform(image=image, bboxes=objects["bbox"], category=objects["category"])

                area.append(objects["area"])
                images.append(out["image"])
                bboxes.append(out["bboxes"])
                categories.append(out["category"])

            targets = [
                {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
                for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
            ]
            return image_processor(images=images, annotations=targets, return_tensors="pt")

        self.lib_dataset_params = {
            'mean': image_processor.image_mean,
            'std': image_processor.image_std,
            'channels': 3
        }
        transform = albumentations.Compose(
            [
                albumentations.Resize(480, 480),
                albumentations.HorizontalFlip(p=1.0),
                albumentations.RandomBrightnessContrast(p=1.0),
            ],
            bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
        )
        cppe5 = load_dataset("cppe-5")
        remove_idx = [590, 821, 822, 875, 876, 878, 879]
        keep = [i for i in range(len(cppe5["train"])) if i not in remove_idx]
        cppe5["train"] = cppe5["train"].select(keep)
        cppe5["train"] = cppe5["train"].with_transform(transform_aug_ann)

        self.trainset = cppe5["train"]
        self.trainset.classes = ["coveralls", "faceshield", "gloves", "mask", "goggles"]
        self.trainset.collate_fn = collate_fn
        self.valset = cppe5["test"]
        self.valset.collate_fn = collate_fn
        self.testset = cppe5["test"]
        self.testset.collate_fn = collate_fn

    def undo_preprocess(self, x):
        return (Denormalize(self.lib_dataset_params['mean'], self.lib_dataset_params['std'])(x) * 255).type(torch.uint8)
