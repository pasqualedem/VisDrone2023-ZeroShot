import os
import numpy as np
import pandas as pd
import torch
import albumentations
import xmltodict

from PIL import Image
from torchvision.datasets.vision import VisionDataset
from torchvision.ops.boxes import box_convert
from transformers import OwlViTProcessor, AutoImageProcessor
from ezdl.data import DatasetInterface


def xml_to_dict(xml_file):
    with open(xml_file, "r") as f:
        return xmltodict.parse(f.read())
    

class VisDroneDatasetInterface(DatasetInterface):
    size = (3, 800, 800)

    def __init__(self, dataset_params):
        super().__init__(dataset_params)
        
        image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
        image_processor.do_resize = False
        image_processor.do_center_crop = False
        image_processor.do_normalize = False
        image_processor.do_rescale = False
        
        mean_std_targets_per_image = dataset_params.get("mean_std_targets_per_image", (2, 1))
        
        def formatted_anns(sample):
            """
            Format annotations of structure:
            {'annotation': {'object': [{'bndbox': {'xmax': '1024',
                                                    'xmin': '0',
                                                    'ymax': '1024',
                                                    'ymin': '0'},
            """
            ann_list = sample["annotations"]["annotation"]["object"]
            if isinstance(ann_list, dict):
                ann_list = [ann_list]
            categories = [VisDroneDataset.label2id[ann['name']] for ann in ann_list]
            bboxes = [ann['bndbox'] for ann in ann_list]
            bboxes = [[int(box['xmin']), int(box['ymin']), int(box['xmax']), int(box['ymax'])] for box in bboxes]
            bboxes = box_convert(torch.tensor(bboxes), in_fmt="xyxy", out_fmt="xywh")
            # annotations = []
            # for i in range(len(category)):
            #     new_ann = {
            #         "image_id": image_id,
            #         "category_id": category[i],
            #         "isCrowd": 0,
            #         "area": area[i],
            #         "bbox": list(bbox[i]),
            #     }
            #     annotations.append(new_ann)
            return {"category": categories, "bbox": bboxes}
    
        def transform_aug_ann(sample):
            image_ids = sample["image_id"]
            annotations = formatted_anns(sample)
            outs = transform(image=sample['image'], bboxes=annotations['bbox'], category=annotations["category"])
            # images, bboxes, area, categories = [], [], [], []
            # for image, objects in zip(examples["image"], examples["objects"]):
            #     image = np.array(image.convert("RGB"))[:, :, ::-1]
            #     out = transform(image=image, bboxes=objects["bbox"], category=objects["category"])

            #     area.append(objects["area"])
            #     images.append(out["image"])
            #     bboxes.append(out["bboxes"])
            #     categories.append(out["category"])
            targets = {"image_id": [image_ids], 
                       "annotations": [{"image_id": image_ids,
                                  "category_id": cat,
                                  "bbox": list(bbox),
                                  "area": [bbox[2]*bbox[3]]} 
                                  for cat, bbox in zip(outs["category"], outs["bboxes"])]}
            
            processed =  image_processor(images=outs['image'], annotations=targets, return_tensors="pt") # Now boxes are in xyxy format
            for anns in processed['labels']:
                anns['boxes'] = box_convert(torch.tensor(anns['boxes']) / torch.tensor(processed['pixel_values'].shape[-2:]).repeat(2), in_fmt="xyxy", out_fmt="cxcywh") # normalize into [0, 1]
                
            processed['input_name'] = [sample['input_name']]
            processed['pixel_values'] = processed['pixel_values'].numpy()
            return {k: v[0] for k, v in processed.items()} # Remove batch dimension

        self.lib_dataset_params = {
            'mean': image_processor.image_mean,
            'std': image_processor.image_std,
            'channels': 3
        }
        transform = albumentations.Compose(
            [
                # albumentations.Resize(480, 480), 
                albumentations.HorizontalFlip(p=.5),
                albumentations.RandomBrightnessContrast(p=.5),
            ],
            bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
        )

        self.trainset = VisDroneDataset(os.path.join(dataset_params["root"], "trainset"), 
                                        transforms=transform_aug_ann, mean_std_targets_per_image=mean_std_targets_per_image)
        self.valset = None
        self.testset = None

class VisDroneDataset(VisionDataset):
    seen_classes = ['airplane', 'baseballfield', 'bridge', 'chimney', 'dam', 'Expressway-Service-area','Expressway-toll-station', 'golffield', 'harbor', 'overpass', 'ship', 'stadium', 'storagetank','tenniscourt', 'trainstation', 'vehicle']
    unseen_classes = ['airport', 'basketballcourt', 'groundtrackfield', 'windmill']
    classes = seen_classes + unseen_classes
    label2id = {label: i for i, label in enumerate(classes)}
    id2label = {i: label for i, label in enumerate(classes)}
    distribution = {
        'airplane': 0.030879676342010498,
        'baseballfield': 0.030810978263616562,
        'bridge': 0.01835949532687664,
        'chimney': 0.010974478907883167,
        'dam': 0.008724624291062355,
        'Expressway-Service-area': 0.0183766707777977,
        'Expressway-toll-station': 0.009720743633806705,
        'golffield': 0.00707587692886591,
        'harbor': 0.04003366082906723,
        'overpass': 0.021450897678732872,
        'ship': 0.4685535728931427,
        'stadium': 0.006062583532184362,
        'storagetank': 0.049479614943265915,
        'tenniscourt': 0.05945797264575958,
        'trainstation': 0.008106344379484653,
        'vehicle': 0.21193280816078186
        }

    def __init__(self, root: str, mean_std_targets_per_image=None, transforms=None):
        """
        Constructor of DroneVisionDataset

        Args:
            root: Root path of the dataset
        """
        super().__init__(root)
        self.root = root
        self.mean_std_targets_per_image = mean_std_targets_per_image
        self.sample_list = [f for f in os.listdir(os.path.join(root, "JPEGImages"))]
        self.transforms = transforms

        checkpoint = "google/owlvit-base-patch32"
        self.processor = OwlViTProcessor.from_pretrained(checkpoint)

    def collate_fn(self, batch):
        """
        Args:
            batch: list of samples returned by __getitem__
        Returns:
            A tuple of tensors ((input_ids, images, attention_mask), annotations, {input_name}).
        """
        pixel_values = [item["pixel_values"] for item in batch]
        labels = [dict(item["labels"]) for item in batch]
        text = [list(set([self.id2label[obj.item()] for obj in item['class_labels']])) for item in labels]
        text = [self.sample_categories(objects) for objects in text]
        encoding = self.processor(images=pixel_values, text=text, return_tensors="pt")
        return ((encoding['input_ids'], encoding["pixel_values"], encoding["attention_mask"]),
                labels,
                {
                    "input_name": [item["input_name"] for item in batch],
                    "included_classes": [[self.label2id[obj] for obj in objects] 
                                         for objects in text]}
                )

    def getFilesName(self):
        """
        Returns:
            List of all the image names in the dataset
        """
        return self.sample_list

    def sample_categories(self, objects):
        """
        Sample categories according to a distribution

        Args:
            objects: list of categories

        Returns:
            List of sampled categories
        """
        mean, std = self.mean_std_targets_per_image
        # Don't sample if there are more than mean + std objects
        if len(objects) > mean + std:
            return objects
        distribution = {k: (v if k not in objects else 0) for k, v in self.distribution.items()}
        probs = torch.tensor(list(distribution.values()))
        probs = probs / probs.sum()
        n_samples = torch.normal(mean, std, size=(1,)).int().item()
        samples = []
        for i in range(n_samples):
            sample = torch.distributions.Categorical(probs=probs).sample().item()
            samples.append(sample)
            probs[sample] = 0
            probs = probs / probs.sum()
        return [self.id2label[sample] for sample in samples] + objects

    def __getitem__(self, index: int):
        """
        Reads the image and the associated annotations from files

        Args:
            index: index of the image name in the sample_list

        Returns:
            image: image tensor encoded with image_processor of the resized image.
            annotations: all the annotations of the image in YOLO format
        """

        image = Image.open(os.path.join(self.root, "JPEGImages", self.sample_list[index]))
        annotation_path = os.path.join(self.root, "Annotations", self.sample_list[index].split("/")[-1].split(".")[0] + ".xml")
        annotations = xml_to_dict(annotation_path)

        sample = {
            "size": torch.tensor(image.size),
            "image": np.array(image),
            "image_id": torch.tensor([index]),
            "input_name": self.sample_list[index],
            "annotations": annotations,
        }
        return self.transforms(sample)

    def __len__(self) -> int:
        return len(self.sample_list)