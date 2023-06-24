import torch.nn as nn

from transformers.models.detr.modeling_detr import DetrLoss, DetrHungarianMatcher

class ObjLoss(nn.Module):
    def __init__(self, num_classes, giou_cost=2, class_cost=1, bbox_cost=5, eos_coefficient=0.1):
        super().__init__()
        self.giou_cost = giou_cost
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.num_labels = num_classes
        losses = ["labels", "boxes", "cardinality"]
        matcher = DetrHungarianMatcher(
            class_cost=class_cost, bbox_cost=bbox_cost, giou_cost=giou_cost
        )

        self.detr_loss = DetrLoss(
            matcher=matcher,
            num_classes=num_classes,
            eos_coef=eos_coefficient,
            losses=losses,
            )
        
    def forward(self, outputs, targets):
        self.detr_loss(outputs, targets)