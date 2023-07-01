import torch
import torch.nn.functional as F

from transformers.models.detr.modeling_detr import DetrLoss, DetrHungarianMatcher
from ezdl.loss import ComposedLoss

class ObjLoss(ComposedLoss):
    def __init__(self, num_classes, giou_cost=2, class_cost_hungarian=1, bbox_cost_hungarian=5, bbox_cost=5, eos_coefficient=0.1):
        super().__init__()
        self.giou_cost = giou_cost
        self.bbox_cost = bbox_cost
        self.num_labels = num_classes
        self.losses = ["labels", "boxes", "cardinality"]
        matcher = DetrHungarianMatcher(
            class_cost=class_cost_hungarian, bbox_cost=bbox_cost_hungarian, giou_cost=giou_cost
        )
        self.weight_dict = {"loss_ce": 1, "loss_bbox": self.bbox_cost}
        self.weight_dict["loss_giou"] = self.giou_cost

        self.detr_loss = DetrLoss(
            matcher=matcher,
            num_classes=num_classes,
            eos_coef=eos_coefficient,
            losses=self.losses,
            )

    @property
    def component_names(self):
        """
        Component names for logging during training.
        These correspond to 2nd item in the tuple returned in self.forward(...).
        See super_gradients.Trainer.train() docs for more info.
        """
        return self.weight_dict.keys()
        
    def forward(self, outputs, targets, context):
        actual_dicts = [{label: i for i, label in enumerate(actual)} for actual in context.included_classes]
        max_len = max([len(actual) for actual in actual_dicts])
        actual_labels = [torch.tensor([actual_dict[label.item()] for label in labels["class_labels"]] + (max_len - len(actual_dict)) * [max_len - 1]
                                      , device=targets[0]["class_labels"].device) 
                         for actual_dict, labels in zip(actual_dicts, targets)]
        actual_boxes = [F.pad(target['boxes'], (0, 0, 0, max_len - len(target['boxes'])), value=-1)
                        for target in targets]
        actual_targets = [{**target, "class_labels": actual_label, "boxes": actual_boxes} for actual_label, target in zip(actual_labels, targets)]
        
        self.detr_loss.empty_weight = torch.ones(max_len, device=actual_labels[0].device)
        self.detr_loss.empty_weight[-1] = 0.1
        
        loss_dict = self.detr_loss(outputs, actual_targets)
        loss = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys() if k in self.weight_dict)
        loss_dict = {k: v for k, v in loss_dict.items() if k in self.weight_dict}
        return loss, torch.cat(tuple(curr.unsqueeze(0) for curr in loss_dict.values())).detach()