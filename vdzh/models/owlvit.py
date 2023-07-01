from typing import Optional
import torch.nn.functional as F
import torch.nn as nn
import torch

from transformers import OwlViTProcessor, OwlViTForObjectDetection, AutoImageProcessor, OwlViTConfig
from transformers.models.owlvit.modeling_owlvit import OwlViTObjectDetectionOutput

class OwlViT(OwlViTForObjectDetection):
    """
    Wrapper for OwlViTForObjectDetection
    """
    pretrained_url = "google/owlvit-base-patch32"
    def __init__(self, input_channels, out_channels, model_name="b0", pretrained=True, main_pretrained=None, threshold=0.1, **kwargs):
        self.threshold = threshold

        if model_name == "custom":
            config = OwlViTConfig(num_channels=input_channels, **kwargs)
        else:
            config = OwlViTConfig().from_pretrained(self.pretrained_url)
            config.num_channels = input_channels
        super().__init__(config)
        if pretrained:
            self.init_pretrained_weights(channels_to_load=main_pretrained)
        self.processor = OwlViTProcessor.from_pretrained(self.pretrained_url)
        
    def init_pretrained_weights(self, weights=None, channels_to_load=None):
        weights = OwlViTForObjectDetection.from_pretrained(self.pretrained_url).state_dict() if weights is None else weights
        self.load_state_dict(weights)
        print(f"Loaded pretrained weights from {self.pretrained_url}")
        
    def forward(self, inputs):
        input_ids, pixel_values, attention_mask = inputs
        pred = super().forward(input_ids, pixel_values, attention_mask, output_attentions=False, output_hidden_states=False)
        return pred

class DummyDetector(nn.Module):
    def __init__(self, input_channels, out_channels, **kwargs):
        super().__init__()
        embed_dim = 32
        self.embed_dim = embed_dim
        self.conv = nn.Conv2d(input_channels, embed_dim, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(4, out_channels + 1)

    def forward(self, x):
        input_ids, pixel_values, attention_mask = x
        x = self.conv(pixel_values)
        x = self.pool(x)
        boxes = F.relu(x.flatten(1).reshape(2, self.embed_dim // 4, 4))
        scores = self.classifier(boxes)
        return {"pred_boxes": boxes, "logits": scores}
    

class RandomDetector(nn.Module):
    def __init__(self, input_channels, out_channels, **kwargs):
        super().__init__()
        self.out_channels = out_channels

    def forward(self, x):
        input_ids, pixel_values, attention_mask = x
        batch_size = pixel_values.shape[0]
        boxes = torch.rand(batch_size, 8, 4, requires_grad=True).to(pixel_values.device)
        scores = torch.rand(batch_size, 8, self.out_channels + 1, requires_grad=True).to(pixel_values.device)
        return {"pred_boxes": boxes, "logits": scores}