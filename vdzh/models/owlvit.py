from typing import Optional
import torch

from transformers import OwlViTProcessor, OwlViTForObjectDetection, AutoImageProcessor, OwlViTConfig
from transformers.models.owlvit.modeling_owlvit import OwlViTObjectDetectionOutput

class OwlViT(OwlViTForObjectDetection):
    pretrained_url = "google/owlvit-base-patch32"
    def __init__(self, input_channels, out_channels, model_name="b0", pretrained=True, main_pretrained=None, **kwargs):

        if model_name == "custom":
            config = OwlViTConfig(num_channels=input_channels, **kwargs)
        else:
            config = OwlViTConfig().from_pretrained(self.pretrained_url)
            config.num_channels = input_channels
        super().__init__(config)
        if pretrained:
            self.init_pretrained_weights(channels_to_load=main_pretrained)
        
    def init_pretrained_weights(self, weights=None, channels_to_load=None):
        weights = OwlViTForObjectDetection.from_pretrained(self.pretrained_url).state_dict() if weights is None else weights
        self.load_state_dict(weights)
        print(f"Loaded pretrained weights from {self.pretrained_url}")
        
    def forward(self, inputs):
        input_ids, pixel_values, attention_mask = inputs
        return super().forward(input_ids, pixel_values, attention_mask, output_attentions=False, output_hidden_states=False, return_dict=False)
        