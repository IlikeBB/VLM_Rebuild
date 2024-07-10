

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from transformers import CLIPProcessor, CLIPModel

class VisionTransformer(nn.Module):
    def __init__(self, model_path = None, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=1024):
        super().__init__()
        self.image_size = img_size
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.forward_encoder = CLIPModel.from_pretrained(model_path, torch_dtype=torch.float16).vision_model
    def forward(self, x):
        x = self.forward_encoder(x)
        if len(x)!=1: #過濾掉clip中post層和自帶的layernorm
            x = x[0]
        return x

def convert_weights_to_fp16(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

    model.apply(_convert_weights_to_fp16)

def create_clip(model_path='None', img_size=None,drop_path_rate=0.4,use_checkpoint=False,precision="fp16"):
    model = VisionTransformer(model_path=model_path, img_size=img_size)
    if precision == "fp16":
#         model.to("cuda") 
        convert_weights_to_fp16(model)
    return model