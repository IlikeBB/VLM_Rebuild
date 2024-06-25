"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

from utils.base_dataset import BaseDataset
from utils.base_processor import BaseProcessor
from PIL import Image
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import random, re

class BlipCaptionProcessor(BaseProcessor):
    def __init__(self, prompt="", max_words=50):
        self.prompt = prompt
        self.max_words = max_words

    def __call__(self, caption):
        caption = self.prompt + self.pre_caption(caption)

        return caption

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        prompt = cfg.get("prompt", "")
        max_words = cfg.get("max_words", 50)

        return cls(prompt=prompt, max_words=max_words)

    def pre_caption(self, caption):
        caption = re.sub(
            r"([.!\"()*#:;~])",
            " ",
            caption.lower(),
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > self.max_words:
            caption = " ".join(caption_words[: self.max_words])

        return caption

class BlipImageBaseProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)
        
class Blip2ImageTrainProcessor(BlipImageBaseProcessor):
    def __init__(self, image_size=448, mean=None, std=None, min_scale=0.5, max_scale=1.0):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size,image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 224)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
        )

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )

class COCOCaptionDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_root = None, ann_paths = None, img_size = 224):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.img_size = img_size
        vis_processor = Blip2ImageTrainProcessor(image_size=self.img_size)
        text_processor = BlipCaptionProcessor()
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.img_ids = {}
        n = 0

        self.filter_anntation = []
        
        for ann in self.annotation:
            if "train" in ann["image"]:
                self.filter_anntation.append(ann)
            else:
                self.filter_anntation.append(ann)
        self.annotation = self.filter_anntation
        print(self.annotation[0])
        img_file = ann["image"].split("/")[-1]
        image_path = os.path.join(self.vis_root, img_file)
        print(image_path,'\n')
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

        self.instruction_pool = [
            "Does this photo have any flaws?", #<=new
            "Please answer Flawed or Flawless.", #<=new
            "Please answer Defective product or Flawless product." #<=new
        ]
    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]
        img_file = ann["image"].split("/")[-1]
        image_path = os.path.join(self.vis_root, img_file)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        caption = self.text_processor(ann["caption"])
        instruction = random.choice(self.instruction_pool)
        # instruction = custom_des
        instruction = "<Img><ImageHere></Img> [caption] {} ".format(instruction)
        return {
            "image": image,
            "answer": caption,
            "instruction_input": instruction,
        }

class CaptionEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        return {
            "image": image,
            "image_id": ann["image_id"],
            "instance_id": ann["instance_id"],
        }
