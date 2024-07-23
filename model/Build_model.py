import os
import logging
import contextlib

from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn as nn
from transformers import LlamaTokenizer, AutoTokenizer, BitsAndBytesConfig
from peft import (LoraConfig, get_peft_model, prepare_model_for_kbit_training)
                #   prepare_model_for_int8_training
from model.eva_vit import create_eva_vit_g
from model.other_vit import create_clip
from model.MiniGPT_LlamaForCausalLM import LlamaForCausalLM

class build_vlm_model(nn.Module):
    def __init__(self,):
        super().__init__()
    @classmethod
    def init_llm(self, llama_model_path, low_resource=False, low_res_device=0, lora_r=0,
                    lora_target_modules=["q_proj","v_proj"], **lora_kargs):
            # logging.info('Loading LLAMA')
            print("Loading LLAMA")
            
            if 'Llama-3' in llama_model_path:
                # token_path = os.path.join(llama_model_path, 'original')
                llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_path, use_fast=False)
            else:
                llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model_path, use_fast=False)
            print("tokenizer pass")
            llama_tokenizer.pad_token = "$$"

            # if low_resource:
            #     llama_model = LlamaForCausalLM.from_pretrained(
            #         llama_model_path,
            #         torch_dtype=torch.float16,
            #         load_in_8bit=True,
            #         device_map="auto"
                # )
            # elif 'Llama-3' in llama_model_path:
            if low_resource:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    # pretraining_tp=1
                )
                llama_model = LlamaForCausalLM.from_pretrained(
                    llama_model_path,
                    quantization_config=quantization_config,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                )
            
            else:
                llama_model = LlamaForCausalLM.from_pretrained(
                    llama_model_path,
                    torch_dtype=torch.float16,
                )

            if lora_r > 0:
                llama_model = prepare_model_for_kbit_training(llama_model)
                loraconfig = LoraConfig(
                    r=lora_r,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=lora_target_modules,
                    **lora_kargs
                )
                llama_model = get_peft_model(llama_model, loraconfig)

                llama_model.print_trainable_parameters()

            else:
                for name, param in llama_model.named_parameters():
                    param.requires_grad = False
            logging.info('Loading LLAMA Done')
            return llama_model, llama_tokenizer

    def disabled_train(self, mode=True):
        """Overwrite model.train with this function to make sure train/eval mode
        does not change anymore."""
        return mode
    @classmethod
    def init_vision_encoder(self, model_name, img_size, drop_path_rate, use_grad_checkpoint, precision, freeze, vit_path=None):
        logging.info('Loading VIT')

        # assert model_name == "eva_clip_g", "vit model must be eva_clip_g for current version of MiniGPT-4"
        if not freeze:
            precision = "fp32"  # fp16 is not for training
        if model_name=='eva_clip_g':
            visual_encoder = create_eva_vit_g(
                img_size, drop_path_rate, use_grad_checkpoint, precision
            )
        else:
            visual_encoder = create_clip(model_path=vit_path, img_size=img_size, precision="fp16")
        ln_vision = LayerNorm(visual_encoder.num_features)

        if freeze:
            for name, param in visual_encoder.named_parameters():
                param.requires_grad = False
            visual_encoder = visual_encoder.eval()
            visual_encoder.train = self.disabled_train
            for name, param in ln_vision.named_parameters():
                param.requires_grad = False
            ln_vision = ln_vision.eval()
            ln_vision.train = self.disabled_train
            logging.info("freeze vision encoder")

        logging.info('Loading VIT Done')
        return visual_encoder, ln_vision

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
    

class Main_model(build_vlm_model):
    def __init__(self, llm_config = None, vit_config = None,):
        super().__init__()
        self.llama_model, self.llama_tokenizer = self.init_llm(llm_config['llama_model'], 
                            low_resource=llm_config['low_resource'], low_res_device=llm_config['low_res_device'], 
                            lora_r=llm_config['lora_r'], lora_target_modules=llm_config['lora_target_modules'],lora_alpha=llm_config['lora_alpha'],lora_dropout=llm_config['lora_dropout'],
                            )
        # if torch.cuda.is_available():
        #     self.llama_model = self.load_to_cuda(self.llama_model)

        # image shape = [bs, rgb, h, w](torch.float16)
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(vit_config['model_name'],vit_config['image_size'],
                                                                vit_config['drop_path_rate'],vit_config['use_grad_checkpoint'],
                                                                vit_config['vit_precision'],vit_config['freeze_vit'],vit_path = vit_config['model_path'])
        img_f_dim = self.visual_encoder.num_features * 4
        # model.visual_encoder.num_features = 1408
        self.llama_proj = nn.Linear(img_f_dim, self.llama_model.config.hidden_size)
