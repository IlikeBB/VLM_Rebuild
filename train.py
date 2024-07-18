import os, sys
import torch
import signal
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from main import Main__

model_config = {'amp':True, 'use_distributed':False,'accum_grad_iters':1,
                'batch_size':4,
                'chat_template': True, 
                'end_sym': [
                    '\n {} </s>',
                    "\n <|start_header_id|>assistant<|end_header_id|>{}<|eot_id|>"
                ],# list[0] = llama2, list[1] = llama3
                'prompt_template': [
                "<s>[INST] {} [/INST]", 
                """<|begin_of_text|><|start_header_id|>user<|end_header_id|>{}<|eot_id|>"""
                ],# list[0] = llama2, list[1] = llama3
                'max_txt_len': 1024, 'max_context_len': 3500,
                'ouput_dir': './model_FT_weight/llama3_vit_L-clip336', #./llama3_vit_L-clip336, ./llama3_vit_B-clip224-b16
                # 'ouput_dir': './demo', #./llama3_vit_L-clip336, ./llama3_vit_B-clip224-b16
                'stage_ckpt': '/ssd3/chih/LLM/MiniGPT-4-ckpt/checkpoint_stage3.pth', 
                'vis_root_train': './dataset/minigpt_casing_train/coco/image/train',
                'ann_paths_train': ['./dataset/minigpt_casing_train/coco_caption/defe_ready_anno.json'],
                'vis_root_valid': './dataset/minigpt_casing_test/coco/image/test',
                'ann_paths_valid': ['./dataset/minigpt_casing_test/coco_caption/defe_ready_anno.json']}

# llm_config = {'llama_model':'/ssd3/chih/LLM/Llama-2-7b-chat-hf', 'low_resource':True, 'low_res_device':0, 
#               'lora_r':64, 'lora_target_modules':["q_proj", "v_proj"], 'lora_alpha':16,'lora_dropout':0.05
#               }
llm_config = {'llama_model':'/ssd3/chih/LLM/Meta-Llama-3-8B-Instruct', 'low_resource':True, 'low_res_device':0, 
              'lora_r':64, 'lora_target_modules':["q_proj", "v_proj"], 'lora_alpha':16,'lora_dropout':0.05
              }
# '/ssd3/chih/LLM/Meta-Llama-3-8B-Instruct'


vit_config = {'model_name':'clip_large_336', #eva_clip_g, clip_large_336
            #   'model_path':"../../VITModel/clip-vit-base-patch16", #clip-vit-base-patch16, clip-vit-large-patch14-336
              'model_path':"../../VITModel/clip-vit-large-patch14-336", #clip-vit-base-patch16, clip-vit-large-patch14-336
            #   'image_size': 224,  #bilp2 = 448, clip = 224 or 336
              'image_size': 336,  #bilp2 = 448, clip = 224 or 336
              'drop_path_rate': 0, 'use_grad_checkpoint': True, 'vit_precision': 'fp16', 'freeze_vit': True, }

lr_config = {'init_lr': 1e-5, 'beta2':0.999,'min_lr': 1e-6, 'decay_rate': None, 'weight_decay':0.05,
                'warmup_start_lr': 1e-6, 'warmup_steps': 1000, 'iters_per_epoch': 1000}


def signal_handler(signal, frame):
    print("\nCaught interrupt signal. Cleaning up CUDA memory...")
    torch.cuda.empty_cache()
    sys.exit(0)
    
signal.signal(signal.SIGINT, signal_handler)

try:
    main_ = Main__(model_config=model_config)
    main_.VLM_build(llm_config=llm_config, vit_config=vit_config)
    print(main_.model)
    main_.main_process(max_epoch=20, lr_config=lr_config)
except KeyboardInterrupt:
    pass
finally:
    print("\nExiting program.")
