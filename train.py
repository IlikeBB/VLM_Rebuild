import os, sys
import torch
import signal
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from main import Main__

model_config = {'amp':True, 'use_distributed':False,'accum_grad_iters':1,
                'chat_template': True, 'end_sym': '\n', 'prompt_template': "[INST] {} [/INST]",
                'max_txt_len': 1024, 'max_context_len': 3500,
                'ouput_dir': './exper01_llama3',
                'stage_ckpt': '/ssd3/chih/LLM/MiniGPT-4-ckpt/checkpoint_stage3.pth', 
                'vis_root_train': './dataset/minigpt_casing_train/coco/image/train',
                'ann_paths_train': ['./dataset/minigpt_casing_train/coco_caption/defe_ready_anno.json'],
                'vis_root_valid': './dataset/minigpt_casing_test/coco/image/test',
                'ann_paths_valid': ['./dataset/minigpt_casing_test/coco_caption/defe_ready_anno.json']}

# llm_config = {'llama_model':'/ssd3/chih/LLM/Llama-2-7b-chat-hf', 'low_resource':False, 'low_res_device':0, 
#               'lora_r':64, 'lora_target_modules':["q_proj", "v_proj"], 'lora_alpha':16,'lora_dropout':0.05
#               }
llm_config = {'llama_model':'/ssd3/chih/LLM/Meta-Llama-3-8B-Instruct', 'low_resource':True, 'low_res_device':0, 
              'lora_r':64, 'lora_target_modules':["q_proj", "v_proj"], 'lora_alpha':16,'lora_dropout':0.05
              }
# '/ssd3/chih/LLM/Meta-Llama-3-8B-Instruct'


vit_config = {'model_name':'eva_clip_g', 'image_size': 448,  'drop_path_rate': 0, 'use_grad_checkpoint': True, 'vit_precision': 'fp16', 'freeze_vit': True, }

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
