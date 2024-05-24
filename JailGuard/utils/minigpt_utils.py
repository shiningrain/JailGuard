import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="2"
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr
import sys
sys.path.append('YOUR MINIGPT4 DIR')
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

from PIL import Image

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg_path", default='./utils/minigpt4_eval.yaml', help="path to configuration file. refer to https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/blob/main/minigpt_inference.py")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--gpu_id", type=str, default='0', help="specify the gpu to load the model.")
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True



def load_dirs(dir,method='txt'):
    # load files in target dir into a list
    output_list=[]
    name_list=[]
    for name in os.listdir(dir):
        path=os.path.join(dir,name)
        if os.path.isfile(path):
            if method=='txt':
                f=open(path,'r')
                line_list=f.readlines()
                f.close()
                output_list.append(line_list)
                name_list.append(name)
            elif method=='img':
                if '.bmp' in path or '.png' in path or '.jpg' in path:
                    output_list.append(path)
                    name_list.append(name)
    return output_list, name_list

def load_mask_dir(dir):
    output_list=[]
    name_list=[]
    for name in os.listdir(dir):
        path=os.path.join(dir,name)
        if os.path.isdir(path):

            output_list.append(path)
            name_list.append(name)
    return output_list, name_list

def filter_dirs(mask_dir_list,method_list,max=20):
    output_dir_list=[]
    output_name_list=[]
    count_dict={method:0 for method in method_list}
    for dirs in mask_dir_list:
        tmp_name=dirs.split('-')[-1]
        if tmp_name not in method_list:
            continue
        if count_dict[tmp_name]>=max:
            continue
        count_dict[tmp_name]+=1
        output_dir_list.append(dirs)
        output_name_list.append(os.path.basename(dirs))
    return output_dir_list,output_name_list

# ========================================
#             Model Initialization
# ========================================

def initialize_model():
    print('Initializing Chat')
    args = parse_args()
    cfg = Config(args)
    os.environ["CUDA_VISIBLE_DEVICES"]=f'{args.gpu_id}'
    model_config = cfg.model_cfg
    model_config.device_8bit = 0
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:0')

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
    print('Initialization Finished')
    return vis_processor,chat,model

# ========================================
#             Gradio Setting
# ========================================

def model_inference(vis_processor,chat,model,prompts_eval):
    # init chat state
    def upload_img(img):
        chat_state = CONV_VISION.copy()
        img_list = []
        chat.upload_img(img, chat_state, img_list)
        return chat_state, img_list


    def ask(user_message, chat_state):
        chat.ask(user_message, chat_state)
        return chat_state


    def answer(chat_state, img_list, num_beams=1, temperature=1.0):
        llm_message  = chat.answer(conv=chat_state,
                                img_list=img_list,
                                num_beams=num_beams,
                                temperature=temperature,
                                max_new_tokens=300,
                                max_length=2000)[0]

        return llm_message, chat_state, img_list
    image_path=prompts_eval[1]
    question=prompts_eval[0]
    img = Image.open(image_path).convert('RGB')
    img = vis_processor(img).unsqueeze(0).to(model.device)
    chat_state, img_list = upload_img(img)

    chat_state = ask(question, chat_state)
    llm_message, chat_state, img_list = answer(chat_state, img_list)
    return llm_message


# prompts_eval = []
# image_path_list=[]
# save_path_list=[]
# if not os.path.exists(args.save_dir):
#     os.makedirs(args.save_dir)

# select_amount=int(args.select_amount)
# mask_dir_list,name_list=load_mask_dir(args.prompt_dir)

# # # for small experiment, only test part of dirs TODO:delete
# # mask_dir_list,name_list=filter_dirs(mask_dir_list,args.method_list,max=999)

# # step1: query
# for i in range(len(mask_dir_list)):
#     mask_dir=mask_dir_list[i]
#     name=name_list[i]
#     new_save_dir=os.path.join(args.save_dir,name)
#     if not os.path.exists(new_save_dir):
#         os.makedirs(new_save_dir)
#     if name.split('-')[-1] not in args.method_list:
#         continue 

#     mask_file_list,mask_name_list=load_dirs(mask_dir,method='img')
#     # guarentee order of files
#     mask_name_list, mask_file_list = (list(t) for t in zip(*sorted(zip(mask_name_list,mask_file_list))))
#     question_path=os.path.join(mask_dir,'question')
#     f=open(question_path,'r')
#     question=f.readlines()
#     question=''.join(question)
#     f.close()
#     count=0
#     for j in range(len(mask_file_list)):
#         img_prompt_path=mask_file_list[j]
#         # image_path_list.append(img_prompt_path)
#         # img = Image.open(img_prompt_path).convert('RGB')
#         # img = vis_processor(img).unsqueeze(0).to(model.device)

#         prompts_eval.append([question,img_prompt_path])
#         save_name=mask_name_list[j].split('.')[0]#replace('.bmp','')
#         new_save_path=os.path.join(new_save_dir,save_name)
#         save_path_list.append(new_save_path)
#         count+=1

#         if count>=select_amount:
#             break


# print("######## Start Chatting ########")

# with torch.no_grad():

#     for i, content in enumerate(tqdm(prompts_eval)):

#         tmp=None
#         if os.path.exists(save_path_list[i]):
#             f=open(save_path_list[i],'r')
#             tmp=f.readlines()
#             f.close()
#             tmp=''.join(tmp)
#             # continue
#         run_count=0
#         while tmp==None or check_keyword(tmp):
#             print(save_path_list[i])
#             # print(tmp)
#             image_path=content[1]
#             question=content[0]
#             img = Image.open(image_path).convert('RGB')
#             img = vis_processor(img).unsqueeze(0).to(model.device)
#             chat_state, img_list = upload_img(img)

#             # print('################ Question %d ################' % (i+1))
#             chat_state = ask(question, chat_state)
#             llm_message, chat_state, img_list = answer(chat_state, img_list)
#             # print('>>> User:', question)
#             # print('\n')

#             # print('>>> LLM:\n')
#             # print(llm_message)
#             # print('\n\n')

#             tmp=llm_message
#             run_count+=1
#         if run_count!=0:
#             f=open(save_path_list[i],'w')
#             # print(save_path_list[i])
#             f.writelines(tmp)
#             f.close()