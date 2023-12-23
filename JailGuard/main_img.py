import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import sys
sys.path.append('./utils')
from minigpt_utils import query_minigpt,load_minigpt4
from utils import *
import numpy as np
import uuid
from mask_utils import *
from tqdm import trange
from augmentations import *
import pickle
import spacy
from PIL import Image
import shutil

def get_method(method_name): 
    try:
        method = img_aug_dict[method_name]
    except:
        print('Check your method!!')
        os._exit(0)
    return method


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mask Text Experiment')
    parser.add_argument('--mutator', default='RR', type=str, help='Random Mask(RM),Gaussian Blur(GB),Horizontal Flip(HF),Vertical Flip(VF),Crop and Resize(CR),Random Grayscale(RG),Random Rotation(RR),Colorjitter(CJ),Random Solarization(RS),Random Posterization(RP)')
    parser.add_argument('--path', default='./demo_case/img/input', type=str, help='dir of image input and questions')
    parser.add_argument('--variant_save_dir', default='./demo_case/img/variant', type=str, help='dir to save the modify results')
    parser.add_argument('--response_save_dir', default='./demo_case/img/response', type=str, help='dir to save the modify results')
    parser.add_argument('--number', default='8', type=str, help='number of generated variants')
    parser.add_argument('--threshold', default=0.0025, type=str, help='Threshold of divergence')
    args = parser.parse_args()

    number=int(args.number)


    if not os.path.exists(args.variant_save_dir):
        os.makedirs(args.variant_save_dir)

    # Step1: mask input
    path=args.path
    for i in range(number):
        tmp_method=get_method(args.method)
        image_path=os.path.join(args.path,'image.bmp')
        question_path=os.path.join(args.path,'question')
        image = Image.open(image_path)
        
        uid_name=str(uuid.uuid4())[:4]
        target_dir = args.variant_save_dir
        if len(os.listdir(target_dir))>=number+1: # skip if finished
            continue

        output_img = tmp_method(image)
        target_path = os.path.join(target_dir,str(uuid.uuid4())[:6]+'.bmp')
        output_img.save(target_path)
    new_question_path=os.path.join(target_dir,'question')
    if not os.path.exists(new_question_path):
        shutil.copy(question_path,new_question_path)
     

    # Step2: query_model 
    variant_list, name_list= load_dirs_images(args.variant_save_dir)
    question_path=os.path.join(args.variant_save_dir,'question')
    f=open(question_path,'r')
    prompt_list=f.readlines()
    f.close()
    prompt=''.join(prompt_list)
    new_save_dir=args.response_save_dir
    if not os.path.exists(new_save_dir):
        os.makedirs(new_save_dir)
    name_list, variant_list = (list(t) for t in zip(*sorted(zip(name_list,variant_list))))
    minigpt_chat=load_minigpt4()#TODO: make sure you have replace the 'YOUR_XXX_PATH' in the ./utils/minigpt_utils.py
    for j in range(len(variant_list)):
        image=variant_list[j]
        save_name=name_list[j].split('.')[0]
        existing_response=[i for i in os.listdir(new_save_dir)]
        if len(existing_response)>=number:
            continue

        new_save_path=os.path.join(new_save_dir,save_name)
        if not os.path.exists(new_save_path):

            res_content = query_minigpt(prompt,image,minigpt_chat)
            
            f=open(new_save_path,'w')
            f.writelines(res_content)
            f.close()

    # Step3: divergence & detect
    diver_save_path=os.path.join(args.response_save_dir,f'diver_result-{args.number}.pkl')
    metric = spacy.load("en_core_web_md")
    avail_dir=args.response_save_dir
    check_list=os.listdir(avail_dir)
    check_list=[os.path.join(avail_dir,check) for check in check_list]
    output_list=read_file_list(check_list)
    max_div,jailbreak_keywords=update_divergence(output_list,os.path.basename(args.path),avail_dir,diver_save_path,select_number=number,metric=metric)
    
    detection_result=detect_attack(max_div,jailbreak_keywords,args.threshold)
    if detection_result:
        print('The Input is a Jailbreak Query!!')
    else:
        print('The Input is a Benign Query!!')