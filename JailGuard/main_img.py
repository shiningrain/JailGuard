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
from minigpt_utils import initialize_model, model_inference

def get_method(method_name): 
    try:
        method = img_aug_dict[method_name]
        method = img_aug_dict[method_name]
    except:
        print('Check your method!!')
        os._exit(0)
    return method


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mask Text Experiment')
    parser.add_argument('--mutator', default='PL', type=str, help='Horizontal Flip(HF),Vertical Flip(VF),Random Rotation(RR),Crop and Resize(CR),Random Mask(RM),Random Solarization(RS),Random Grayscale(GR),Gaussian Blur(BL), Colorjitter(CJ), Random Posterization(RP) Policy(PL)')
    parser.add_argument('--serial_num', default='287', type=str, help='the serial number of the data under test in the dataset')
    parser.add_argument('--path', default='../dataset/image/dataset', type=str, help='dataset path')
    parser.add_argument('--variant_save_dir', default='./demo_case/variant', type=str, help='dir to save the modify results')
    parser.add_argument('--response_save_dir', default='./demo_case/response', type=str, help='dir to save the modify results')
    parser.add_argument('--number', default='8', type=str, help='number of generated variants')
    parser.add_argument('--threshold', default=0.025, type=str, help='Threshold of divergence')
    args = parser.parse_args()

    number=int(args.number)


    if not os.path.exists(args.variant_save_dir):
        os.makedirs(args.variant_save_dir)
    target_dir=args.variant_save_dir

    # Step1: mask input
    data_path=os.path.join(args.path,args.serial_num)
    image_path=os.path.join(data_path,'image.bmp')
    if not os.path.exists(image_path):
        image_path=os.path.join(data_path,'image.jpg')
    for i in range(number):
        tmp_method=get_method(args.mutator)
        pil_img = Image.open(image_path)
        new_image=tmp_method(img=pil_img)

        # save image
        if '.bmp' in image_path:
            target_path = os.path.join(target_dir,str(i)+f'-{args.mutator}.bmp')
        else:
            target_path = os.path.join(target_dir,str(i)+f'-{args.mutator}.jpg')
        if len(os.listdir(target_dir))>=number+1:
            break # early stop
        # cv2.imwrite()
        # creating a image object (main image)  
        new_image.save(target_path)
        target_question_path=os.path.join(target_dir,'question')
        if not os.path.exists(target_question_path):
            shutil.copy(os.path.join(data_path,'question'),target_question_path)

    # Step2: query_model
    vis_processor,chat,model=initialize_model()# initialize minigpt-4 model.refer to https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/blob/main/minigpt_inference.py

    variant_list, name_list= load_mask_dir(target_dir)
    question_path=os.path.join(target_dir,'question')
    f=open(question_path,'r')
    question=f.readlines()
    question=''.join(question)
    f.close()
    new_save_dir=args.response_save_dir
    if not os.path.exists(new_save_dir):
        os.makedirs(new_save_dir)
    for j in range(len(variant_list)):
        img_prompt_path=variant_list[j]
        prompts_eval=[question,img_prompt_path]
        # read_file_in_line(mask_file_list[i])
        save_name=name_list[j].split('.')[0]
        existing_response=[i for i in os.listdir(new_save_dir) if'.png' not in i]
        if len(existing_response)>=number:
            continue
        new_save_path=os.path.join(new_save_dir,save_name)
        if not os.path.exists(new_save_path):

            result = model_inference(vis_processor,chat,model,prompts_eval)

            f=open(new_save_path,'w')
            f.writelines(result)
            f.close()

    # Step3: divergence & detect
    diver_save_path=os.path.join(args.response_save_dir,f'diver_result-{args.number}.pkl')
    metric = spacy.load("en_core_web_md")
    avail_dir=args.response_save_dir
    check_list=os.listdir(avail_dir)
    check_list=[os.path.join(avail_dir,check) for check in check_list]
    output_list=read_file_list(check_list)
    max_div,jailbreak_keywords=update_divergence(output_list,args.serial_num,avail_dir,select_number=number,metric=metric,top_string=100)
    
    detection_result=detect_attack(max_div,jailbreak_keywords,args.threshold)
    if detection_result:
        print('The Input is a Attack Query!!')
    else:
        print('The Input is a Benign Query!!')