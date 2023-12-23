import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import sys
sys.path.append('./utils')
from utils import *
import numpy as np
import uuid
from mask_utils import *
from tqdm import trange
from augmentations import *
import pickle
import spacy

def get_method(method_name): 
    try:
        method = text_aug_dict[method_name]
    except:
        print('Check your method!!')
        os._exit(0)
    return method


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mask Text Experiment')
    parser.add_argument('--mutator', default='TR', type=str, help='Random Replacement(RR),Random Insertion(RI),Targeted Replacement(TR),Targeted Insertion(TI),Random Deletion(RD),Synonym Replacement(SR),Punctuation Insertion(PI),Translation(TL),Rephrasing(RE)')
    parser.add_argument('--path', default='./demo_case/text/input/28-MasterKey-poc', type=str, help='path of the text input')
    parser.add_argument('--variant_save_dir', default='./demo_case/text/variant', type=str, help='dir to save the modify results')
    parser.add_argument('--response_save_dir', default='./demo_case/text/response', type=str, help='dir to save the modify results')
    parser.add_argument('--number', default='8', type=str, help='number of generated variants')
    parser.add_argument('--threshold', default=0.01, type=str, help='Threshold of divergence')
    args = parser.parse_args()

    number=int(args.number)


    if not os.path.exists(args.variant_save_dir):
        os.makedirs(args.variant_save_dir)

    # Step1: mask input
    path=args.path
    for i in range(number):
        tmp_method=get_method(args.method)
        f=open(path,'r')
        text_lines=f.readlines()
        # whole_text=''.join(text_lines)
        f.close()
        uid_name=str(uuid.uuid4())[:4]
        target_dir = args.variant_save_dir
        if len(os.listdir(target_dir))>=number+1: # skip if finished
            continue

        output_result = tmp_method(text_list=text_lines)
        target_path = os.path.join(target_dir,str(uuid.uuid4())[:6]+f'-{args.method}')
        f=open(target_path,'w')
        f.writelines(output_result)
        f.close()

    # Step2: query_model 
    variant_list, name_list= load_dirs(dir)
    new_save_dir=args.response_save_dir
    variant_list=[r'\n'.join(i) for i in variant_list]
    name_list, variant_list = (list(t) for t in zip(*sorted(zip(name_list,variant_list))))
    for j in range(len(variant_list)):
        prompt=variant_list[j]
        # read_file_in_line(mask_file_list[i])
        save_name=os.path.basename(variant_list[j])
        existing_response=[i for i in os.listdir(new_save_dir) if'.png' not in i]
        if len(existing_response)>=args.number:
            continue

        new_save_path=os.path.join(new_save_dir,save_name)
        if not os.path.exists(new_save_path):
            try:
                result = query_gpt('gpt-3.5-turbo',prompt,sleep=5)#TODO: make sure you have add your API key in this ./utils/config
                res_content=result['message'].content
            except openai.error.InvalidRequestError: # handle refusal
                res_content='I cannot assist with that!'
                print(f'Blocked in {new_save_path}')
            except:
                res_content='No response!' # handle exceptions
                print(f'NO response in {new_save_path}')
            
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