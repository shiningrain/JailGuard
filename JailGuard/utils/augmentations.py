from mask_utils import *
import sys
from utils import *
import numpy as np
from baseline_utils import get_safety_checker,llama_check_text
from textaugment import EDA,AEDA,Word2vec, Translate

def sample_float_level(maxl,minl=0.1):
  return np.random.uniform(low=minl, high=maxl)

def sample_int_level(maxl,minl=1):
  return np.random.randint(low=minl, high=maxl)

def sample_odd_level(maxl,minl=1):
  return 2*np.random.randint(low=minl, high=maxl)-1

'''=======================Text Level Mask======================='''


def rand_replace_text(text_list,level=0.01):
    #  level: None
    rate=0.5*level#sample_float_level(level,0)
    output_list=mask_text(text_list,rate,method='replace',mode='random')
    return output_list

def target_replace_text(text_list,level=0.01):
    #  level: None
    rate=0.5*level#sample_float_level(level,0)
    output_list=mask_text(text_list,rate,method='replace',mode='heat')
    return output_list

def rand_add_text(text_list,level=0.01):
    #  level: None
    rate=0.5*level#sample_float_level(level,0)
    output_list=mask_text(text_list,rate,method='add',mode='random')
    return output_list

def target_add_text(text_list,level=0.01): 
    #  level: None
    rate=0.5*level#sample_float_level(level,0)
    output_list=mask_text(text_list,rate,method='add',mode='heat')
    return output_list

def sm_swap_text(text_list,level=0.2,fix=False): # random replace q% 
    #  level: None
    rate=sample_float_level(level,0)
    if fix: rate=level
    whole_text=''.join(text_list)
    amount=int(len(whole_text)*rate)
    whole_text=sm_process(whole_text,amount,'swap')
    output_list=whole_text.split('\n')
    output_list=[output+'\n' for output in output_list]
    return output_list

def sm_insert_text(text_list,level=0.2,fix=False): 
    #  level: None
    rate=sample_float_level(level,0)
    if fix: rate=level
    whole_text=''.join(text_list)
    amount=int(len(whole_text)*rate)
    whole_text=sm_process(whole_text,amount,'insert')
    output_list=whole_text.split('\n')
    output_list=[output+'\n' for output in output_list]
    return output_list

def sm_patch_text(text_list,level=0.2,fix=False):  #random replace q% consecutive characters
    #  level: None
    rate=sample_float_level(level,0)
    if fix: rate=level
    whole_text=''.join(text_list)
    amount=int(len(whole_text)*rate)
    whole_text=sm_process(whole_text,amount,'patch')
    output_list=whole_text.split('\n')
    output_list=[output+'\n' for output in output_list]
    return output_list

def rephrase_text(text_list,level=None,model='gpt-3.5-turbo'):
    # input text_list with '\n' at the end of each piece of text
    whole_text=''.join(text_list)
    prompt="Please rephrase the following paragraph while ensuring its core semantics and contents unchanged. \n The paragraph is :`"+whole_text+'`'
    result = query_gpt(model,prompt,sleep=3)
    rephrase_result=result['message'].content
    output_list=rephrase_result.split('\n')
    output_list=[line+'\n' for line in output_list]
    return output_list


def synonym_replace_text(text_list,level=20):
    rate=sample_int_level(level,0)
    whole_text=''.join(text_list)
    length=len(whole_text.split(' '))
    rate=min(int(length/3),rate)
    whole_text=synonym_replacement(whole_text,rate)
    output_list=whole_text.split('\n')
    output_list=[output+'\n' for output in output_list]
    return output_list

def rand_del_text(text_list,level=0.01):
    #  level: None
    rate=0.5*level#sample_float_level(level,0)
    whole_text=''.join(text_list)
    whole_text=random_deletion(whole_text,rate)
    output_list=whole_text.split('\n')
    output_list=[output+'\n' for output in output_list]
    return output_list

def aeda_punc_text(text_list,level=None,misc=None):
    whole_text=''.join(text_list)
    if misc==None:
        t = AEDA()
    else:
        t = misc
    whole_text=t.punct_insertion(whole_text)
    output_list=whole_text.split('\n')
    output_list=[output+'\n' for output in output_list]
    return output_list

def translate_text(text_list,level=10,misc="en"):
    rate=sample_int_level(level,0)
    target_list=['ru','fr','de','el','id','it','ja','ko','la','pl']
    whole_text=''.join(text_list)

    t = Translate(src=misc, to=target_list[rate])
    whole_text=t.augment(whole_text)
    output_list=whole_text.split('\n')
    output_list=[output+'\n' for output in output_list]
    return output_list

'''=======================Image Level Mask======================='''

def mask_image(img,level=None,position='rand',mask_type='r',mask_size=(200,200)):
    img_size=img.size # width, height
    new_mask_size=[min(mask_size[sz],(0.5*img_size[sz])) for sz in range(len(mask_size))]
    # mask size should not larget than 0.5*image size
    position=load_position(position,img_size,new_mask_size)
    # generate mask
    new_image = generate_mask_pil(img,mask_type,new_mask_size,position)
    # new_image = generate_dark_mask(img,mask_type,mask_size,position)
    return new_image

def blur_image(img,level=5):
    import torchvision.transforms as T
    # level: int: 5 --> kernel size
    k1=sample_odd_level(level)
    k2=sample_odd_level(level)
    transform = T.GaussianBlur(kernel_size=(k1, k2))
    new_image = transform(img)
    return new_image

def flip_image(img,level=1.0):
    import torchvision.transforms as T
    # level: float: 1.0 --> p
    p=sample_float_level(level)
    transform = T.RandomHorizontalFlip(p=p)
    new_image = transform(img)
    return new_image

def resize_crop_image(img,level=1000):
    import torchvision.transforms as T
    # level: int: 500 --> image size
    size=img.size
    s1=max(sample_int_level(level),0.8*size[0])
    s2=max(sample_int_level(level),0.8*size[1])
    transform = T.RandomResizedCrop((s2, s1),scale=(0.9,1))
    new_image = transform(img)
    return new_image

def gray_image(img,level=1):
    import torchvision.transforms as T
    rate=sample_float_level(level)
    if rate >=0.5:
        transform = T.Grayscale(num_output_channels=len(img.split()))
        new_image = transform(img)
    else:
        new_image = img
    return new_image

# text_aug = [
#     rephrase_text, rand_replace_text, rand_add_text, target_replace_text, target_add_text,sm_insert_text,
#     sm_patch_text,sm_swap_text,rand_del_text,synonym_replace_text,aeda_punc_text,translate_text
#     ]

text_aug = [
    rephrase_text, rand_replace_text, rand_add_text, target_replace_text, target_add_text,
    rand_del_text,synonym_replace_text,aeda_punc_text,translate_text
    ]

text_aug_dict={
    'RR':rand_replace_text,
    'RI':rand_add_text,
    'TR':target_replace_text,
    'TI':target_add_text,
    'RE':rephrase_text,
    'RD':rand_del_text,
    'SR':synonym_replace_text,
    'PI':aeda_punc_text,
    'TL':translate_text,
}

img_aug = [
    mask_image, blur_image, flip_image, resize_crop_image, gray_image
]

img_aug_dict={
    'RR':rand_replace_text,
    'RI':rand_add_text,
    'TR':target_replace_text,
    'TI':target_add_text,
    'RE':rephrase_text,
    'RD':rand_del_text,
    'SR':synonym_replace_text,
    'PI':aeda_punc_text,
    'TL':translate_text,
}

