from mask_utils import *
import sys
from utils import *
import numpy as np
import torchvision.transforms as T



def remove_non_utf8(text):
    # Filter out non-UTF-8 characters
    utf8_chars = [char for char in text if ord(char) < 128]
    # Join the filtered characters to form a new string
    return ''.join(utf8_chars)

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
    from textaugment import AEDA
    whole_text=''.join(text_list)
    if misc==None:
        t = AEDA()
    else:
        t = misc
    try:
        whole_text=t.punct_insertion(whole_text)
    except ValueError as e:
        print(e)
        whole_text=whole_text
    output_list=whole_text.split('\n')
    output_list=[output+'\n' for output in output_list]
    return output_list

def translate_text(text_list,level=10,misc="en"):
    from textaugment import Translate
    rate=sample_int_level(level,0)
    target_list=['ru','fr','de','el','id','it','ja','ko','la','pl']
    whole_text=''.join(text_list)
    whole_text=remove_non_utf8(whole_text)

    t = Translate(src=misc, to=target_list[rate])
    try:
        whole_text=t.augment(whole_text)
    except Exception as e:
        print(e)
        whole_text=whole_text
    # whole_text=t.augment(whole_text)
    output_list=whole_text.split('\n')
    output_list=[output+'\n' for output in output_list]
    return output_list

def find_index(L, b):
    for i in range(len(L) - 1):
        if b > L[i] and b < L[i + 1]:
            return i
    return -1

def policy_aug_text(text_list,level='0.24-0.52-0.24',pool='PI-TI-TL'):
    mutator_list=[text_aug_dict[_mut] for _mut in pool.split('-')]
    probability_list=[float(_value) for _value in level.split('-')]
    probability_list=[sum(probability_list[:i]) for i in range(len(level))]
    randnum=np.random.random()
    index=find_index(probability_list,randnum)
    print(index,randnum)
    output_list=mutator_list[index](text_list)
    return output_list

'''=======================Image Level Mask======================='''

def mask_image(img,level=None,position='rand',mask_type='r',mask_size=(200,200)):
    img_size=img.size # width, height
    new_mask_size=[min(mask_size[sz],(0.3*img_size[sz])) for sz in range(len(mask_size))]
    # mask size should not larget than 0.5*image size
    position=load_position(position,img_size,new_mask_size)
    # generate mask
    new_image = generate_mask_pil(img,mask_type,new_mask_size,position)
    # new_image = generate_dark_mask(img,mask_type,mask_size,position)
    return new_image

def blur_image(img,level=5):
    # level: int: 5 --> kernel size
    k1=sample_odd_level(level)
    k2=sample_odd_level(level)
    transform = T.GaussianBlur(kernel_size=(k1, k2))
    new_image = transform(img)
    return new_image

def flip_image(img,level=1.0):
    # level: float: 1.0 --> p
    p=sample_float_level(level)
    transform = T.RandomHorizontalFlip(p=p)
    new_image = transform(img)
    return new_image

def vflip_image(img,level=1.0):
    # level: float: 1.0 --> p
    p=sample_float_level(level)
    transform = T.RandomVerticalFlip(p=p)
    new_image = transform(img)
    return new_image

def resize_crop_image(img,level=500):
    # level: int: 500 --> image size
    size=img.size
    s1=int(max(sample_int_level(level),0.8*size[0]))
    s2=int(max(sample_int_level(level),0.8*size[1]))
    transform = T.RandomResizedCrop((s2, s1),scale=(0.9,1))
    new_image = transform(img)
    return new_image

def gray_image(img,level=1):
    rate=sample_float_level(level)
    if rate >=0.5:
        transform = T.Grayscale(num_output_channels=len(img.split()))
        new_image = transform(img)
    else:
        new_image = img
    return new_image

def rotation_image(img,level=180):
    rate=sample_float_level(level)
    transform = T.RandomRotation(degrees=(0, rate))
    new_image = transform(img)
    return new_image

def colorjitter_image(img,level1=1,level2=0.5):
    rate1=sample_float_level(level1)
    rate2=sample_float_level(level2)
    transform = T.ColorJitter(brightness=rate1, hue=rate2)
    new_image = transform(img)
    return new_image

def solarize_image(img,level=200):
    rate=sample_float_level(level)
    transform = T.RandomSolarize(threshold=rate)
    new_image = transform(img)
    return new_image

def posterize_image(img,level=3):
    rate=sample_int_level(level)
    transform = T.RandomPosterize(bits=rate)
    new_image = transform(img)
    return new_image

def policy_aug_image(img,level='0.34-0.45-0.21',pool='RR-BL-RP'):
    mutator_list=[img_aug_dict[_mut] for _mut in pool.split('-')]
    probability_list=[float(_value) for _value in level.split('-')]
    probability_list=[sum(probability_list[:i]) for i in range(len(level))]
    randnum=np.random.random()
    index=find_index(probability_list,randnum)
    
    new_image=mutator_list[index](img)
    return new_image


# ===============================================
text_aug = [
    rand_replace_text, rand_add_text, target_replace_text, target_add_text,
    rand_del_text,synonym_replace_text,aeda_punc_text,translate_text
    ]

text_aug_dict={
    'RR':rand_replace_text,
    'RI':rand_add_text,
    'TR':target_replace_text,
    'TI':target_add_text,
    'RD':rand_del_text,
    'SR':synonym_replace_text,
    'PI':aeda_punc_text,
    'TL':translate_text,
    'PL':policy_aug_text,
}

img_aug = [
    mask_image, blur_image, flip_image, resize_crop_image, gray_image, rotation_image, colorjitter_image,
    vflip_image,solarize_image,posterize_image
]
img_aug_dict={
    'RM':mask_image,
    'BL':blur_image,
    'HF':flip_image,
    'CR':resize_crop_image,
    'GR':gray_image,
    'RR':rotation_image,
    'CJ':colorjitter_image,
    'VF':vflip_image,
    'RS':solarize_image,
    'RP':posterize_image,
    'PL':policy_aug_image,
}
translate_text(['We observe that the detection accuracy of the \sys\'s mutators do not drop significantly when the LLM query budgets (\ie, the number of generated variants) reduce from the LLM query budget \(N=8\) to \(N=4\) and is always better than that of the best baseline SmoothLLM. This finding can provide guidance on attack detection and defense in low-budget scenarios.'])