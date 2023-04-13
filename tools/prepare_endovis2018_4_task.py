import os
import sys
import cv2
import json
import numpy as np


class2sents = {
    'anatomy': {'classid': [0,4,5,10],
                   'sents': ['tissues', 'anatomy']},            
    'tool': {'classid': [1,2,3,6,7,8,9,11], 
                   'sents': [ 'tool', 'surgical tool']}, #😉 Does not exist itself as a class.
                    # it is a combination of shaft, wrist, clasper. This is debatable. It could also include thread and suction instrumet.
    
    'surgical_instrument': {'classid': [1,2,3], 
                   'sents': ['surgical robotic instrument', 'da vinci surgical instrument']}, 
    'instrument_shaft': {'classid': [1],
                         'sents': ['instrument shaft', 'shaft', 'tool shaft', 'instrument body',
                                   'tool body', 'instrument handle', 'tool handle']},
    'instrument_clasper': {'classid': [2],
                           'sents': ['instrument claspers', 'claspers', 'tool claspers', 'instrument head',
                                     'tool head']},                                  
    'instrument_wrist': {'classid': [3],
                         'sents': ['instrument wrist', 'wrist', 'tool wrist', 'instrument neck',
                                   'tool neck', 'instrument hinge', 'tool hinge']},
                                   
    'background_tissue': {'classid': [0],
                          'sents': ['background tissues', 'other tissues']},   
    'kidney_parenchyma': {'classid': [4],
                          'sents': ['kidney parenchyma', 'parenchyma', 'uncovered kidney']},
    'covered_kidney': {'classid': [5],
                       'sents': ['covered kidney', 'kidney sheath layer']},
    'small_intestine': {'classid': [10],
                        'sents': ['small intestine', 'bowels']},

    'thread': {'classid': [6],
               'sents': ['thread', 'surgical thread']},
    'clamps': {'classid': [7],
               'sents': ['clamps', 'surgical clamps', 'surgical clips']},
    'suturing_needle': {'classid': [8],
                        'sents': ['suturing needle', 'needle']},
    'suction_instrument': {'classid': [9],
                           'sents': ['suction instrument', 'suction']},
    'ultrasound_probe': {'classid': [11],    
                         'sents': ['ultrasound probe', 'ultrasound']},                       

}

factor = 20
binary_classification = ['anatomy', 'tool'] 
parts_classification = ['anatomy', 'instrument_shaft', 'instrument_wrist', 'instrument_clasper']
instruments_classification = ['anatomy', 'surgical_instrument', 'thread', 'clamps', 'suturing_needle', 'suction_instrument', 'ultrasound_probe']
anatomy_classification = ['background_tissue', 'kidney_parenchyma', 'covered_kidney', 'small_intestine']

def get_one_sample(root_dir, image_file, image_path, save_dir, mask, class_name):
    if '.jpg' in image_file:
        suffix = '.jpg'
    elif '.png' in image_file:
        suffix = '.png'
    mask_path = os.path.join(
        save_dir,
        image_file.replace(suffix, '') + '_{}.png'.format(class_name))
    cv2.imwrite(mask_path, mask)
    cris_data = {
        'img_path': image_path.replace(root_dir, ''),
        'mask_path': mask_path.replace(root_dir, ''),
        'num_sents': len(class2sents[class_name]),
        'sents': class2sents[class_name],
    }
    return cris_data


   

def process(root_dir, cris_data_file):
    cris_data_list = []
    if 'train' in root_dir:
        dataset_folder_array_num = [1,2,3,4,5,6,7,9,10,11,12,13,14,15,16] #😉 There is no seq_8 dataset for train
    elif 'val' in root_dir:
        dataset_folder_array_num = [1,2,3,4]
    for i in dataset_folder_array_num:
        image_dir = os.path.join(root_dir, 'seq_{}'.format(i),
                                 'left_frames')
        print('process: {} ...'.format(image_dir))
        cris_masks_dir = os.path.join(root_dir,
                                      'seq_{}'.format(i),
                                      'cris_masks') #😉 this is the cris masks folder for saving all masks and training. 
        if not os.path.exists(cris_masks_dir):
            os.makedirs(cris_masks_dir)
        image_files = os.listdir(image_dir)
        image_files.sort()
        for image_file in image_files:
            print(image_file)
            image_path = os.path.join(image_dir, image_file)
            mask_path = image_path.replace('left_frames','labels').replace('.png','_label_map.png')
            
            mask = cv2.imread(mask_path)
            mask = (mask / factor).astype(np.uint8)
            #binary
            for class_name in binary_classification:
                # print(class_name)
                class_ids = class2sents[class_name]['classid']
                target_mask = np.zeros_like(mask)
                for class_id in class_ids:
                    target_mask = np.logical_or(target_mask, (mask == class_id))
                target_mask = target_mask.astype(np.uint8) * 255  
                if target_mask.sum() != 0:  #😉 why is the mask only saved when there are examples? I guess for positive things only. 
                    cris_data_list.append(
                        get_one_sample(root_dir, image_file, image_path,
                                        cris_masks_dir, target_mask,
                                        class_name))
            # parts 
            for class_name in parts_classification:
                if class_name == 'anatomy':
                    continue
                class_ids = class2sents[class_name]['classid']
                target_mask = np.zeros_like(mask)

                for class_id in class_ids:
                    target_mask = np.logical_or(target_mask, (mask == class_id))
                target_mask = target_mask.astype(np.uint8) * 255  
                
                
                if target_mask.sum() != 0:
                    cris_data_list.append(
                        get_one_sample(root_dir, image_file, image_path,
                                       cris_masks_dir, target_mask,
                                       class_name))
            # instruments
            for class_name in instruments_classification:
                if class_name == 'anatomy':
                    continue
                class_ids = class2sents[class_name]['classid']
                target_mask = np.zeros_like(mask)
                for class_id in class_ids:
                    target_mask = np.logical_or(target_mask, (mask == class_id))
                target_mask = target_mask.astype(np.uint8) * 255  
                
                if target_mask.sum() != 0:
                    cris_data_list.append(
                        get_one_sample(root_dir, image_file, image_path,
                                       cris_masks_dir, target_mask,
                                       class_name))
                    
            # anatomy   
            for class_name in anatomy_classification:
                class_ids = class2sents[class_name]['classid']
                target_mask = np.zeros_like(mask)
                for class_id in class_ids:
                    target_mask = np.logical_or(target_mask, (mask == class_id))
                target_mask = target_mask.astype(np.uint8) * 255  
                                  
                if target_mask.sum() != 0:  #😉 why is the mask only saved when there are examples? I guess for positive things only. 
                    cris_data_list.append(
                        get_one_sample(root_dir, image_file, image_path,
                                        cris_masks_dir, target_mask,
                                        class_name))           
    
    with open(os.path.join(root_dir, cris_data_file), 'w') as f:
        json.dump(cris_data_list, f)


if __name__ == '__main__':
    # must add last "/"
    # /jmain02/home/J2AD019/exk01/zxz35_exk01/data/cambridge_1/EndoVis2017/cropped_test/
    root_dir = sys.argv[1]
    # cris_test.json
    cris_data_file = sys.argv[2]
    process(root_dir, cris_data_file)