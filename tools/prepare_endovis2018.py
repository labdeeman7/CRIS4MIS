import os
import sys
import cv2
import json
import numpy as np

class2sents = {
    'background': ['background', 'body tissues', 'organs'],
    'instrument': ['instrument', 'medical instrument', 'tool', 'medical tool'],
    'shaft': [
        'shaft', 'instrument shaft', 'tool shaft', 'instrument body',
        'tool body', 'instrument handle', 'tool handle'
    ],
    'wrist': [
        'wrist', 'instrument wrist', 'tool wrist', 'instrument neck',
        'tool neck', 'instrument hinge', 'tool hinge'
    ],
    'claspers': [
        'claspers', 'instrument claspers', 'tool claspers', 'instrument head',
        'tool head'
    ],
    'bipolar_forceps': ['bipolar forceps'],
    'prograsp_forceps': ['prograsp forceps'],
    'large_needle_driver': ['large needle driver', 'needle driver'],
    'vessel_sealer': ['vessel sealer'],
    'grasping_retractor': ['grasping retractor'],
    'monopolar_curved_scissors': ['monopolar curved scissors'],
    'ultrasound_probe': ['ultrasound probe'],
    'suction_instrument': ['suction instrument'],
    'clip_applier': ['clip applier'],
}

labels = [{
    "name": "bipolar_forceps",
    "classid": 1
}, {
    "name": "prograsp_forceps",
    "classid": 2
}, {
    "name": "large_needle_driver",
    "classid": 3
}, {
    "name": "monopolar_curved_scissors",
    "classid": 4
}, {
    "name": "ultrasound_probe",
    "classid": 5
}, {
    "name": "suction_instrument",
    "classid": 6
}, {
    "name": "clip_applier",
    "classid": 7
}]

label_id2name = {x['classid']: x['name'] for x in labels}
label_name2id = {x['name']: x['classid'] for x in labels}

seq2path = {
    '1': '../train_val/miccai_challenge_2018_release_1/seq_1/labels',
    '2': '../train_val/miccai_challenge_2018_release_1/seq_2/labels',
    '3': '../train_val/miccai_challenge_2018_release_1/seq_3/labels',
    '4': '../train_val/miccai_challenge_2018_release_1/seq_4/labels',
    '5': '../train_val/miccai_challenge_release_2/seq_5/labels',
    '6': '../train_val/miccai_challenge_release_2/seq_6/labels',
    '7': '../train_val/miccai_challenge_release_2/seq_7/labels',
    '9': '../train_val/miccai_challenge_release_3/seq_9/labels',
    '10': '../train_val/miccai_challenge_release_3/seq_10/labels',
    '11': '../train_val/miccai_challenge_release_3/seq_11/labels',
    '12': '../train_val/miccai_challenge_release_3/seq_12/labels',
    '13': '../train_val/miccai_challenge_release_4/seq_13/labels',
    '14': '../train_val/miccai_challenge_release_4/seq_14/labels',
    '15': '../train_val/miccai_challenge_release_4/seq_15/labels',
    '16': '../train_val/miccai_challenge_release_4/seq_16/labels',
}


def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :,
                     0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


def get_one_sample(root_dir, image_file, image_path, save_dir, mask,
                   class_name):
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
        process_type = 'train'
    elif 'val' in root_dir:
        process_type = 'val'
    image_dir = os.path.join(root_dir, 'images')
    print('process: {} ...'.format(image_dir))
    cris_masks_dir = os.path.join(root_dir,
                                  'cris_{}_masks'.format(process_type))
    if not os.path.exists(cris_masks_dir):
        os.makedirs(cris_masks_dir)
    image_files = os.listdir(image_dir)
    image_files.sort()
    for image_file in image_files:
        print(image_file)
        image_path = os.path.join(image_dir, image_file)
        anno_path = image_path.replace('images', 'annotations')
        mask = cv2.imread(anno_path)
        # binary
        for class_id, class_name in enumerate(['background', 'instrument']):
            if class_id == 0:
                target_mask = (mask == 0) * 255
            elif class_id > 0:
                target_mask = (mask > 0) * 255
            if target_mask.sum() != 0:
                cris_data_list.append(
                    get_one_sample(root_dir, image_file, image_path,
                                   cris_masks_dir, target_mask, class_name))
        # parts
        _, seq_id, image_name = image_file.split('_')
        parts_mask_path = os.path.join(root_dir, seq2path[seq_id], image_name)
        parts_mask = cv2.imread(parts_mask_path)
        parts_mask = cv2.cvtColor(parts_mask, cv2.COLOR_BGR2RGB)
        parts_mask = rgb2id(parts_mask)
        for class_name in ['shaft', 'wrist', 'claspers']:
            if class_name == 'shaft':
                target_mask = (parts_mask
                               == (0 + 255 * 256 + 0 * 256 * 256)) * 255
            elif class_name == 'wrist':
                target_mask = (parts_mask
                               == (125 + 255 * 256 + 12 * 256 * 256)) * 255
            elif class_name == 'claspers':
                target_mask = (parts_mask
                               == (0 + 255 * 256 + 255 * 256 * 256)) * 255
            if target_mask.sum() != 0:
                cris_data_list.append(
                    get_one_sample(root_dir, image_file, image_path,
                                   cris_masks_dir, target_mask, class_name))
        # instruments
        for class_id, class_name in label_id2name.items():
            target_mask = (mask == class_id) * 255
            if target_mask.sum() != 0:
                cris_data_list.append(
                    get_one_sample(root_dir, image_file, image_path,
                                   cris_masks_dir, target_mask, class_name))

    with open(os.path.join(root_dir, cris_data_file), 'w') as f:
        json.dump(cris_data_list, f)


if __name__ == '__main__':
    # must add last "/"
    # /jmain02/home/J2AD019/exk01/zxz35-exk01/data/cambridge-1/EndoVis2018/train/
    root_dir = sys.argv[1]
    # cris_train.json
    cris_data_file = sys.argv[2]
    process(root_dir, cris_data_file)