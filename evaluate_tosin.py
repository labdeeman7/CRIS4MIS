import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

original_height, original_width = 1080, 1920
height, width = 1024, 1280
h_start, w_start = 28, 320

binary_factor = 255
parts_factor = 85
instruments_factor = 32


def general_jaccard(y_true, y_pred):
    result = []

    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    for instrument_id in set(y_true.flatten()):
        if instrument_id == 0:
            continue
        result += [jaccard(y_true == instrument_id, y_pred == instrument_id)]

    return np.mean(result)


def general_dice(y_true, y_pred):
    result = []

    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    for instrument_id in set(y_true.flatten()):
        if instrument_id == 0:
            continue
        result += [dice(y_true == instrument_id, y_pred == instrument_id)]

    return np.mean(result)


def jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)


def dice(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() +
                                                    y_pred.sum() + 1e-15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--test_path',
        type=str,
        default=
        '/jmain02/home/J2AD019/exk01/zxz35-exk01/data/cambridge-1/EndoVis2017/cropped_test',
        help='path where train images with ground truth are located')
    arg('--pred_path',
        type=str,
        default=
        '/jmain02/home/J2AD019/exk01/zxz35-exk01/data/cambridge-1/CRIS/exp/endovis2017/v1_0/score',
        help='path with predictions')
    arg('--problem_type',
        type=str,
        default='parts',
        choices=['binary', 'parts', 'instruments', 'anatomy', 'endovis_2018_style'])
    arg('--vis', action='store_true')
    args = parser.parse_args()

    result_dice = []
    result_jaccard = []

    factor = 20

    if args.problem_type == 'binary':
        class_name_list = ['anatomy', 'tool'] 

    elif args.problem_type == 'parts':
        class_name_list = ['anatomy', 'instrument_shaft', 'instrument_wrist', 'instrument_clasper']
        
    elif args.problem_type == 'instruments':
        class_name_list = ['anatomy', 'surgical_instrument', 'thread', 'clamps', 'suturing_needle', 'suction_instrument', 'ultrasound_probe']

    elif args.problem_type == 'anatomy':
        class_name_list = ['background_tissue', 'kidney_parenchyma', 'covered_kidney', 'small_intestine']

    elif args.problem_type == 'endovis_2018_style':
        class_name_list =  ["background_tissue", "instrument_shaft", "instrument_clasper", "instrument_wrist", "kidney_parenchyma", "covered_kidney", 
                                        "thread", "clamps","suturing_needle", "suction_instrument", "small_intestine", "ultrasound_probe" ]    

    # palette
    if args.vis:
        eval_dir = os.path.join(args.pred_path.replace('/score', '/eval_vis'))
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)
        palette_list = [(255, 128, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255),
                        (0, 255, 255), (255, 0, 255), (255, 255, 0),
                        (0, 128, 255)]
        palette = np.zeros((8, height, width, 3))
        for i in range(8):
            for j in range(3):
                palette[i][:, :, j] = palette_list[i][j]

    # evaluate
    if 'train' in args.test_path:
        dataset_folder_array_num = [1,2,3,4,5,6,7,9,10,11,12,13,14,15,16]
    elif 'val' in args.test_path:
        dataset_folder_array_num = [1,2,3,4]
    for folder_id in dataset_folder_array_num:
        seq_folder_name = 'seq_{}'.format(folder_id)
        file_dir = os.path.join(args.test_path, seq_folder_name,
                                'cris_masks')
        
        image_dir = os.path.join(args.test_path, seq_folder_name,
                                     'left_frames')
        for file_name in tqdm(os.listdir(image_dir),desc=seq_folder_name):
            file_id = file_name.split('.')[0]

            pred_image_list = []
            gt_image_list = []
            # print(class_name_list)
            for class_name in class_name_list:
                # print(class_name)
                
                if class_name == 'background_tissue': 
                    pred_class_name = 'background_tissues'  #😉 Made a mistake in preparation of data.     
                elif class_name == 'instrument_clasper':
                    pred_class_name = 'instrument_claspers' 
                elif class_name == 'surgical_instrument':    
                    pred_class_name = 'surgical_robotic_instrument'
                elif class_name == 'anatomy':
                    pred_class_name = 'tissues'    
                else:
                    pred_class_name = class_name


                pred_file_name = os.path.join(
                    args.pred_path,
                    'score-{}-{}-{}.npz'.format(seq_folder_name,
                                                file_id, pred_class_name))
                gt_file_path = os.path.join(file_dir, file_id + '_' + class_name + '.png' )
                
                if os.path.exists(pred_file_name):
                    # print("i reached here and used an actual groundtruth")
                    # print(gt_file_path)
                    gt_image = cv2.imread(gt_file_path, 0).astype(np.uint8) #😉 I have multiple y_trues.
                    pred_dict = np.load(pred_file_name)
                    pred_image = cv2.warpAffine(pred_dict.get('pred'),
                                                pred_dict.get('mat'),
                                                (width, height),
                                                flags=cv2.INTER_CUBIC,
                                                borderValue=0.)
                else:
                    # print("I just used zeros")
                    image_path = os.path.join(image_dir,'{}.png'.format(file_id))                  
                    image_reference = cv2.imread(image_path, 0)
                    pred_image = np.zeros_like(image_reference)
                    gt_image = np.zeros_like(image_reference)

  

                pred_image_list.append(pred_image)
                gt_image_list.append(gt_image)

            pred_image = np.array(pred_image_list)
            gt_image = np.array(gt_image_list)



            y_pred = np.argmax(pred_image, axis=0)
            y_true = np.argmax(gt_image, axis=0)
    

            if args.vis:
                image_path = os.path.join(image_dir,'{}.png'.format(file_id))   
                image = cv2.imread(image_path)

                gt_mask = y_true 
                show = np.zeros_like(image)
                for i_h in range(height):
                    for i_w in range(width):
                        show[i_h, i_w] = palette[gt_mask[i_h, i_w], i_h, i_w]
                # show = np.take(palette, gt_mask)
                gt_vis_image = image * 0.5 + show * 0.5

                show = np.zeros_like(image)
                for i_h in range(height):
                    for i_w in range(width):
                        show[i_h, i_w] = palette[y_pred[i_h, i_w], i_h, i_w]
                pred_vis_image = image * 0.5 + show * 0.5

                vis_image = np.concatenate([gt_vis_image, pred_vis_image],
                                           axis=1)
                cv2.imwrite(
                    '{}/{}_{}_{}.jpg'.format(eval_dir, args.problem_type,
                                             seq_folder_name, file_id),
                    vis_image)


            result_jaccard += [general_jaccard(y_true, y_pred)]
            result_dice += [general_dice(y_true, y_pred)]

    print('Jaccard (IoU): mean={:.2f}, std={:.4f}'.format(
        np.mean(result_jaccard) * 100, np.std(result_jaccard)))
    print('Dice: mean={:.2f}, std={:.4f}'.format(
        np.mean(result_dice) * 100, np.std(result_dice)))
    
    # print(f'iou is {result_jaccard}' )
    # print(f'result_dice is {result_dice}' )
