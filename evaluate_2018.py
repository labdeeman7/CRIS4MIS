import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

original_height, original_width = 1080, 1920
height, width = 1024, 1280
h_start, w_start = 28, 320

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


def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :,
                     0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--test_path',
        type=str,
        default=
        '/jmain02/home/J2AD019/exk01/zxz35-exk01/data/cambridge-1/EndoVis2018/val',
        help='path where train images with ground truth are located')
    arg('--pred_path',
        type=str,
        default=
        '/jmain02/home/J2AD019/exk01/zxz35-exk01/data/cambridge-1/CRIS/exp/endovis2018/v1_0/score',
        help='path with predictions')
    arg('--problem_type',
        type=str,
        default='parts',
        choices=['binary', 'parts', 'instruments'])
    arg('--vis', action='store_true')
    args = parser.parse_args()

    result_jaccard = []
    result_dice = []

    if args.problem_type == 'binary':
        class_name_list = ['background', 'instrument']
    elif args.problem_type == 'parts':
        class_name_list = ['background', 'shaft', 'wrist', 'claspers']
    elif args.problem_type == 'instruments':
        class_name_list = [
            'background',
            'bipolar_forceps',
            'prograsp_forceps',
            'large_needle_driver',
            'monopolar_curved_scissors',
            'ultrasound_probe',
            'suction_instrument',
            'clip_applier',
        ]

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
    image_dir = os.path.join(args.test_path, 'images')
    for image_file in tqdm(os.listdir(image_dir)):
        image_id = image_file.split('.')[0]
        image_path = os.path.join(image_dir, image_file)
        anno_path = image_path.replace('images', 'annotations')
        mask = cv2.imread(anno_path)

        # generate mask label
        if args.problem_type == 'binary':
            y_true = (mask > 0) * 1
        elif args.problem_type == 'parts':
            _, seq_id, image_name = image_file.split('_')
            parts_mask_path = os.path.join(args.test_path, seq2path[seq_id],
                                           image_name)
            parts_mask = cv2.imread(parts_mask_path)
            parts_mask = cv2.cvtColor(parts_mask, cv2.COLOR_BGR2RGB)
            parts_mask = rgb2id(parts_mask)
            y_true = np.zeros(parts_mask.shape)
            for class_name in ['shaft', 'wrist', 'claspers']:
                if class_name == 'shaft':
                    y_true += (parts_mask
                               == (125 + 255 * 256 + 12 * 256 * 256)) * 1
                elif class_name == 'wrist':
                    y_true += (parts_mask
                               == (0 + 255 * 256 + 0 * 256 * 256)) * 2
                elif class_name == 'claspers':
                    y_true += (parts_mask
                               == (0 + 255 * 256 + 255 * 256 * 256)) * 3
        elif args.problem_type == 'instruments':
            y_true = mask

        # vis
        if args.vis:
            image = cv2.imread(image_path)
            gt_mask = y_true
            show = np.zeros_like(image)
            for i_h in range(height):
                for i_w in range(width):
                    show[i_h, i_w] = palette[gt_mask[i_h, i_w], i_h, i_w]
            gt_vis_image = image * 0.5 + show * 0.5

        image_split = '_'.join(image_file.split('_')[:2])
        file_id = image_file.split('_')[2]
        pred_image_list = []
        for class_name in class_name_list:
            pred_file_name = os.path.join(
                args.pred_path,
                'score-{}-{}-{}.npz'.format(image_split, file_id, class_name))
            if os.path.exists(pred_file_name):
                pred_dict = np.load(pred_file_name)
                pred_image = cv2.warpAffine(pred_dict.get('pred'),
                                            pred_dict.get('mat'),
                                            (width, height),
                                            flags=cv2.INTER_CUBIC,
                                            borderValue=0.)
            else:
                pred_image = np.zeros_like(y_true)
            pred_image_list.append(pred_image)
        pred_image = np.array(pred_image_list)
        y_pred = np.argmax(pred_image, axis=0)

        if args.vis:
            show = np.zeros_like(image)
            for i_h in range(height):
                for i_w in range(width):
                    show[i_h, i_w] = palette[y_pred[i_h, i_w], i_h, i_w]
            pred_vis_image = image * 0.5 + show * 0.5

            vis_image = np.concatenate([gt_vis_image, pred_vis_image], axis=1)
            cv2.imwrite(
                '{}/{}_{}_{}.jpg'.format(eval_dir, args.problem_type,
                                         image_split, file_id), vis_image)

            result_jaccard += [general_jaccard(y_true, y_pred)]
            result_dice += [general_dice(y_true, y_pred)]

    print('Jaccard (IoU): mean={:.2f}, std={:.4f}'.format(
        np.mean(result_jaccard) * 100, np.std(result_jaccard)))
    print('Dice: mean={:.2f}, std={:.4f}'.format(
        np.mean(result_dice) * 100, np.std(result_dice)))
