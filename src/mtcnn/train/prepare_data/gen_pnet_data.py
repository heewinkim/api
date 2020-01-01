# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
import numpy.random as npr
from utils import IoU
from tqdm import tqdm

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)


def make_pnet_dataset(root_dir,annotation_filepath):
    """
    root_dir 경로밑에 pnet에 필요한 데이터셋을 만듭니다.
    annotation_file_format : annotation file image_path x1 x2 y1 y2

    :param root_dir: 데이터셋 저장 루트디렉토리
    :param annotation_filename: wider_face annotation txt file
    """

    anno_file = annotation_filepath
    neg_save_dir = root_dir + "/12/negative"
    pos_save_dir = root_dir + "/12/positive"
    part_save_dir = root_dir + "/12/part"
    save_dir = root_dir + "/12"

    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(part_save_dir):
        os.mkdir(part_save_dir)
    if not os.path.exists(neg_save_dir):
        os.mkdir(neg_save_dir)
    if not os.path.exists(pos_save_dir):
        os.mkdir(pos_save_dir)

    f1 = open(os.path.join(save_dir, 'pos_12.txt'), 'w')
    f2 = open(os.path.join(save_dir, 'neg_12.txt'), 'w')
    f3 = open(os.path.join(save_dir, 'part_12.txt'), 'w')

    with open(anno_file, 'r') as f:
        annotations = f.readlines()

    num = len(annotations)
    print("{} pics in total".format(num))
    p_idx = 0 # positive
    n_idx = 0 # negative
    d_idx = 0 # dont care
    idx = 0
    box_idx = 0

    for annotation in tqdm(annotations):
        annotation = annotation.strip().split(' ')
        im_path = annotation[0]
        bbox = map(float, annotation[1:])
        boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
        img = cv2.imread(im_path)
        idx += 1

        height, width, channel = img.shape

        neg_num = 0
        while neg_num < 50:
            size = npr.randint(12, min(width, height) / 2)
            nx = npr.randint(0, width - size)
            ny = npr.randint(0, height - size)
            crop_box = np.array([nx, ny, nx + size, ny + size])

            Iou = IoU(crop_box, boxes)

            cropped_im = img[ny : ny + size, nx : nx + size, :]
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dir, "{}.jpg".format(n_idx))
                f2.write("12/negative/%s"%n_idx + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1

        for box in boxes:
            # box (x_left, y_top, x_right, y_bottom)
            x1, y1, x2, y2 = box
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            # ignore small faces
            # in case the ground truth boxes of small faces are not accurate
            if max(w, h) < 40 or x1 < 0 or y1 < 0:
                continue

            # generate negative examples that have overlap with gt
            for i in range(5):
                size = npr.randint(12,  min(width, height) / 2)
                # delta_x and delta_y are offsets of (x1, y1)
                delta_x = npr.randint(max(-size, -x1), w)
                delta_y = npr.randint(max(-size, -y1), h)
                nx1 = max(0, x1 + delta_x)
                ny1 = max(0, y1 + delta_y)
                if nx1 + size > width or ny1 + size > height:
                    continue
                crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
                Iou = IoU(crop_box, boxes)

                cropped_im = img[int(ny1): int(ny1 + size), int(nx1): int(nx1 + size), :]
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

                if np.max(Iou) < 0.3:
                    # Iou with all gts must below 0.3
                    save_file = os.path.join(neg_save_dir, "{}.jpg".format(n_idx))
                    f2.write("12/negative/{}".format(n_idx) +' 0\n')
                    cv2.imwrite(save_file, resized_im)
                    n_idx += 1

            # generate positive examples and part faces
            for i in range(20):
                size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

                # delta here is the offset of box center
                delta_x = npr.randint(int(-w * 0.2), w * 0.2)
                delta_y = npr.randint(int(-h * 0.2), h * 0.2)

                nx1 = max(x1 + w / 2 + delta_x - size / 2, 0)
                ny1 = max(y1 + h / 2 + delta_y - size / 2, 0)
                nx2 = nx1 + size
                ny2 = ny1 + size

                if nx2 > width or ny2 > height:
                    continue
                crop_box = np.array([nx1, ny1, nx2, ny2])

                offset_x1 = (x1 - nx1) / float(size)
                offset_y1 = (y1 - ny1) / float(size)
                offset_x2 = (x2 - nx2) / float(size)
                offset_y2 = (y2 - ny2) / float(size)

                cropped_im = img[int(ny1) : int(ny2), int(nx1) : int(nx2), :]
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

                box_ = box.reshape(1, -1)
                if IoU(crop_box, box_) >= 0.65:
                    save_file = os.path.join(pos_save_dir, "{}.jpg".format(p_idx))
                    f1.write("12/positive/%s"%p_idx + ' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1
                elif IoU(crop_box, box_) >= 0.4:
                    save_file = os.path.join(part_save_dir, "{}.jpg".format(d_idx))
                    f3.write("12/part/{}".format(d_idx) + ' -1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
            box_idx += 1

    f1.close()
    f2.close()
    f3.close()


if __name__ == '__main__':

    import os

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    data_rootdir=parent_dir+'/data'

    # pnet dataset이 생성될 루트 경로
    train_dir = data_rootdir+'/train'

    # pnet 데이터 생성을 하기위한 annotation 파일경로
    annotation_filepath = data_rootdir+'/train/train_anno.txt'

    # pnet 데이터 생성
    make_pnet_dataset(train_dir,annotation_filepath)
