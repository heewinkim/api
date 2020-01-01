# -*- coding: utf-8 -*-

import numpy as np
import mxnet as mx
import argparse
import os
import cPickle
import cv2
from core.symbol import P_Net, R_Net, O_Net
from core.imdb import IMDB
from config import config
from core.loader import TestLoader
from core.detector import Detector
from core.fcn_detector import FcnDetector
from tools.load_model import load_param
from core.MtcnnDetector import MtcnnDetector
from utils import *
from tqdm import tqdm
import traceback


current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)


def save_hard_example(root_dir,net,pkl_path,type='train'):

    if net == "rnet":
        image_size = 24
    if net == "onet":
        image_size = 48

    image_dir = root_dir+ "/images"
    neg_save_dir = root_dir+"/{}/negative".format(image_size)
    pos_save_dir = root_dir+"/{}/positive".format(image_size)
    part_save_dir = root_dir+"/{}/part".format(image_size)
    save_path = root_dir+"/{}".format(image_size)

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(part_save_dir):
        os.mkdir(part_save_dir)
    if not os.path.exists(neg_save_dir):
        os.mkdir(neg_save_dir)
    if not os.path.exists(pos_save_dir):
        os.mkdir(pos_save_dir)
    
    # load ground truth from annotation file
    # format of each line: image/path [x1,y1,x2,y2] for each gt_box in this image
    anno_file = root_dir+"/train_anno.txt"
    with open(anno_file, 'r') as f:
        annotations = f.readlines()

    im_idx_list = list()
    gt_boxes_list = list()
    num_of_images = len(annotations)
    print "processing %d images in total"%num_of_images

    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        im_idx = annotation[0]

        boxes = map(float, annotation[1:])
        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        im_idx_list.append(im_idx)
        gt_boxes_list.append(boxes)


    f1 = open(os.path.join(save_path, 'pos_%d.txt'%image_size), 'w')
    f2 = open(os.path.join(save_path, 'neg_%d.txt'%image_size), 'w')
    f3 = open(os.path.join(save_path, 'part_%d.txt'%image_size), 'w')

    det_boxes = cPickle.load(open(pkl_path, 'r'))
    assert len(det_boxes) == num_of_images, "incorrect detections or ground truths"

    # index of neg, pos and part face, used as their image names
    n_idx = 0
    p_idx = 0
    d_idx = 0
    image_done = 0
    for im_idx, dets, gts in tqdm(zip(im_idx_list, det_boxes, gt_boxes_list)):
        # if image_done % 100 == 0:
            # print "%d images done"%image_done
        # image_done += 1

        if dets.shape[0]==0:
            continue
        img = cv2.imread(im_idx)
        if img is None:
            print('[ERROR] img is None!!')
            return

        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        for box in dets:
            x_left, y_top, x_right, y_bottom, _ = box.astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            # ignore box that is too small or beyond image border
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue

            # compute intersection over union(IoU) between current box and all gt boxes
            Iou = IoU(box, gts)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (image_size, image_size),
                                    interpolation=cv2.INTER_LINEAR)

            # save negative images and write label
            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
                f2.write("%s/negative/%s"%(image_size, n_idx) + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
            else:
                # find gt_box with the highest iou
                idx = np.argmax(Iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt

                # compute bbox reg label
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom ) / float(height)

                # save positive and part-face images and write labels
                if np.max(Iou) >= 0.65:
                    save_file = os.path.join(pos_save_dir, "%s.jpg"%p_idx)
                    f1.write("%s/positive/%s"%(image_size, p_idx) + ' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1

                elif np.max(Iou) >= 0.4:
                    save_file = os.path.join(part_save_dir, "%s.jpg"%d_idx)
                    f3.write("%s/part/%s"%(image_size, d_idx) + ' -1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
    f1.close()
    f2.close()
    f3.close()

def test_net(root_path, model_rootpath,prefix, epoch, batch_size, ctx, test_mode="rnet",
             thresh=[0.6, 0.6, 0.7], min_face_size=24, stride=2, slide_window=False,
             shuffle=False, vis=False):

    detectors = [None, None, None]
    proj_rootdir = os.path.dirname(model_rootdir)

    # load pnet model
    args, auxs = load_param(proj_rootdir+prefix[0],epoch[0],convert=True,ctx=ctx)
    if slide_window:
        PNet = Detector(P_Net("test"), 12, batch_size[0], ctx, args, auxs)
    else:
        PNet = FcnDetector(P_Net("test"), ctx, args, auxs)
    detectors[0] = PNet

    # load rnet model
    if test_mode in ["rnet", "onet"]:
        args, auxs = load_param(proj_rootdir+prefix[1], epoch[0], convert=True, ctx=ctx)
        RNet = Detector(R_Net("test"), 24, batch_size[1], ctx, args, auxs)
        detectors[1] = RNet

    # load onet model
    if test_mode == "onet":
        args, auxs = load_param(proj_rootdir+prefix[2], epoch[2], convert=True, ctx=ctx)
        ONet = Detector(O_Net("test"), 48, batch_size[2], ctx, args, auxs)
        detectors[2] = ONet

    mtcnn_detector = MtcnnDetector(detectors=detectors, ctx=ctx, min_face_size=min_face_size,
                                   stride=stride, threshold=thresh, slide_window=slide_window)


    imdb = IMDB("wider", 'train_anno', root_path, root_path , 'test')

    gt_imdb = imdb.gt_imdb()

    test_data = TestLoader(gt_imdb)
    detections = mtcnn_detector.detect_face(imdb, test_data, vis=vis)

    if test_mode == "pnet":
        net = "rnet"
    elif test_mode == "rnet":
        net = "onet"

    # save testing result pickle data
    save_path = os.path.dirname(os.path.abspath(__file__))+"/%s"%test_mode
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_file = os.path.join(save_path, "detections.pkl")
    with open(save_file, 'wb') as f:
        cPickle.dump(detections, f, cPickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':

    import os

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # dataset root 디렉토리
    root_path = parent_dir+'/data'

    # model root 디렉토리
    model_rootdir = parent_dir+'/model'

    # 테스트 모델
    test_mode = 'rnet'

    # 각 모델 prefix
    prefix = ['/model/pnet', '/model/rnet', '/model/onet']

    # 각 스테이지의 에포크
    epoch = [16, 16, 16]

    # 각 스테이지의 배치
    batch_size=[32, 16, 8]

    # 각 스테이지의 임계값
    thresh=[0.7, 0.7, 0.5]

    # 최소 검출 얼굴 크기
    min_face = 24

    # 컨볼루션 파라미터
    stride = 2

    # 슬라이딩 윈도우 사용여부
    slide_window = False

    # gpu id , 없으면 -1
    gpu_id = -1

    # 데이터 셔플
    shuffle = False

    # visualize on/off
    vis = False

    # mxnet cpu/gpu 설정
    ctx = mx.gpu(gpu_id)
    if gpu_id == -1:
        ctx = mx.cpu(0)

    print('Test {} and save result pickle data...'.format(test_mode))
    test_net(root_path, model_rootdir, prefix,epoch, batch_size, ctx, test_mode,thresh, min_face, stride,slide_window, shuffle, vis)
    print('done testing.')


    print('make next train data...')
    if test_mode == "pnet":        net = "rnet"
    elif test_mode == "rnet":        net = "onet"
    save_file = os.path.dirname(os.path.abspath(__file__))+"/%s"%test_mode+"/detections.pkl"
    save_hard_example(root_path + '/train', net, save_file)
    print('all done')
