import argparse
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mxnet as mx
from imdb_train import IMDB
from train.train import train_net
from core.symbol import P_Net

def train_P_net(image_set, root_path, dataset_path, prefix, ctx,
                pretrained, epoch, begin_epoch,
                end_epoch, frequent, lr, resume):
    imdb = IMDB("mtcnn", image_set, root_path, dataset_path)
    gt_imdb = imdb.gt_imdb()
    gt_imdb = imdb.append_flipped_images(gt_imdb)
    sym = P_Net()

    train_net(sym, prefix, ctx, pretrained, epoch, begin_epoch, end_epoch, gt_imdb, 12, frequent, not resume, lr)

if __name__ == '__main__':

    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    image_set = 'train_12'
    root_path = parent_dir+'/data'
    dataset_path=root_path+'/train'
    prefix = '/model/pnet'
    gpu_ids = -1
    pretrained = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/model/pnet'
    epoch=16
    begin_epoch=0
    end_epoch=16
    frequent=200
    lr=0.01
    resume=True

    if gpu_ids == -1:
        ctx = mx.cpu(0)
    else:
        ctx = [mx.gpu(int(i)) for i in gpu_ids.split(',')]

    train_P_net(image_set, root_path, dataset_path, prefix, ctx, pretrained, epoch, begin_epoch, end_epoch, frequent, lr, resume)
