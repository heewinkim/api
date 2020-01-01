import argparse
import mxnet as mx
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.imdb import IMDB
from train import train_net
from core.symbol import R_Net

def train_R_net(image_set, root_path, dataset_path, prefix, ctx,
                pretrained, epoch, begin_epoch,
                end_epoch, frequent, lr, resume):
    imdb = IMDB("mtcnn", image_set, root_path, dataset_path)
    gt_imdb = imdb.gt_imdb()
    gt_imdb = imdb.append_flipped_images(gt_imdb)
    sym = R_Net()

    train_net(sym, prefix, ctx, pretrained, epoch, begin_epoch, end_epoch, gt_imdb,
              24, frequent, not resume, lr)

if __name__ == '__main__':


    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    image_set = 'train_24'
    root_path = parent_dir+'/data'
    dataset_path = root_path + '/train'
    prefix =  '/model/rnet'
    gpu_ids = -1
    pretrained = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/model/pnet'
    epoch = 0
    begin_epoch = 0
    end_epoch=16
    frequent = 200
    lr = 0.01
    resume = True

    if gpu_ids == -1:
        ctx = mx.cpu(0)
    else:
        ctx = [mx.gpu(int(i)) for i in gpu_ids.split(',')]


    train_R_net(image_set, root_path, dataset_path, prefix,ctx, pretrained, epoch, begin_epoch,end_epoch, frequent, lr, resume)
