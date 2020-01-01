import numpy.random as npr
import os

def gen_pnet_traindata(data_rootdir):

    size = 12
    dataset_type = 'train'

    dataset_path=data_rootdir+'/{}/{}'.format(dataset_type,size)

    print('reading data....')

    with open('%s/pos_%s.txt'%(dataset_path, size), 'r') as f:
        pos = f.readlines()

    with open('%s/neg_%s.txt'%(dataset_path, size), 'r') as f:
        neg = f.readlines()

    with open('%s/part_%s.txt'%(dataset_path, size), 'r') as f:
        part = f.readlines()

    print('writing data....')

    with open("%s/%s/imglists/train_%s.txt"%(data_rootdir,dataset_type, size), "w") as f:
        f.writelines(pos)
        neg_keep = npr.choice(len(neg), size=600000, replace=False)
        part_keep = npr.choice(len(part), size=300000, replace=False)
        for i in neg_keep:
            f.write(neg[i])
        for i in part_keep:
            f.write(part[i])

    print('done')

if __name__ == '__main__':

    import os

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    data_rootdir = parent_dir+'/data'

    gen_pnet_traindata(data_rootdir)
