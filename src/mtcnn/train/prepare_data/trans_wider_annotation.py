# -*- coding: utf-8 -*-
from tqdm import tqdm
import h5py
import os


class DATA(object):
    def __init__(self, image_name, bboxes):
        self.image_name = image_name
        self.bboxes = bboxes


class WIDER(object):
    def __init__(self, file_to_label, path_to_image):
        self.file_to_label = file_to_label
        self.path_to_image = path_to_image

        self.f = h5py.File(file_to_label, 'r')
        self.event_list = self.f.get('event_list')
        self.file_list = self.f.get('file_list')
        self.face_bbx_list = self.f.get('face_bbx_list')

    def next(self):

        for event_idx, event in enumerate(self.event_list.value[0]):
            directory = self.f[event].value.tostring().decode('utf-16')
            for im_idx, im in enumerate(
                    self.f[self.file_list.value[0][event_idx]].value[0]):

                im_name = self.f[im].value.tostring().decode('utf-16')
                face_bbx = self.f[self.f[self.face_bbx_list.value
                [0][event_idx]].value[0][im_idx]].value

                bboxes = []

                for i in range(face_bbx.shape[1]):
                    xmin = int(face_bbx[0][i])
                    ymin = int(face_bbx[1][i])
                    xmax = int(face_bbx[0][i] + face_bbx[2][i])
                    ymax = int(face_bbx[1][i] + face_bbx[3][i])
                    bboxes.append((xmin, ymin, xmax, ymax))

                yield DATA(os.path.join(self.path_to_image, directory,
                                        im_name + '.jpg'), bboxes)


def trans_wider2mtcnn_annotation(img_dir,mat_filepath,save_path):
    """
    wider 데이터셋을 Mtcnn 학습에 맞는 annotation으로 변환합니다.

    :param img_dir: wider image directory ( train, validation 중 하나)
    :param mat_filepath: wider_face_<type>.mat 파일, (type : train, val 중 하나)
    :param save_path: 최종 변환된 Annotation 파일이 저장될 fullpath 경로
    """

    wider = WIDER(mat_filepath, img_dir)

    line_count = 0
    box_count = 0

    print('start transforming....')

    with open(save_path, 'w+') as f:
        # press ctrl-C to stop the process
        for data in tqdm(wider.next()):
            line = []
            line.append(str(data.image_name))
            line_count += 1
            for i, box in enumerate(data.bboxes):
                box_count += 1
                for j, bvalue in enumerate(box):
                    line.append(str(bvalue))

            line.append('\n')

            line_str = ' '.join(line)
            f.write(line_str)

    print('end transforming')
    print('total line(images): {}'.format(line_count))
    print('total boxes(faces): {}'.format(box_count))


if __name__ == '__main__':

    import os

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 하위 디렉토리로 trian,validation,test,wider_annotation을 포함하는 루트경로
    root_dir = parent_dir+'/data'

    # train or validation
    wider_dataset_type = 'train'

    # wider 데이터셋의 이미지 디렉토리 경로 ( 학습 혹은 검증 이미지)
    img_dir = root_dir+'/{}/images'.format(wider_dataset_type)

    # wider annotation 파일중 mat 파일 ( 학습 혹은 검증 이미지)
    mat_filepath = root_dir+'/wider_annotation/wider_face_{}.mat'.format(
        wider_dataset_type[:3] if wider_dataset_type == 'validation' else wider_dataset_type)

    # 변환된 annotation이 저장될 경로 (txt format)
    save_path = root_dir+'/{0}/{0}_anno.txt'.format(wider_dataset_type)

    # 변환
    trans_wider2mtcnn_annotation(img_dir,mat_filepath,save_path)

