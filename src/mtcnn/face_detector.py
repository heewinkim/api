# -*- coding: utf-8 -*-
"""
===============================================
run_fd module
===============================================

========== ====================================
========== ====================================
 Module     run_fd module
 Date       2018-12-17
 Author     heewinkim
 Comment    `관련문서링크 <>`_
========== ====================================

*Abstract*
    * FaceDetector 클래스를 생성하여 모델관련 기능을 제공합니다. ( MTCNN FD 모델을 사용)
    * run 메서드를 제공합니다.


===============================================
"""

import os
import mxnet as mx
from .src.core.symbol import P_Net, R_Net, O_Net
from .src.core.detector import Detector
from .src.core.fcn_detector import FcnDetector
from .src.tools.load_model import load_param
from .src.core.mtcnn_detector import MtcnnDetector


class FaceDetector(object):

    def __init__(self):

        #if common.fd_hardware_type == 'gpu':
        #    self.ctx = mx.gpu()
        #else :
        #    self.ctx = mx.cpu()

        current_dir = os.path.dirname(os.path.realpath(__file__))

        self.ctx = mx.cpu()
        self.prefix = [current_dir+'/src/model/pnet',current_dir+'/src/model/rnet', current_dir+'/src/model/onet']
        self.min_face_size = 40
        self.epoch = [16, 16, 16]
        self.batch_size = [32, 16, 8]
        self.thresh = [0.7, 0.7, 0.9]
        self.stride = 2
        self.slide_window = False

        self._load_net()

    def _load_net(self, slide_window=False):
        """
        모델을 로드하여 self.mtcnn_detector 에 할당합니다.

        :param slide_window: slide window 알고리즘 사용여부
        :return: None
        """

        detectors = [None, None, None]

        # load pnet model
        args, auxs = load_param(self.prefix[0], self.epoch[0], convert=True, ctx=self.ctx)
        if slide_window:
            PNet = Detector(P_Net("test"), 12, self.batch_size[0], self.ctx, args, auxs)
        else:
            PNet = FcnDetector(P_Net("test"), self.ctx, args, auxs)
        detectors[0] = PNet

        # load rnet model
        args, auxs = load_param(self.prefix[1], self.epoch[0], convert=True, ctx=self.ctx)
        RNet = Detector(R_Net("test"), 24, self.batch_size[1], self.ctx, args, auxs)
        detectors[1] = RNet

        # load onet model
        args, auxs = load_param(self.prefix[2], self.epoch[2], convert=True, ctx=self.ctx)
        ONet = Detector(O_Net("test"), 48, self.batch_size[2], self.ctx, args, auxs)
        detectors[2] = ONet

        # 검출기 설정
        self.mtcnn_detector = MtcnnDetector(detectors=detectors, ctx=self.ctx, min_face_size=self.min_face_size,
                                       stride=self.stride, threshold=self.thresh, slide_window=self.slide_window)

    def run(self, img_cv):
        """
        이미지를 입력받아 얼굴검출 결과데이터를 반환합니다.
        얼굴검출 결과데이터는 전체 얼굴의 좌표와, 각각의 얼굴의 좌표정보 및 검출모델의 타입정보(mtcnn)를 포함합니다ㅏ.


        :param img_cv: cv 포맷의 이미지(numpy.array,BGR color)
        :return: {"x": 0, "y": 0, "xw": 0, "yh": 0, "w": 0, "h": 0, "fn": 0, "type": "mtcnn", "faces": []}
        """

        # face detection information
        fdi = {"x": 0, "y": 0, "xw": 0, "yh": 0, "w": 0, "h": 0, "fn": 0, "faces": []}
        faces = []

        # 얼굴 검출 수행
        boxes, boxes_c = self.mtcnn_detector.detect_pnet(img_cv)
        boxes, boxes_c = self.mtcnn_detector.detect_rnet(img_cv, boxes_c)
        boxes, boxes_c = self.mtcnn_detector.detect_onet(img_cv, boxes_c)

        if boxes_c is None:
            return fdi

        # 얼굴 검출 영역 저장 배열
        x1=[]; y1=[]; x2=[]; y2=[]

        fdi['fn'] = len(boxes_c)  # 검출 된 얼굴 수

        # 얼굴 검출 영역이 존재하면
        for idx, box in enumerate(boxes_c):

            x1.append(int(box[0]) if box[0] > 0 else 0)
            y1.append(int(box[1]) if box[1] > 0 else 0)
            x2.append(int(box[2]) if box[2] > 0 else 0)
            y2.append(int(box[3]) if box[3] > 0 else 0)
            faces.append({'index':idx,
                          'x': int(box[0]) if box[0] > 0 else 0,
                          'y': int(box[1]) if box[1] > 0 else 0,
                          'xw': int(box[2]) if box[2] > 0 else 0,
                          'yh': int(box[3]) if box[3] > 0 else 0,
                          'w': int(box[2] - box[0]) if (box[2] - box[0]) > 0 else 0,
                          'h': int(box[3] - box[1]) if (box[3] - box[1]) > 0 else 0,
                          'confidence':box[-1]
                          })

        # 그룹핑된 영역
        fdi["x"] = min(x1)
        fdi["y"] = min(y1)
        fdi["xw"] = max(x2)
        fdi["yh"] = max(y2)
        fdi['w'] = abs(fdi['xw'] - fdi['x'])
        fdi['h'] = abs(fdi['yh'] - fdi['y'])
        fdi['faces'] = faces

        return fdi
