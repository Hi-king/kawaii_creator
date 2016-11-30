# -*- coding: utf-8 -*-
import os
import cv2

import chainer
import numpy
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import kawaii_creator


def load_modelfile(path, model):
    if os.path.splitext(path)[1] == ".npz":
        chainer.serializers.load_npz(path, model)
    elif os.path.splitext(path)[1] == ".h5":
        chainer.serializers.load_hdf5(path, model)
    else:
        raise Exception()


def clip_img(x):
    return numpy.float32(-1 if x < -1 else (1 if x > 1 else x))


def to_img(x):
    img = ((numpy.vectorize(clip_img)(x[0, :, :, :]) + 1) / 2).transpose(1, 2, 0)
    return cv2.cvtColor(img * 256, cv2.COLOR_RGB2BGR)


class Real2AnimeConverter(object):
    def __init__(self, vectorizer, generator,  margin_ratio=0.3):
        ROOTPATH = os.path.join(os.path.dirname(__file__), "..")
        self.margin_ratio = margin_ratio
        self.vectorizer = vectorizer
        self.generator = generator
        self.classifier = cv2.CascadeClassifier(os.path.join(ROOTPATH, "haarcascade_frontalface_default.xml"))

    def _convert_each(self, input_array):
        rgb_img = cv2.cvtColor(cv2.resize(
            input_array,
            (96, 96),
            interpolation=cv2.INTER_AREA
        ), cv2.COLOR_BGR2RGB)

        float_img = rgb_img.astype(numpy.float32)
        converted_matrix = float_img / 256

        face_img_var = chainer.Variable(
            numpy.array([converted_matrix.transpose(2, 0, 1) * 2 - 1.0]))
        reconstructed = self.vectorizer(face_img_var)
        regenerated = self.generator(reconstructed, test=True)
        converted = cv2.resize(
            to_img(regenerated.data),
            input_array.shape[:2],
            interpolation=cv2.INTER_AREA
        )
        return converted

    def convert(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        facerect = self.classifier.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(64, 64))
        if len(facerect) > 0:
            # 検出した顔を囲む矩形の作成
            for rect in facerect:
                x, y, width, height = rect
                image_height, image_width, _ = frame.shape
                margin = min(
                    y, image_height - y - height,
                    x, image_width - x - width,
                    int(width * self.margin_ratio)
                )
                crop = frame[y - margin:y + height + margin, x - margin:x + width + margin]
                if crop.shape[0] == 0 or crop.shape[1] == 0:
                    continue
                frame[y - margin:y + height + margin, x - margin:x + width + margin] = self._convert_each(crop)
            return frame
        else:
            return None
