import math
import os

import chainer
import cv2
import numpy


class Generator(chainer.Chain):
    def __init__(self, inputdim=100):
        super(Generator, self).__init__(
            l0z=chainer.links.Linear(inputdim, 6 * 6 * 512, wscale=0.02 * math.sqrt(inputdim)),
            dc1=chainer.links.Deconvolution2D(512, 256, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 512)),
            dc2=chainer.links.Deconvolution2D(256, 128, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 256)),
            dc3=chainer.links.Deconvolution2D(128, 64, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 128)),
            dc4=chainer.links.Deconvolution2D(64, 3, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64)),
            bn0l=chainer.links.BatchNormalization(6 * 6 * 512),
            bn0=chainer.links.BatchNormalization(512),
            bn1=chainer.links.BatchNormalization(256),
            bn2=chainer.links.BatchNormalization(128),
            bn3=chainer.links.BatchNormalization(64),
        )

    def __call__(self, z, test=False):
        h = chainer.functions.reshape(chainer.functions.relu(self.bn0l(self.l0z(z), test=test)),
                                      (z.data.shape[0], 512, 6, 6))
        h = chainer.functions.relu(self.bn1(self.dc1(h), test=test))
        h = chainer.functions.relu(self.bn2(self.dc2(h), test=test))
        h = chainer.functions.relu(self.bn3(self.dc3(h), test=test))
        x = (self.dc4(h))
        return x


class Discriminator(chainer.Chain):
    def __init__(self):
        super(Discriminator, self).__init__(
            c0=chainer.links.Convolution2D(3, 64, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 3)),
            c1=chainer.links.Convolution2D(64, 128, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64)),
            c2=chainer.links.Convolution2D(128, 256, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 128)),
            c3=chainer.links.Convolution2D(256, 512, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 256)),
            l4l=chainer.links.Linear(6 * 6 * 512, 2, wscale=0.02 * math.sqrt(6 * 6 * 512)),
            bn0=chainer.links.BatchNormalization(64),
            bn1=chainer.links.BatchNormalization(128),
            bn2=chainer.links.BatchNormalization(256),
            bn3=chainer.links.BatchNormalization(512),
        )

    def __call__(self, x, test=False):
        h = chainer.functions.elu(self.c0(x))  # no bn because images from generator will katayotteru?
        h = chainer.functions.elu(self.bn1(self.c1(h), test=test))
        h = chainer.functions.elu(self.bn2(self.c2(h), test=test))
        h = chainer.functions.elu(self.bn3(self.c3(h), test=test))
        l = self.l4l(h)
        return l


class Vectorizer(chainer.Chain):
    def __init__(self):
        super(Vectorizer, self).__init__(
            c0=chainer.functions.Convolution2D(3, 64, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 3)),
            c1=chainer.functions.Convolution2D(64, 128, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64)),
            c2=chainer.functions.Convolution2D(128, 256, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 128)),
            c3=chainer.functions.Convolution2D(256, 512, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 256)),
            l0z=chainer.functions.Linear(6 * 6 * 512, 100, wscale=0.02 * math.sqrt(6 * 6 * 512)),
            bn0=chainer.functions.BatchNormalization(64),
            bn1=chainer.functions.BatchNormalization(128),
            bn2=chainer.functions.BatchNormalization(256),
            bn3=chainer.functions.BatchNormalization(512),
        )

    def __call__(self, x, test=False):
        h = chainer.functions.relu(self.c0(x))  # no bn because images from generator will katayotteru?
        h = chainer.functions.relu(self.bn1(self.c1(h), test=test))
        h = chainer.functions.relu(self.bn2(self.c2(h), test=test))
        h = chainer.functions.relu(self.bn3(self.c3(h), test=test))
        l = self.l0z(h)
        return l


class FaceExtractor(object):
    def __init__(self, margin=0.3):
        self.classifier = cv2.CascadeClassifier(
            os.path.join(os.path.dirname(__file__), "..", "animeface", "lbpcascade_animeface.xml"))
        self.margin = margin

    def extract(self, img_file):
        target_img = cv2.imread(img_file)
        gray_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        gray_img_preprocessed = cv2.equalizeHist(gray_img)
        faces = self.classifier.detectMultiScale(
            gray_img_preprocessed,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(24, 24))
        print(len(faces))
        if len(faces) == 0:
            raise (Exception("Could not find face from img"))

        x, y, width, height = faces[0]
        image_width, image_height, _ = target_img.shape
        margin = min(
            y, image_height - y - width,
            x, image_width - x - width,
               width * self.margin
        )
        rgb_img = cv2.cvtColor(cv2.resize(
            target_img[y - margin:y + height + margin, x - margin:x + width + margin],
            (96, 96),
            # interpolation=cv2.INTER_LANCZOS4
            # interpolation=cv2.INTER_NEAREST,
            interpolation=cv2.INTER_AREA
        ), cv2.COLOR_BGR2RGB)

        # import pylab
        # pylab.imshow(cv2.cvtColor(
        #     target_img[y-margin:y+height+margin, x-margin:x+width+margin],
        #     cv2.COLOR_BGR2RGB
        # ))
        # pylab.show()

        float_img = rgb_img.astype(numpy.float32)
        return float_img / 256
