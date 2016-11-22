import argparse
import os
import sys
import chainer
import numpy as np
import numpy
import pylab
import cv2
from chainer import Variable

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import kawaii_creator

nz = 100

parser = argparse.ArgumentParser()
parser.add_argument("generator_model_file")
parser.add_argument("--out_file", default="output.png")
args = parser.parse_args()
xp = numpy

generator = kawaii_creator.models.Generator()
kawaii_creator.utility.load_modelfile(args.generator_model_file, generator)


def clip_img(x):
    return numpy.float32(-1 if x < -1 else (1 if x > 1 else x))


def save(x: numpy.ndarray, filepath, num=10):
    def to_image(x: numpy.ndarray):
        return cv2.cvtColor(
            ((numpy.vectorize(clip_img)(x) + 1) / 2).transpose(1, 2, 0) * 256,
            cv2.COLOR_RGB2BGR)

    cv2.imwrite(
        filepath,
        numpy.hstack(
            [to_image(x[i]) for i in range(num)]
        )
    )


xp = numpy
pylab.rcParams['figure.figsize'] = (22.0, 22.0)
pylab.clf()
z = (xp.random.uniform(-1, 1, (100, 100)).astype(np.float32))
z = Variable(z)
x = generator(z, test=True)
x = x.data
save(x, args.out_file, num=10)
