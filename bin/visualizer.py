import argparse
import os
import sys
import chainer
import numpy as np
import numpy
import pylab
import cv2
from chainer import Variable
from chainer import serializers



sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import model

nz = 100

parser = argparse.ArgumentParser()
parser.add_argument("generator_model_file")
parser.add_argument("--out_file", default="output.png")
args = parser.parse_args()
xp = numpy

generator = model.Generator()
chainer.serializers.load_hdf5(args.generator_model_file, generator)

def clip_img(x):
    return numpy.float32(-1 if x < -1 else (1 if x > 1 else x))

def save(x, filepath):
    img = ((numpy.vectorize(clip_img)(x[0, :, :, :]) + 1) / 2).transpose(1, 2, 0)
    cv2.imwrite(
        filepath,
        cv2.cvtColor(img * 256, cv2.COLOR_RGB2BGR)
    )


xp = numpy

pylab.rcParams['figure.figsize'] = (22.0,22.0)
pylab.clf()
vissize = 100
z = (xp.random.uniform(-1, 1, (100, 100)).astype(np.float32))
z = Variable(z)
x = generator(z, test=True)
x = x.data
# for i_ in range(100):
#     tmp = ((np.vectorize(clip_img)(x[i_,:,:,:])+1)/2).transpose(1,2,0)
#     pylab.subplot(10,10,i_+1)
#     pylab.imshow(tmp)
#     pylab.axis('off')
# pylab.savefig(args.out_file)
save(x, args.out_file)
