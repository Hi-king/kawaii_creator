import argparse
import os
import random
import sys

import chainer.optimizers
import matplotlib
import numpy

matplotlib.use('Agg')
import pylab

import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from kawaii_creator import model

parser = argparse.ArgumentParser()
parser.add_argument("model_file")
parser.add_argument("iter", type=int)
parser.add_argument("--out_dir", default=".")
parser.add_argument("--gpu", type=int, default=-1)
args = parser.parse_args()

if args.gpu >= 0:
    chainer.cuda.check_cuda_available()
    # chainer.Function.type_check_enable = False
    chainer.cuda.get_device(args.gpu).use()
    xp = chainer.cuda.cupy
else:
    xp = numpy


def augment(original_img, max_margin=10):
    margin = random.randint(0, max_margin)
    original_width, original_height, _ = original_img.shape
    left = random.randint(0, margin)
    top = random.randint(0, margin)
    cropped_img = original_img[
                  left:left + (original_width - 2 * margin),
                  top:top + (original_height - 2 * margin),
                  ]
    return cv2.resize(cropped_img, (original_width, original_height))


generator = model.Generator()
chainer.serializers.load_hdf5(args.model_file, generator)

extractor = model.FaceExtractor()

optimizer = chainer.optimizers.Adam(alpha=0.001)
vectorizer = model.Vectorizer()
if args.gpu >= 0:
    generator.to_gpu()
    vectorizer.to_gpu()
optimizer.setup(vectorizer)

for i in xrange(args.iter):
    z_data = (xp.random.uniform(-1, 1, (1, 100)).astype(xp.float32))
    optimizer.zero_grads()
    z = chainer.Variable(z_data)
    x = generator(z, test=True)

    # augmentation
    if args.gpu >= 0:
        augmented = xp.array([augment(x.data.get()[0].transpose(1, 2, 0)).transpose(2, 0, 1)])
    else:
        augmented = xp.array([augment(x.data[0].transpose(1, 2, 0)).transpose(2, 0, 1)])

    reconstructed = vectorizer(chainer.Variable(augmented))

    loss = chainer.functions.mean_squared_error(
        reconstructed,
        z
    )
    loss.backward()
    optimizer.update()

    if i % 1000 == 0:
        print("i: {}, loss: {}".format(i, loss.data))

    if i % 10000 == 0:
        print(loss.data)
        chainer.serializers.save_hdf5(os.path.join(args.out_dir, "vectorizer_model_{}".format(i)), vectorizer)

    if i % 10000 == 0:
        def clip_img(x):
            return numpy.float32(-1 if x < -1 else (1 if x > 1 else x))


        def save(x, filepath):
            img = ((numpy.vectorize(clip_img)(x[0, :, :, :]) + 1) / 2).transpose(1, 2, 0)
            pylab.imshow(img)
            pylab.axis('off')
            pylab.savefig(filepath)


        reconstructed = vectorizer(x)
        regenerated = generator(reconstructed, test=True)

        if args.gpu >= 0:
            save(x.data.get(), os.path.join(args.out_dir, "constructed.png"))
            save(regenerated.data.get(), os.path.join(args.out_dir, "reconstructed.png"))
        else:
            save(x.data, os.path.join(args.out_dir, "constructed.png"))
            save(regenerated.data, os.path.join(args.out_dir, "reconstructed.png"))
