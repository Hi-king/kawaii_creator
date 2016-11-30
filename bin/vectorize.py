import argparse
import os
import sys

import chainer.optimizers
import cv2
import numpy
import pylab

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import kawaii_creator

parser = argparse.ArgumentParser()
parser.add_argument("generator_model_file")
parser.add_argument("vectorizer_model_file")
parser.add_argument("target_img")
parser.add_argument("--out_file", default="reconstructed.png")
parser.add_argument("--show", action="store_true")
parser.add_argument("--real_face", action="store_true")
args = parser.parse_args()
xp = numpy

generator = kawaii_creator.models.Generator()
kawaii_creator.utility.load_modelfile(args.generator_model_file, generator)

extractor = kawaii_creator.models.FaceExtractor(real_face=args.real_face)

vectorizer = kawaii_creator.models.Vectorizer()
kawaii_creator.utility.load_modelfile(args.vectorizer_model_file, vectorizer)

face_img = extractor.extract(args.target_img)
# face_img = cv2.resize(cv2.cvtColor(cv2.imread(args.target_img), cv2.COLOR_BGR2RGB).astype(numpy.float32) / 256, (96, 96))
face_img_var = chainer.Variable(
    numpy.array([face_img.transpose(2, 0, 1) * 2 - 1.0]))

if args.show:
    pylab.imshow(face_img)
    pylab.show()

reconstructed = vectorizer(face_img_var)
regenerated = generator(reconstructed, test=True)


def clip_img(x):
    return numpy.float32(-1 if x < -1 else (1 if x > 1 else x))


def save(x, filepath):
    img = ((numpy.vectorize(clip_img)(x[0, :, :, :]) + 1) / 2).transpose(1, 2, 0)
    # pylab.savefig(filepath)
    if args.show:
        pylab.imshow(img)
        pylab.axis('off')
        pylab.show()

    print(img.shape)
    cv2.imwrite(
        filepath,
        cv2.cvtColor(img * 256, cv2.COLOR_RGB2BGR)
    )


save(regenerated.data, args.out_file)
