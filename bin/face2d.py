# -*- coding: utf-8 -*-
import argparse
import os
import sys

import chainer.optimizers
import cv2
import numpy

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from kawaii_creator import models
import kawaii_creator

parser = argparse.ArgumentParser()
parser.add_argument("generator_model_file")
parser.add_argument("vectorizer_model_file")
parser.add_argument("--out_dir", default=".")
args = parser.parse_args()
xp = numpy

generator = models.Generator()
kawaii_creator.utility.load_modelfile(args.generator_model_file, generator)

extractor = models.FaceExtractor()

vectorizer = models.Vectorizer()
kawaii_creator.utility.load_modelfile(args.vectorizer_model_file, vectorizer)

# classifier = cv2.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt_tree.xml")
classifier = cv2.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml")

converter = kawaii_creator.utility.Real2AnimeConverter(vectorizer, generator, margin_ratio=0.3)

cap = cv2.VideoCapture(0)
margin_ratio = 0.3

for i in range(10000):
    ret, frame = cap.read()
    print(frame.shape)

    converted = converter.convert(frame)
    if converted is not None:
        cv2.imshow("test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.waitKey(-1)
cap.release()
cv2.destroyAllWindows()
