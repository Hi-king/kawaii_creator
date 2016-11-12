# -*- coding: utf-8 -*-
import argparse
import os
import sys
import chainer.optimizers
import cv2
import numpy
import pylab




sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import model

parser = argparse.ArgumentParser()
parser.add_argument("generator_model_file")
parser.add_argument("vectorizer_model_file")
parser.add_argument("--out_dir", default=".")
args = parser.parse_args()
xp = numpy

generator = model.Generator()
chainer.serializers.load_hdf5(args.generator_model_file, generator)

extractor = model.FaceExtractor()

vectorizer = model.Vectorizer()
chainer.serializers.load_hdf5(args.vectorizer_model_file, vectorizer)



def clip_img(x):
    return numpy.float32(-1 if x<-1 else (1 if x>1 else x))

def to_img(x):
    img = ((numpy.vectorize(clip_img)(x[0,:,:,:])+1)/2).transpose(1,2,0)
    return cv2.cvtColor(img*256, cv2.COLOR_RGB2BGR)

def convert(input_array, vectorizer, generator):
    rgb_img = cv2.cvtColor(cv2.resize(
        input_array,
        (96, 96),
        interpolation=cv2.INTER_AREA
    ), cv2.COLOR_BGR2RGB)


    float_img = rgb_img.astype(numpy.float32)
    converted_matrix = float_img / 256

    face_img_var = chainer.Variable(
        numpy.array([converted_matrix.transpose(2, 0, 1) * 2 - 1.0]))
    reconstructed = vectorizer(face_img_var)
    regenerated = generator(reconstructed, test=True)
    converted = cv2.resize(
        to_img(regenerated.data),
        input_array.shape[:2],
        interpolation=cv2.INTER_AREA
    )
    return converted

# classifier = cv2.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt_tree.xml")
classifier = cv2.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml")


cap = cv2.VideoCapture(0)
margin_ratio = 0.3

for i in range(10000):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # facerect = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
    facerect = classifier.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(64, 64))
    if len(facerect) > 0:
        #検出した顔を囲む矩形の作成
        for rect in facerect:
            x, y, width, height = rect
            image_height, image_width, _ = frame.shape
            margin = min(
                y, image_height - y - height,
                x, image_width - x - width,
                   int(width * margin_ratio)
            )
            crop = frame[y - margin:y + height + margin, x - margin:x + width + margin]
            if crop.shape[0] == 0 or crop.shape[1] == 0:
                continue
            frame[y - margin:y + height + margin, x - margin:x + width + margin] = convert(crop, vectorizer, generator)
        cv2.imshow("test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.waitKey(-1)
cap.release()
cv2.destroyAllWindows()