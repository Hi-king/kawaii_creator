import argparse
import os
import sys

import cv2
import numpy

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

converter = kawaii_creator.utility.Real2AnimeConverter(vectorizer, generator, margin_ratio=0.3)

cv2.imwrite(
    args.out_file,
    converter.convert(cv2.imread(args.target_img))
)
