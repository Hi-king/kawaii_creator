# -*- coding: utf-8 -*-
import argparse
import glob
import random

import cv2
import numpy
import chainer
import sys
import os
import pipe
import time
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import kawaii_creator


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


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--batchsize", type=int, default=10)
parser.add_argument("--use_accuracy_threshold", action="store_true")
parser.add_argument("--use_vectorizer", action="store_true")
parser.add_argument("--vectorizer_training_dataset")
parser.add_argument("--classifier_training_image_dataset")
parser.add_argument("--classifier_training_attribute_dataset")
parser.add_argument("--outprefix", default="")
parser.add_argument("--classifier_loss_weight", default=1.0, type=float)
parser.add_argument("--pretrained_generator")
parser.add_argument("--pretrained_discriminator")
parser.add_argument("--pretrained_vectorizer")
parser.add_argument('--disable_generator_training', action='store_false', dest='generator_training')
args = parser.parse_args()

GENERATOR_INPUT_DIMENTIONS = 100
outdirname = "_".join([
                          args.outprefix,
                          "finetune" if args.pretrained_generator is not None else "",
                          "batch{}".format(args.batchsize),
                          "accthresh" if args.use_accuracy_threshold else "",
                          "withvec" if args.use_vectorizer else "",
                          "disable_gen_train" if not args.generator_training else "",
                          "finetunevectorizer" if args.pretrained_vectorizer is not None else "",
                          "withvectrain" if args.vectorizer_training_dataset is not None else "",
                          "withattributeclassifier" if args.classifier_training_attribute_dataset is not None else "",
                          "withtrain" if args.vectorizer_training_dataset is not None else "",
                          str(int(time.time())),
                      ] | pipe.where(lambda x: len(x) > 0))

OUTPUT_DIRECTORY = os.path.join(os.path.dirname(__file__), "..", "output", outdirname)
os.makedirs(OUTPUT_DIRECTORY)

logging.basicConfig(filename=os.path.join(OUTPUT_DIRECTORY, "log.txt"), level=logging.DEBUG)
console = logging.StreamHandler()
logging.getLogger('').addHandler(console)

logging.info(args)
if args.pretrained_generator is not None:
    logging.info("pretrained_generator: {}".format(os.path.abspath(args.pretrained_generator)))
if args.pretrained_discriminator is not None:
    logging.info("pretrained_discriminator: {}".format(os.path.abspath(args.pretrained_discriminator)))
if args.pretrained_vectorizer is not None:
    logging.info("pretrained_vectorizer: {}".format(os.path.abspath(args.pretrained_vectorizer)))

if args.gpu >= 0:
    chainer.cuda.check_cuda_available()
    chainer.cuda.get_device(args.gpu).use()
    xp = chainer.cuda.cupy
else:
    xp = numpy

batchsize = args.batchsize

paths = glob.glob("{}/*".format(args.dataset))
dataset = kawaii_creator.datasets.PreprocessedDataset(
    kawaii_creator.datasets.ResizedImageDataset(paths=paths, resize=(96, 96)))
# iterator = chainer.iterators.SerialIterator(dataset, batch_size=batchsize, repeat=True, shuffle=True)
iterator = chainer.iterators.MultiprocessIterator(dataset, batch_size=batchsize, repeat=True, shuffle=True)

if args.vectorizer_training_dataset is not None:
    vectorizer_training_paths = glob.glob("{}/*".format(args.vectorizer_training_dataset))
    vectorizer_training_dataset = kawaii_creator.datasets.PreprocessedDataset(
        kawaii_creator.datasets.ResizedImageDataset(paths=vectorizer_training_paths, resize=(96, 96)))
    vectorizer_training_iterator = chainer.iterators.MultiprocessIterator(vectorizer_training_dataset,
                                                                          batch_size=batchsize, repeat=True,
                                                                          shuffle=True)

if args.classifier_training_attribute_dataset is not None:
    classifier_training_image_dataset = kawaii_creator.datasets.PreprocessedDataset(
        kawaii_creator.datasets.ResizedImageDataset(paths=args.classifier_training_image_dataset, resize=(96, 96))
    )
    classifier_training_attribute_label_dataset = kawaii_creator.datasets.AttributeLabelDataset(
        args.classifier_training_attribute_dataset)
    classifier_training_attribute_dataset = kawaii_creator.datasets.ZippedDataset(
        classifier_training_image_dataset,
        classifier_training_attribute_label_dataset
    )
    classifier_attribute_training_iterator = chainer.iterators.MultiprocessIterator(
        classifier_training_attribute_dataset,
        batch_size=batchsize, repeat=True,
        shuffle=True)

generator = kawaii_creator.models.Generator(GENERATOR_INPUT_DIMENTIONS)
if args.pretrained_generator is not None:
    kawaii_creator.utility.load_modelfile(args.pretrained_generator, generator)
discriminator = kawaii_creator.models.Discriminator()
if args.pretrained_discriminator is not None:
    kawaii_creator.utility.load_modelfile(args.pretrained_discriminator, discriminator)
if args.use_vectorizer:
    vectorizer = kawaii_creator.models.Vectorizer()
else:
    vectorizer = None
if args.pretrained_vectorizer is not None:
    kawaii_creator.utility.load_modelfile(args.pretrained_vectorizer, vectorizer)
if args.classifier_training_attribute_dataset is not None:
    classifier = kawaii_creator.models.Vectorizer(outdim=40)
if args.gpu >= 0:
    generator.to_gpu()
    discriminator.to_gpu()
    if args.use_vectorizer:
        vectorizer.to_gpu()
    if args.classifier_training_attribute_dataset is not None:
        classifier.to_gpu()

updater = kawaii_creator.updaters.Updater(
    generator=generator, discriminator=discriminator, xp=xp, batchsize=batchsize,
    generator_input_dimentions=GENERATOR_INPUT_DIMENTIONS)
if args.use_vectorizer:
    vectorizer_updater = kawaii_creator.updaters.VectorizerUpdater(vectorizer)
if args.classifier_training_attribute_dataset is not None:
    classifier_updater = kawaii_creator.updaters.ClassifierUpdater(classifier)

count_processed, sum_loss_discriminator, sum_loss_generator, sum_accuracy, sum_loss_classifier, sum_accuracy_classifier = 0, 0, 0, 0, 0, 0
for batch in iterator | pipe.select(xp.array) | pipe.select(chainer.Variable):
    loss_generator = chainer.Variable(xp.zeros((), dtype=xp.float32))
    loss_discriminator = chainer.Variable(xp.zeros((), dtype=xp.float32))
    loss_vectorizer = chainer.Variable(xp.zeros((), dtype=xp.float32))
    loss_classifier = chainer.Variable(xp.zeros((), dtype=xp.float32))

    if args.generator_training:
        # forward
        generated, random_seed = updater.generate_random()
        discriminated_from_generated = updater.discriminator(generated)
        discriminated_from_dataset = updater.discriminator(batch)
        accuracy = updater.discriminator_accuracy(discriminated_from_generated=discriminated_from_generated,
                                                  discriminated_from_dataset=discriminated_from_dataset)
        sum_accuracy += chainer.cuda.to_cpu(accuracy.data)  # update generator
        loss_generator_each = updater.loss_generator(discriminated_from_generated=discriminated_from_generated)
        loss_generator += loss_generator_each
        sum_loss_generator += chainer.cuda.to_cpu(loss_generator_each.data)

        # update discriminator
        if (not args.use_accuracy_threshold) or accuracy.data < 0.8:
            loss_discriminator_this = updater.loss_discriminator(
                discriminated_from_generated=discriminated_from_generated,
                discriminated_from_dataset=discriminated_from_dataset)
            loss_discriminator += loss_discriminator_this * args.classifier_loss_weight
            sum_loss_discriminator += chainer.cuda.to_cpu(loss_discriminator_this.data)

        updater.optimizer_generator.zero_grads()
        loss_generator.backward()

        updater.optimizer_discriminator.zero_grads()



        loss_discriminator.backward()

        if args.classifier_training_attribute_dataset is not None:
            classifier_updater.optimizer.zero_grads()
            array_batch = next(classifier_attribute_training_iterator)
            image_array = xp.array([each[0] for each in array_batch])
            label_array = xp.array([each[1] for each in array_batch], dtype=xp.int32)
            classified = classifier(generator(vectorizer(chainer.Variable(image_array))))
            loss_classifier += chainer.functions.sigmoid_cross_entropy(
                classified,
                chainer.Variable(label_array)
            )
            sum_accuracy_classifier += ((chainer.functions.sigmoid(
                classified).data > 0.5) == label_array).sum() / 40 / batchsize
            sum_loss_classifier += chainer.cuda.to_cpu(loss_classifier.data)
            loss_classifier.backward()
            classifier_updater.optimizer.update()

        updater.optimizer_discriminator.update()
        updater.optimizer_generator.update()

    # update vectorizer
    if args.use_vectorizer:
        generated, random_seed = updater.generate_random(batchsize=1, test=True)
        generated_augmented = chainer.Variable(
            xp.array([augment(chainer.cuda.to_cpu(generated.data[0]).transpose(1, 2, 0)).transpose(2, 0, 1)]))
        vectorized = vectorizer(generated_augmented)
        vectorized = vectorizer(generated)
        vectorizer_updater.update_vectorizer(seed=random_seed, vectorized=vectorized)

    # finetune generator with vectorizer
    if args.vectorizer_training_dataset is not None:
        vectorizer_training_batch = chainer.Variable(xp.array(next(vectorizer_training_iterator)))
        vectorized = chainer.Variable(
            xp.clip(vectorizer(vectorizer_training_batch).data, -1, 1))
        generated_from_vectorized = generator(vectorized)
        discriminated_from_vectorizer_training = updater.discriminator(chainer.Variable(generated_from_vectorized.data))
        updater.loss_generator(discriminated_from_generated=discriminated_from_vectorizer_training)

    report_span = batchsize * 10
    save_span = batchsize * 100
    count_processed += len(batch.data)
    if count_processed % report_span == 0:
        logging.info("processed: {}".format(count_processed))
        logging.info("accuracy_discriminator: {}".format(sum_accuracy * batchsize / report_span))
        logging.info("accuracy_classifier: {}".format(sum_accuracy_classifier * batchsize / report_span))
        logging.info("loss_classifier: {}".format(sum_loss_classifier / report_span))
        logging.info("loss_discriminator: {}".format(sum_loss_discriminator / report_span))
        logging.info("loss_generator: {}".format(sum_loss_generator / report_span))
        sum_loss_discriminator, sum_loss_generator, sum_accuracy, sum_loss_classifier, sum_accuracy_classifier = 0, 0, 0, 0, 0
    if count_processed % save_span == 0:
        chainer.serializers.save_npz(
            os.path.join(OUTPUT_DIRECTORY, "discriminator_model_{}.npz".format(count_processed)), discriminator)
        chainer.serializers.save_npz(
            os.path.join(OUTPUT_DIRECTORY, "generator_model_{}.npz".format(count_processed)), generator)
        if args.use_vectorizer:
            chainer.serializers.save_npz(
                os.path.join(OUTPUT_DIRECTORY, "vectorizer_model_{}.npz".format(count_processed)), vectorizer)
        if args.classifier_training_attribute_dataset is not None:
            chainer.serializers.save_npz(
                os.path.join(OUTPUT_DIRECTORY, "classifier_model_{}.npz".format(count_processed)), classifier)
