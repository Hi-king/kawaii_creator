# -*- coding: utf-8 -*-
import argparse
import glob
import numpy
import chainer
import sys
import os
import pipe
import time
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import kawaii_creator

GENERATOR_INPUT_DIMENTIONS = 100
OUTPUT_DIRECTORY = os.path.join(os.path.dirname(__file__), "..", "output", str(time.time()))
os.makedirs(OUTPUT_DIRECTORY)

logging.basicConfig(filename=os.path.join(OUTPUT_DIRECTORY, "log.txt"), level=logging.DEBUG)
console = logging.StreamHandler()
logging.getLogger('').addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--batchsize", type=int, default=10)
parser.add_argument("--use_accuracy_threshold", action="store_true")
parser.add_argument("--use_vectorizer", action="store_true")
args = parser.parse_args()
logging.info(args)

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

generator = kawaii_creator.models.Generator(GENERATOR_INPUT_DIMENTIONS)
discriminator = kawaii_creator.models.Discriminator()
if args.use_vectorizer:
    vectorizer = kawaii_creator.models.Vectorizer()
else:
    vectorizer = None
if args.gpu >= 0:
    generator.to_gpu()
    discriminator.to_gpu()
    if args.use_vectorizer:
        vectorizer.to_gpu()

updater = kawaii_creator.updaters.Updater(
    generator=generator, discriminator=discriminator, xp=xp, batchsize=batchsize,
    generator_input_dimentions=GENERATOR_INPUT_DIMENTIONS)
vectorizer_updater = kawaii_creator.updaters.VectorizerUpdater(vectorizer)

count_processed, sum_loss_discriminator, sum_loss_generator, sum_accuracy = 0, 0, 0, 0
for batch in iterator | pipe.select(xp.array):
    variable_batch = chainer.Variable(batch)

    # forward
    generated, random_seed = updater.generate_random()
    discriminated_from_generated = updater.discriminator(generated)
    discriminated_from_dataset = updater.discriminator(variable_batch)
    accuracy = updater.discriminator_accuracy(discriminated_from_generated=discriminated_from_generated,
                                              discriminated_from_dataset=discriminated_from_dataset)
    sum_accuracy += chainer.cuda.to_cpu(accuracy.data)

    # update generator
    sum_loss_generator += updater.update_generator(discriminated_from_generated=discriminated_from_generated)

    # update discriminator
    if (not args.use_accuracy_threshold) or accuracy.data < 0.8:
        sum_loss_discriminator += updater.update_discriminator(
            discriminated_from_generated=discriminated_from_generated,
            discriminated_from_dataset=discriminated_from_dataset)

    # update vectorizer
    if args.use_vectorizer:
        vectorized = vectorizer(generated)
        vectorizer_updater.update_vectorizer(seed=random_seed, vectorized=vectorized)

    count_processed += len(batch)
    report_span = batchsize * 100
    save_span = batchsize * 1000
    if count_processed % report_span == 0:
        logging.info("processed: {}".format(count_processed))
        logging.info("accuracy_discriminator: {}".format(sum_accuracy * batchsize / report_span))
        logging.info("loss_discriminator: {}".format(sum_loss_discriminator / report_span))
        logging.info("loss_generator: {}".format(sum_loss_generator / report_span))
        sum_loss_discriminator, sum_loss_generator, sum_accuracy = 0, 0, 0
    if count_processed % save_span == 0:
        chainer.serializers.save_npz(
            os.path.join(OUTPUT_DIRECTORY, "discriminator_model_{}.npz".format(count_processed)), discriminator)
        chainer.serializers.save_npz(
            os.path.join(OUTPUT_DIRECTORY, "generator_model_{}.npz".format(count_processed)), generator)
        if args.use_vectorizer:
            chainer.serializers.save_npz(
                os.path.join(OUTPUT_DIRECTORY, "vectorizer_model_{}.npz".format(count_processed)), vectorizer)
