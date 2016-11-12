# -*- coding: utf-8 -*-
import argparse
import glob
import numpy
import chainer
import sys
import os
import pipe
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import kawaii_creator

GENERATOR_INPUT_DIMENTIONS = 100
OUTPUT_DIRECTORY = os.path.join(os.path.dirname(__file__), "..", "output", str(time.time()))
os.makedirs(OUTPUT_DIRECTORY)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--batchsize", type=int, default=10)
args = parser.parse_args()

if args.gpu >= 0:
    chainer.cuda.check_cuda_available()
    chainer.cuda.get_device(args.gpu).use()
    xp = chainer.cuda.cupy
else:
    xp = numpy

batchsize = args.batchsize

paths = glob.glob("{}/*".format(args.dataset))
dataset = kawaii_creator.datasets.ResizedImageDataset(paths=paths, resize=(96, 96))
iterator = chainer.iterators.SerialIterator(dataset, batch_size=10, repeat=True, shuffle=True)

generator = kawaii_creator.models.Generator(GENERATOR_INPUT_DIMENTIONS)
discriminator = kawaii_creator.models.Discriminator()

if args.gpu >= 0:
    generator.to_gpu()
    discriminator.to_gpu()
optimizer_generator = chainer.optimizers.Adam(alpha=0.0002, beta1=0.5)
optimizer_discriminator = chainer.optimizers.Adam(alpha=0.0002, beta1=0.5)
optimizer_generator.setup(generator)
optimizer_discriminator.setup(discriminator)
optimizer_generator.add_hook(chainer.optimizer.WeightDecay(0.00001))
optimizer_discriminator.add_hook(chainer.optimizer.WeightDecay(0.00001))

count_processed, sum_loss_discriminator, sum_loss_generator = 0, 0, 0
for batch in iterator | pipe.select(xp.array):
    variable_batch = chainer.Variable(batch)

    # Generator Round
    random_seed = chainer.Variable(
        xp.array(xp.random.uniform(-1, 1, (batchsize, GENERATOR_INPUT_DIMENTIONS)), dtype=xp.float32))
    generated = generator(random_seed)

    # Discriminator Round
    discriminated_from_generated = discriminator(generated)
    discriminated_from_dataset = discriminator(variable_batch)

    # update
    optimizer_generator.zero_grads()
    loss_generator = chainer.functions.softmax_cross_entropy(
        discriminated_from_generated,
        chainer.Variable(xp.zeros(discriminated_from_generated.data.shape[0], dtype=xp.int32))
    )
    sum_loss_generator += chainer.cuda.to_cpu(loss_generator.data)
    loss_generator.backward()
    optimizer_generator.update()

    optimizer_discriminator.zero_grads()
    loss_discriminator = chainer.functions.softmax_cross_entropy(
        discriminated_from_generated,
        chainer.Variable(xp.ones(discriminated_from_generated.data.shape[0], dtype=xp.int32))
    ) + chainer.functions.softmax_cross_entropy(
        discriminated_from_dataset,
        chainer.Variable(xp.zeros(discriminated_from_dataset.data.shape[0], dtype=xp.int32))
    )
    sum_loss_discriminator += chainer.cuda.to_cpu(loss_discriminator.data)
    loss_discriminator.backward()
    optimizer_discriminator.update()

    count_processed += len(batch)
    report_span = batchsize * 10
    if count_processed % report_span == 0:
        print("processed: {}".format(count_processed))
        print("loss_discriminator: {}".format(sum_loss_discriminator / report_span))
        print("loss_generator: {}".format(sum_loss_generator / report_span))
        sum_loss_discriminator, sum_loss_generator = 0, 0
        chainer.serializers.save_npz(
            os.path.join(OUTPUT_DIRECTORY, "discriminator_model_{}.npz".format(count_processed)), discriminator)
        chainer.serializers.save_npz(
            os.path.join(OUTPUT_DIRECTORY, "generator_model_{}.npz".format(count_processed)), generator)
