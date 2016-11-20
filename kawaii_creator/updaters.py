# -*- coding: utf-8 -*-
import chainer


class Updater(object):
    def __init__(self, generator, discriminator, xp, batchsize, generator_input_dimentions):
        self.generator_input_dimentions = generator_input_dimentions
        self.batchsize = batchsize
        self.generator = generator
        self.discriminator = discriminator
        self.xp = xp

        self.optimizer_generator = chainer.optimizers.Adam(alpha=0.0002, beta1=0.5)
        self.optimizer_discriminator = chainer.optimizers.Adam(alpha=0.0002, beta1=0.5)
        self.optimizer_generator.setup(generator)
        self.optimizer_discriminator.setup(discriminator)
        self.optimizer_generator.add_hook(chainer.optimizer.WeightDecay(0.00001))
        self.optimizer_discriminator.add_hook(chainer.optimizer.WeightDecay(0.00001))

    def generate_random(self, batchsize=None):
        if batchsize is None:
            batchsize = self.batchsize
        random_seed = chainer.Variable(
            self.xp.array(self.xp.random.uniform(-1, 1, (batchsize, self.generator_input_dimentions)),
                          dtype=self.xp.float32))
        return self.generator(random_seed), random_seed

    def discriminator_accuracy(self, discriminated_from_generated, discriminated_from_dataset):
        return (chainer.functions.accuracy(
            discriminated_from_generated,
            chainer.Variable(self.xp.ones(discriminated_from_generated.data.shape[0], dtype=self.xp.int32))
        ) + chainer.functions.accuracy(
            discriminated_from_dataset,
            chainer.Variable(self.xp.zeros(discriminated_from_dataset.data.shape[0], dtype=self.xp.int32))
        )) / 2

    def update_generator(self, discriminated_from_generated):
        self.optimizer_generator.zero_grads()
        loss_generator = chainer.functions.softmax_cross_entropy(
            discriminated_from_generated,
            chainer.Variable(self.xp.zeros(discriminated_from_generated.data.shape[0], dtype=self.xp.int32))
        )
        loss_generator.backward()
        self.optimizer_generator.update()
        return chainer.cuda.to_cpu(loss_generator.data)

    def update_discriminator(self, discriminated_from_generated, discriminated_from_dataset):
        self.optimizer_discriminator.zero_grads()
        loss_discriminator = chainer.functions.softmax_cross_entropy(
            discriminated_from_generated,
            chainer.Variable(self.xp.ones(discriminated_from_generated.data.shape[0], dtype=self.xp.int32))
        ) + chainer.functions.softmax_cross_entropy(
            discriminated_from_dataset,
            chainer.Variable(self.xp.zeros(discriminated_from_dataset.data.shape[0], dtype=self.xp.int32))
        )
        loss_discriminator.backward()
        self.optimizer_discriminator.update()
        return chainer.cuda.to_cpu(loss_discriminator.data)


class VectorizerUpdater(object):
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer
        self.optimizer = chainer.optimizers.Adam()
        self.optimizer.setup(vectorizer)
        self.optimizer.add_hook(chainer.optimizer.WeightDecay(10 ** -5))

    def update_vectorizer(self, seed, vectorized: chainer.Variable):
        self.optimizer.zero_grads()
        loss = chainer.functions.mean_squared_error(
            vectorized,
            seed
        )
        loss.backward()
        self.optimizer.update()
