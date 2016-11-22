# -*- coding: utf-8 -*-
import os

import chainer


def load_modelfile(path, model):
    if os.path.splitext(path)[1] == ".npz":
        chainer.serializers.load_npz(path, model)
    elif os.path.splitext(path)[1] == ".h5":
        chainer.serializers.load_hdf5(path, model)
    else:
        raise Exception()
