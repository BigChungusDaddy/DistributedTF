import json
import os
import sys
import tensorflow as tf
import mnist_setup

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.pop('TF_CONFIG', None)


def main():
    batch_size = 64
    single_worker_dataset = mnist_setup.mnist_dataset(batch_size)
    single_worker_model = mnist_setup.build_and_compile_cnn_model()
    single_worker_model.fit(single_worker_dataset, epochs=3, steps_per_epoch=70)

if __name__ == '__main__':
    main()