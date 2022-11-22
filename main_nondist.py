import json
import os
import sys
import mnist_setup
import time

def main(batch_size):
    single_worker_dataset = mnist_setup.mnist_dataset(batch_size)
    single_worker_model = mnist_setup.build_and_compile_cnn_model()
    start = time.time()
    single_worker_model.fit(single_worker_dataset, epochs=100, steps_per_epoch=75)
    print(time.time() - start)

if __name__ == '__main__':
    main(int(sys.argv[1]))