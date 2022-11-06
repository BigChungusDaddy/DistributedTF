import json
import os
import sys
import tensorflow as tf
import mnist_setup
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.pop('TF_CONFIG', None)

def main(worker):
    tf_config = {
        'cluster': {
            'worker': ['192.168.1.1:12345', '192.168.1.2:23456']
            },
            'task': {'type': 'worker', 'index': 0}
        }
    if worker == 1:
        tf_config['task']['index'] = 1
        
    os.environ['TF_CONFIG'] = json.dumps(tf_config)
    per_worker_batch_size = 64
    tf_config = json.loads(os.environ['TF_CONFIG'])
    num_workers = len(tf_config['cluster']['worker'])

    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    global_batch_size = per_worker_batch_size * num_workers
    multi_worker_dataset = mnist_setup.mnist_dataset(global_batch_size)

    with strategy.scope():
    # Model building/compiling need to be within `strategy.scope()`.
        multi_worker_model = mnist_setup.build_and_compile_cnn_model()
    
    start = time.time()
    multi_worker_model.fit(multi_worker_dataset, epochs=100, steps_per_epoch=70)
    print(time.time() - start)

if __name__ == '__main__':
    main(int(sys.argv[1]))