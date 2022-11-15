import json
import os
import sys
import tensorflow as tf
import mnist_setup
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.pop('TF_CONFIG', None)

def main(worker, batch_size):
    tf_config = {
        'cluster': {
            'worker': ['192.168.1.1:12345', '192.168.1.2:23456', '192.168.1.3:23456', '192.168.1.4:23456']
            },
            'task': {'type': 'worker', 'index': 0}
        }
    if worker != 0:
        tf_config['task']['index'] = worker
        
    os.environ['TF_CONFIG'] = json.dumps(tf_config)
    tf_config = json.loads(os.environ['TF_CONFIG'])

    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO

    global_batch_size = batch_size
    multi_worker_dataset = mnist_setup.mnist_dataset(global_batch_size).with_options(options)


    with strategy.scope():
    # Model building/compiling need to be within `strategy.scope()`.
        multi_worker_model = mnist_setup.build_and_compile_cnn_model()
    
    start = time.time()
    multi_worker_model.fit(multi_worker_dataset, epochs=100, steps_per_epoch=70)
    print(time.time() - start)

if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]))