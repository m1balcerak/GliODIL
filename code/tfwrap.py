import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TENSORFLOW_INTER_OP_PARALLELISM"] = "1"
os.environ["TENSORFLOW_INTRA_OP_PARALLELISM"] = "1"

if not int(os.environ.get("USEGPU", 0)):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if int(os.environ.get("NOWARN", 0)):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)

import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
    except RuntimeError as e:
        print(e)

