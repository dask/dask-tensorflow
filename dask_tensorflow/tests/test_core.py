import tensorflow as tf

from distributed.utils_test import gen_cluster
from dask_tensorflow import _start_tensorflow

@gen_cluster(client=True)
def test_basic(c, s, a, b):
    spec = yield _start_tensorflow(c)
    assert isinstance(a.tensorflow_server, tf.train.Server)
    assert isinstance(b.tensorflow_server, tf.train.Server)
