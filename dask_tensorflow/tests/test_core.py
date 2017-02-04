import tensorflow as tf

from distributed import Client
from distributed.utils_test import gen_cluster, loop, cluster
from dask_tensorflow import _start_tensorflow, start_tensorflow

@gen_cluster(client=True)
def test_basic(c, s, a, b):
    spec = yield _start_tensorflow(c)
    assert isinstance(a.tensorflow_server, tf.train.Server)
    assert isinstance(b.tensorflow_server, tf.train.Server)
    assert isinstance(spec, tf.train.ClusterSpec)
    assert sum(map(len, spec.as_dict().values())) == len(s.workers)


def test_basic_sync(loop):
    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop) as c:
            spec = start_tensorflow(c)
            assert isinstance(spec, tf.train.ClusterSpec)
