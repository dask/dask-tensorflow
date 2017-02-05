import tensorflow as tf
from toolz import valmap

from distributed import Client
from distributed.utils_test import gen_cluster, loop, cluster
from dask_tensorflow import _start_tensorflow, start_tensorflow

@gen_cluster(client=True)
def test_basic(c, s, a, b):
    tf_spec, dask_spec = yield _start_tensorflow(c)
    assert isinstance(a.tensorflow_server, tf.train.Server)
    assert isinstance(b.tensorflow_server, tf.train.Server)
    assert isinstance(tf_spec, tf.train.ClusterSpec)
    assert sum(map(len, tf_spec.as_dict().values())) == len(s.workers)
    assert valmap(len, tf_spec.as_dict()) == valmap(len, dask_spec)
    assert set(sum(dask_spec.values(), [])) == s.workers


def test_basic_sync(loop):
    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop) as c:
            tf_spec, dask_spec = start_tensorflow(c)
            assert isinstance(tf_spec, tf.train.ClusterSpec)


@gen_cluster(client=True, ncores=[('127.0.0.1', 1)] * 10)
def test_user_defined_spec(c, s, *workers):
    tf_spec, dask_spec = yield _start_tensorflow(c, ps=2, worker=4)
    assert sum(hasattr(w, 'tensorflow_server') for w in workers) == 6
    tf_spec = tf_spec.as_dict()
    assert len(tf_spec['ps']) == 2
    assert len(tf_spec['worker']) == 4
    assert valmap(len, tf_spec) == valmap(len, dask_spec)
