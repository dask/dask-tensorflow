import tensorflow as tf

from distributed import Client
from distributed.utils_test import gen_cluster, loop, cluster
from dask_tensorflow import _start_tensorflow, start_tensorflow

@gen_cluster(client=True)
def test_basic(c, s, a, b):
    spec, d = yield _start_tensorflow(c)
    assert isinstance(a.tensorflow_server, tf.train.Server)
    assert isinstance(b.tensorflow_server, tf.train.Server)
    assert isinstance(spec, tf.train.ClusterSpec)
    assert sum(map(len, spec.as_dict().values())) == len(s.workers)
    assert set(d) == set(s.workers)


def test_basic_sync(loop):
    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop) as c:
            spec, d = start_tensorflow(c)
            assert isinstance(spec, tf.train.ClusterSpec)


@gen_cluster(client=True, ncores=[('127.0.0.1', 1)] * 10)
def test_user_defined_spec(c, s, *workers):
    spec, d = yield _start_tensorflow(c, ps=2, worker=4)
    assert sum(hasattr(w, 'tensorflow_server') for w in workers) == 6
    spec = spec.as_dict()
    assert len(spec['ps']) == 2
    assert len(spec['worker']) == 4

    assert set(d).issubset(s.workers)
    import pdb; pdb.set_trace()
