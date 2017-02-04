from collections import defaultdict

from toolz import merge
from tornado import gen

from distributed.utils import sync
from distributed.comm.tcp import parse_host_port
import tensorflow as tf


def start_and_attach_server(spec, job_name=None, task_index=None, dask_worker=None):
    server = tf.train.Server(spec, job_name=job_name, task_index=task_index)
    dask_worker.tensorflow_server = server
    return 'OK'


@gen.coroutine
def _start_tensorflow(client):
    info = yield client.scheduler.identity()
    if not info['workers']:
        return

    ports = defaultdict(lambda: 2221)
    spec = {'worker': []}
    task_index = {}
    for i, w in enumerate(info['workers']):
        host = parse_host_port(w)[0].strip('/')
        ports[host] += 1
        spec['worker'].append('%s:%d' % (host, ports[host]))
        task_index[w] = i

    spec = tf.train.ClusterSpec(spec)

    resp = yield {w: client._run(start_and_attach_server, spec,
                                 job_name='worker',
                                 task_index=task_index[w],
                                 workers=[w]) for w in info['workers']}
    resp = merge(resp.values())
    if not all(v == 'OK' for v in resp.values()):
        raise ValueError("Setup did not succeed")

    return spec


def start_tensorflow(client, **kwargs):
    return sync(client.loop, _start_tensorflow, client, **kwargs)
