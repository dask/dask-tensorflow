from collections import defaultdict

from toolz import merge
from tornado import gen

from queue import Queue
from distributed.utils import sync
import tensorflow as tf


def start_and_attach_server(spec, job_name=None, task_index=None, dask_worker=None):
    server = tf.train.Server(spec, job_name=job_name, task_index=task_index)
    dask_worker.tensorflow_server = server
    dask_worker.tensorflow_queue = Queue()
    return 'OK'


@gen.coroutine
def _start_tensorflow(client, **job_counts):
    info = yield client.scheduler.identity()
    if not info['workers']:
        return

    if not job_counts:
        job_counts = {'worker': len(info['workers'])}

    if sum(job_counts.values()) > len(info['workers']):
        raise ValueError("Dask cluster not large enough."
                         "Need %d workers, have %d"
                         % (sum(job_counts.values()), len(info['workers'])))

    ports = defaultdict(lambda: 2221)
    tf_spec = {job_name: [] for job_name in job_counts}
    dask_spec = {job_name: [] for job_name in job_counts}
    job_names = {}
    task_index = {}
    workers = iter(info['workers'])
    for job_name, count in job_counts.items():
        for i in range(count):
            w = next(workers)
            host = w.split('://')[-1].rsplit(':')[0]
            ports[host] += 1
            tf_name = '%s:%d' % (host, ports[host])
            tf_spec[job_name].append(tf_name)
            dask_spec[job_name].append(w)
            task_index[w] = i
            job_names[w] = job_name

    tf_spec = tf.train.ClusterSpec(tf_spec)

    resp = yield {w: client._run(start_and_attach_server, tf_spec,
                                 job_name=job_names[w],
                                 task_index=task_index[w],
                                 workers=[w]) for w in task_index}
    resp = merge(resp.values())
    if not all(v == 'OK' for v in resp.values()):
        raise ValueError("Setup did not succeed")

    raise gen.Return((tf_spec, dask_spec))


def start_tensorflow(client, **kwargs):
    """ Start Tensorflow on Dask Cluster

    This launches Tensorflow Servers alongside Dask workers

    Examples
    --------
    >>> client = Client('dask-scheduler-address:8786')
    >>> tf_spec, dask_spec = start_tensorflow(client)
    >>> tf_spec.as_dict()
    {'worker': ['192.168.1.100:2222', '192.168.1.101:2222']}

    Specify desired number of jobs types as keyword args

    >>> tf_spec, dask_spec = start_tensorflow(client, ps=2, worker=4)
    >>> tf_spec.as_dict()
    {'worker': ['192.168.1.100:2222', '192.168.1.101:2222',
                '192.168.1.102:2222', '192.168.1.103:2222'],
     'ps': ['192.168.1.104:2222', '192.168.1.105:2222']}
    """
    return sync(client.loop, _start_tensorflow, client, **kwargs)
