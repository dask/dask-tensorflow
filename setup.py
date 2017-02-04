#!/usr/bin/env python

import os
from setuptools import setup
import sys

requires = open('requirements.txt').read().strip().split('\n')
install_requires = []
extras_require = {}
for r in requires:
    if ';' in r:
        # requirements.txt conditional dependencies need to be reformatted for wheels
        # to the form: `'[extra_name]:condition' : ['requirements']`
        req, cond = r.split(';', 1)
        cond = ':' + cond
        cond_reqs = extras_require.setdefault(cond, [])
        cond_reqs.append(req)
    else:
        install_requires.append(r)

setup(name='dask-tensorflow',
      version='0.0.1',
      description='Interactions between Dask and Tensorflow',
      maintainer='Matthew Rocklin',
      maintainer_email='mrocklin@gmail.com',
      license='BSD',
      install_requires=install_requires,
      extras_require=extras_require,
      packages=['dask_tensorflow'],
      long_description=(open('README.md').read() if os.path.exists('README.md')
                        else ''),
      zip_safe=False)
