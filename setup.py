# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python

from setuptools import setup
import shlex
import subprocess


def git_version():
    cmd = 'git log --format="%h" -n 1'
    return subprocess.check_output(shlex.split(cmd)).decode()


version = git_version()

setup(
    name='incSDF',
    version=version,
    author='Joe Ortiz',
    author_email='joeaortiz16@gmail.com',
    py_modules=[]
)
