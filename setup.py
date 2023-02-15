# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python

from setuptools import setup, find_packages
import shlex
import subprocess

print(find_packages(where="."))

def git_version():
    cmd = 'git log --format="%h" -n 1'
    return subprocess.check_output(shlex.split(cmd)).decode()


#version = git_version()
version = '1.1.0'
setup(
    name='isdf',
    version=version,
    author='Joe Ortiz',
    author_email='joeaortiz16@gmail.com',
    packages= find_packages(where="."),
    py_modules=[]
)
