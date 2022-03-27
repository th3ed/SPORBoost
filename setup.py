#!/usr/bin/env python

from setuptools import setup

packages = [
      "sporboost"
]

setup(name='sporboost',
      version='2022.3.0',
      description='Sparse Projection Oblique Randomer Boosting',
      author='Ed Andrews',
      author_email='ed4ndrews@gmail.com',
      tests_require=["pytest"],
      packages=packages
     )