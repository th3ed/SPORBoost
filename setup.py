#!/usr/bin/env python

from setuptools import setup

packages = [
      "sporgboost"
]

setup(name='sporgboost',
      version='0.1',
      description='Sparse Projection Oblique Randomer Gradient Boosting',
      author='Ed Andrews',
      author_email='ed4ndrews@gmail.com',
      tests_require=["pytest"],
      packages=packages
     )