#!/usr/bin/env python

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('cmblensing',parent_package,top_path)
    config.add_extension('wignerd', ['cmblensing/wignerd.c', 'cmblensing/wignerd.pyf'])
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(name='cmblensing',
          packages=['cmblensing'],
          configuration=configuration)
