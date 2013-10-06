import os
import re
from distutils.core import setup


def version():
    f = open(os.path.join('pybenchmarks', '__init__.py')).read()
    m = re.search(r"__version__ = '(.*)'", f)
    return m.groups()[0]

setup(name='pybenchmarks',
      description='Automate benchmark tables',
      long_description=open('README.rst').read(),
      url='http://github.com/pchanial/pybenchmarks',
      author='Pierre Chanial',
      author_email='pierre.chanial@gmail.com',
      packages=['pybenchmarks'],
      version=version(),
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'License :: Public Domain',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.6',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Topic :: Software Development'])
