============
PyBenchmarks
============

Automate the creation of benchmark tables.

The benchmark function times one or more code snippets by iterating
through sequences of input keywords. It returns a dictionary containing
the elapsed time (all platforms) and optionally memory usage (linux only)
for each combination of code snippets and keywords.


========
Examples
========

>>> import numpy as np
>>> from pybenchmarks import benchmark
>>> f1 = np.empty
>>> f2 = np.ones
>>> b = benchmark('f(n, dtype=dtype)', f=(f1, f2),
...               dtype=(int, complex), n=(100, 10000, 1000000))
dtype=int     f=empty n=100       1.48 us
dtype=complex f=empty n=100       1.35 us
dtype=int     f=ones  n=100       5.29 us
dtype=complex f=ones  n=100       4.15 us
dtype=int     f=empty n=10000     1.21 us
dtype=complex f=empty n=10000     1.82 us
dtype=int     f=ones  n=10000     9.61 us
dtype=complex f=ones  n=10000    12.55 us
dtype=int     f=empty n=1000000   2.57 us
dtype=complex f=empty n=1000000   1.18 us
dtype=int     f=ones  n=1000000   2.41 ms
dtype=complex f=ones  n=1000000   5.16 ms

>>> import time
>>> f = time.sleep
>>> benchmark('f(t)', t=(1, 2, 3), setup='from __main__ import f')
t=1   1.00 s
t=2   2.00 s
t=3   3.00 s

>>> shapes = (100, 10000, 1000000)
>>> setup = """
... import numpy as np
... a = np.random.random_sample(shape)
... """
>>> b = benchmark('np.dot(a, a)', shape=shapes, setup=setup)
shape=100       1.38 us
shape=10000     6.33 us
shape=1000000 855.44 us

>>> shapes = (10, 100, 1000)
>>> setup="""
... import numpy as np
... a = np.random.random_sample((m, n))
... b = np.random.random_sample(n)
... """
>>> b = benchmark('np.dot(a, b)', m=shapes, n=shapes, setup=setup)
m=10   n=10     1.08 us
m=100  n=10     1.61 us
m=1000 n=10     6.91 us
m=10   n=100    1.48 us
m=100  n=100    4.16 us
m=1000 n=100   20.69 us
m=10   n=1000   4.42 us
m=100  n=1000  39.23 us
m=1000 n=1000 931.04 us
