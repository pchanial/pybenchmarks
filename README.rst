============
PyBenchmarks
============

Automate the creation of benchmark tables.

The benchmark utility times one or more code snippets or functions by iterating
through input arguments or keyed variables. It returns a dictionary
containing the elapsed time (all platforms) and memory usage (linux only)
for each combination of the input variables. An argument or keyed variable
is iterated if and only if it is a list, a tuple, a generator or a range.


========
Examples
========

>>> import numpy as np
>>> from pybenchmarks import benchmark
>>> b = benchmark((np.empty, np.ones), (100, 10000, 1000000),
...               dtype=(int, complex))
empty: 100     dtype=int       1.62 us
ones : 100     dtype=int       3.61 us
empty: 100     dtype=complex   1.70 us
ones : 100     dtype=complex   5.42 us
empty: 10000   dtype=int       1.53 us
ones : 10000   dtype=int       7.53 us
empty: 10000   dtype=complex   2.33 us
ones : 10000   dtype=complex  16.29 us
empty: 1000000 dtype=int       1.87 us
ones : 1000000 dtype=int       1.84 ms
empty: 1000000 dtype=complex   2.19 us
ones : 1000000 dtype=complex   4.20 ms

>>> b = benchmark(['np.empty(n, dtype=dtype)', 'np.ones(n, dtype=dtype)'],
...               dtype=(int, complex), n=(100, 10000, 1000000),
...               setup='from __main__ import np')
'np.empty(n, dt...: dtype=int     n=100       1.36 us
'np.ones(n, dty...: dtype=int     n=100       2.83 us
'np.empty(n, dt...: dtype=complex n=100       1.44 us
'np.ones(n, dty...: dtype=complex n=100       3.50 us
'np.empty(n, dt...: dtype=int     n=10000     1.22 us
'np.ones(n, dty...: dtype=int     n=10000     7.05 us
'np.empty(n, dt...: dtype=complex n=10000     1.35 us
'np.ones(n, dty...: dtype=complex n=10000    23.78 us
'np.empty(n, dt...: dtype=int     n=1000000   1.47 us
'np.ones(n, dty...: dtype=int     n=1000000   2.04 ms
'np.empty(n, dt...: dtype=complex n=1000000   2.91 us
'np.ones(n, dty...: dtype=complex n=1000000   4.26 ms

>>> import time
>>> benchmark(time.sleep, (1, 2, 3))
1   1.00 s
2   2.00 s
3   3.00 s

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

>>> def f(x, n, start=1):
...     out = start
...     for i in range(n):
...         out *= x
...     return out
>>> b = benchmark(f, np.full(10, 2), xrange(10), start=2.)
0   1.09 us
1   4.15 us
2   5.25 us
3   5.53 us
4  13.10 us
5   9.23 us
6   9.69 us
7  10.46 us
8  13.03 us
9  10.77 us
