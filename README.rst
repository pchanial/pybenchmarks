===========
PyBenchmark
===========

Automate the creation of benchmark tables.

The benchmark function times a code snippet by iterating through sequences of
input arguments and keywords. It returns a dictionary containing
the elapsed time (all platforms) and memory usage (linux only)
for each combination of input arguments and keywords.


========
Examples
========

>>> import numpy as np
>>> def f(dtype, n=10):
...     return np.zeros(n, dtype)
>>> b = benchmark(f, (int, float), n=(10, 100))
<type 'int'>,n=10: 1000000 loops, best of 3: 1.52 us per loop.
<type 'int'>,n=100: 1000000 loops, best of 3: 1.69 us per loop.
<type 'float'>,n=10: 1000000 loops, best of 3: 1.53 us per loop.
<type 'float'>,n=100: 1000000 loops, best of 3: 1.7 us per loop.


>>> benchmark('sleep(1)', setup='from time import sleep')

>>> shapes = (10, 100, 1000)
>>> b = benchmark('np.dot(a, a)', shapes,
...               setup='import numpy as np;'
...                     'a = np.random.random_sample(*args)')

>>> b = benchmark('np.dot(a, a)', size=shapes,
...               setup='import numpy as np;'
...                     'a = np.random.random_sample(**keywords)')

>>> b = benchmark('np.dot(a, b)', shapes, shapes,
...               setup='import numpy as np;'
...                     'm, n = *args;'
...                     'a = np.random.random_sample((m, n));'
...                     'b = np.random.random_sample(n)')

>>> class A():
...     def __init__(self, dtype, n=10):
...         self.a = 2 * np.ones(n, dtype)
...     def run(self):
...         np.sqrt(self.a)
>>> b = benchmark('a.run()', ('int', 'float'), n=(1, 10),
...               setup='from __main__ import A; a=A(*args, **keywords)')