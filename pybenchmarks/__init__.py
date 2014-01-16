from __future__ import division
from __future__ import print_function

import gc
import itertools
import numpy as np
import os
import timeit
from collections import OrderedDict

__all__ = ['benchmark']
__version__ = '2.0'

keyword = {}


def benchmark(stmt, *args, **keywords):
    """
    Automate the creation of benchmark tables.

    This function times a code snippet by iterating through sequences
    of input variables. It returns a dictionary containing the elapsed time
    (all platforms) and memory usage (linux only) for each combination of
    the input variables.

    Parameters
    ----------
    stmt : callable or string
        The function or snippet to be timed. If is it a string, the arguments
        are matched by '*args' and the keywords by '**keywords'.
        Caveat: these must be interpreted correctly through their repr.
        Otherwise, they should be passed as a string (then both stmt and setup
        should be a string). See examples below.
    repeat : int, optional
        Number of times the timing is repeated (default: 3).
    setup : callable or string, optional
        Initialisation before timing. In case of a string, arguments and
        keywords are passed with the same mechanism as the stmt argument.

    Examples
    --------
    >>> import numpy as np
    >>> def f(dtype, n=10):
    ...     return np.zeros(n, dtype)
    >>> b = benchmark(f, (int, float), n=(10, 100))

    >>> b = benchmark('sleep(t)', t=(1, 2, 3), setup='from time import sleep')

    >>> shapes = (10, 100, 1000)
    >>> b = benchmark('np.dot(a, a)', shape=shapes,
    ...               setup='import numpy as np;'
    ...                     'a = np.random.random_sample(shape)')

    >>> b = benchmark('np.dot(a, b)', m=shapes, n=shapes,
    ...               setup='import numpy as np;'
    ...                     'a = np.random.random_sample((m, n));'
    ...                     'b = np.random.random_sample(n)')

    >>> class A():
    ...     def __init__(self, dtype, n=10):
    ...         self.a = 2 * np.ones(n, dtype)
    ...     def run(self):
    ...         np.sqrt(self.a)
    >>> b = benchmark('a.run()', dtype=('int', 'float'), n=(1, 10),
    ...               setup='from __main__ import A; a = A(dtype, n=n)')

    Overhead:
    >>> def f():
    ...     pass
    >>> b = benchmark(f)

    """
    global keyword

    if not callable(stmt) and not isinstance(stmt, str):
        raise TypeError(
            'The argument stmt is neither a string nor a callable.')

    repeat = keywords.pop('repeat', 3)
    setup = keywords.pop('setup', 'pass')
    if not callable(setup) and not isinstance(setup, str):
        raise TypeError(
            'The argument setup is neither a string nor a callable.')
    if isinstance(stmt, str) and len(args) > 0:
        raise ValueError('Variables should be passed through keywords.')

    # ensure args is a sequence of sequences
    args = tuple(a if isinstance(a, (list, tuple)) else [a] for a in args)
    # ensure keywords is a dict of sequences
    keywords = OrderedDict((str(k), keywords[k]
                            if isinstance(keywords[k], (list, tuple))
                            else [keywords[k]])
                           for k in sorted(keywords.keys()))

    iterargs = itertools.product(*args)
    iterkeys = _iterkeywords(keywords)
    iterinputs = itertools.product(iterargs, iterkeys)
    shape = tuple(len(_) for _ in args) + \
            tuple(len(v) for v in keywords.values())

    variables = OrderedDict()
    if len(args) > 0:
        variables['*args'] = args
    variables.update(keywords)
    result = {
        'variables': variables,
        'info': np.empty(shape, 'S256'),
        'time': np.empty(shape)}

    try:
        memory = memory_usage()
    except IOError:
        do_memory = False
        memory = {}
    else:
        do_memory = True
        result.update(dict((k, np.empty(shape)) for k in memory))

    if len(keywords) > 0:
        setup_init = (
            'from pybenchmarks import keyword;' +
            ';\n'.join("{0}=keyword['{0}']".format(k) for k in keywords) +
            ';\n')

    for iresult, (arg, keyword) in enumerate(iterinputs):

        if callable(stmt):
            class wrapper(object):
                def __call__(self):
                    stmt(*self.args, **self.keywords)
            w = wrapper()
            w.args = arg
            w.keywords = keyword
            t = timeit.Timer(w, setup=setup)
        else:
            t = timeit.Timer(stmt, setup=setup_init + setup)

        # determine number so that 0.2 <= total time < 2.0
        for i in range(10):
            number = 10**i
            x = t.timeit(number)
            if x >= 0.2:
                break

        # actual runs
        gc.collect()
        if do_memory:
            memory = memory_usage()
        if number > 1 or x <= 2:
            r = t.repeat(repeat, number)
        else:
            r = t.repeat(repeat-1, number)
            r = [x] + r
        best = min(r)
        if do_memory:
            memory = memory_usage(since=memory)

        info = _get_info(arg, keyword)
        usec = best * 1e6 / number
        if usec < 1:
            unit = 'ns'
            value = usec * 1000
        elif usec < 1000:
            unit = 'us'
            value = usec
        elif usec < 1000000:
            unit = 'ms'
            value = usec / 1000
        else:
            unit = 's'
            value = usec / 1000000
        print('{0}{1}{2} loops, best of {3}: {4:.2f} {5} per loop.{6}'.format(
              info, ': ' if info else '', number, repeat, value, unit,
              ' ' + ', '.join([k + ':' + str(v) + 'MiB'
                               for k, v in memory.items()])))

        result['info'].ravel()[iresult] = info
        result['time'].ravel()[iresult] = best / number
        for k in memory:
            result[k].ravel()[iresult] = memory[k]

    return result


def memory_usage(keys=('VmRSS', 'VmData', 'VmSize'), since=None):
    """
    Return a dict containing information about the process' memory usage.

    Parameters
    ----------
    keys : sequence of strings
        Process status identifiers (see /proc/###/status). Default are
        the resident, data and virtual memory sizes.
    since : dict
        Dictionary as returned by a previous call to memory_usage function and
        used to compute the difference of memory usage since then.

    """
    proc_status = '/proc/%d/status' % os.getpid()
    scale = {'kB': 1024, 'mB': 1024 * 1024,
             'KB': 1024, 'MB': 1024 * 1024}

    # get pseudo file  /proc/<pid>/status
    with open(proc_status) as f:
        status = f.read()

    result = {}
    for k in keys:
        # get VmKey line e.g. 'VmRSS:  9999  kB\n ...'
        i = status.index(k)
        v = status[i:].split(None, 3)  # whitespace
        if len(v) < 3:
            raise ValueError('Invalid format.')

        # convert Vm value to Mbytes
        result[k] = float(v[1]) * scale[v[2]] / 2**20

    if since is not None:
        if not isinstance(since, dict):
            raise TypeError('The input is not a dict.')
        common_keys = set(result.keys())
        common_keys.intersection_update(since.keys())
        result = dict((k, result[k] - since[k]) for k in common_keys)

    return result


def _get_info(args, keywords):
    id = ''
    if len(args) > 0:
        id = ', '.join(repr(a) for a in args)
    else:
        id = ''
    if len(keywords) > 0:
        if id != '':
            id += ', '
        id += ', '.join(str(k) + '=' + repr(v)
                        for k, v in keywords.items())
    return id


def _iterkeywords(keywords):
    keys = keywords.keys()
    itervalues = itertools.product(*keywords.values())
    for values in itervalues:
        yield dict((k, v) for k, v in zip(keys, values))
