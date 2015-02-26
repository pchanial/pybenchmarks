"""
Module to easily create benchmark tables.

"""
from __future__ import division
from __future__ import print_function

import gc
import itertools
import numpy as np
import os
import re
import timeit
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

__all__ = ['benchmark']
__version__ = '2.2'

keyword = {}
refortran = re.compile('^([a-z0-9,_ ]+ = )?([a-z0-9_]+)\(', re.I)


def benchmark(stmts, *args, **keywords):
    """
    Automate the creation of benchmark tables.

    This function times one or more code snippets by iterating through
    sequences of keyed input variables. It returns a dictionary containing
    the elapsed time (all platforms) and memory usage (linux only) for each
    combination of the input variables.

    Parameters
    ----------
    stmts : string, callable, or sequence of
        The code snippet(s) or functions to be timed. The input arguments and
        keywords are iterated and their values are passed to the function or
        the code snippet.
    repeat : int, optional
        Number of times the timing is repeated (default: 3).
    maxloop : int, optional
        Maximum number of loops (default: 100).
    setup : string, optional
        Initialisation before timing. The input keywords are passed with
        the same mechanism as the stmt argument.
    memory_usage : boolean, optional
        If True, print memory usage (default is False)
    verbose : boolean or integer, optional
        If False (or 0), don't print the benchmark results. If 2, also print
        the number of loops and repeats. Default is True (or 1).

    Examples
    --------
    >>> import numpy as np
    >>> from pybenchmarks import benchmark
    >>> b = benchmark('f(n, dtype=dtype)', f=(np.empty, np.ones),
    ...               dtype=(int, complex), n=(100, 10000, 1000000))
    >>> b = benchmark((np.empty, np.ones), (100, 10000, 1000000),
    ...               dtype=(int, complex))

    >>> import time
    >>> f = time.sleep
    >>> benchmark(f, (1, 2, 3))
    >>> benchmark('f(t)', t=(1, 2, 3), setup='from __main__ import f')

    >>> shapes = (100, 10000, 1000000)
    >>> setup = '''
    ... import numpy as np
    ... a = np.random.random_sample(shape)
    ... '''
    >>> b = benchmark('np.dot(a, a)', shape=shapes, setup=setup)

    >>> shapes = (10, 100, 1000)
    >>> setup='''
    ... import numpy as np
    ... a = np.random.random_sample((m, n))
    ... b = np.random.random_sample(n)
    ... '''
    >>> b = benchmark('np.dot(a, b)', m=shapes, n=shapes, setup=setup)

    Overhead:
    >>> b = benchmark('pass')

    """
    global keyword

    if callable(stmts) or isinstance(stmts, str):
        stmts = (stmts,)
        shape_stmts = ()
    else:
        shape_stmts = (len(stmts),)
    if any(not callable(_) and not isinstance(_, str) for _ in stmts):
        raise TypeError(
            'The argument stmts is not a string not a sequence of strings.')

    repeat = keywords.pop('repeat', 3)
    setup = keywords.pop('setup', 'pass')
    if not callable(setup) and not isinstance(setup, str):
        raise TypeError(
            'The argument setup is not string.')
    if isinstance(stmts[0], str) and len(args) > 0:
        raise ValueError('Variables should be passed through keywords.')
    do_memory = keywords.pop('memory_usage', False)
    maxloop = keywords.pop('maxloop', 100)
    if maxloop < 1:
        raise ValueError('Invalid value for maxloop.')
    verbose = keywords.pop('verbose', False)

    # ensure args is a sequence of sequences
    args = tuple(a if isinstance(a, (list, tuple)) else [a] for a in args)
    # ensure keywords is a dict of sequences
    keywords = OrderedDict((str(k), keywords[k]
                            if isinstance(keywords[k], (list, tuple))
                            else [keywords[k]])
                           for k in sorted(keywords.keys()))

    iterargs = itertools.product(*args)
    iterkeys = _iterkeywords(keywords)
    iterinputs = itertools.product(iterargs, iterkeys, stmts)
    shape = tuple(len(_) for _ in reversed(args)) + \
            tuple(len(_) for _ in reversed(keywords.values())) + shape_stmts
    nbenches = np.product(shape)

    variables = OrderedDict({'stmts': stmts})
    if len(args) > 0:
        variables['*args'] = args
    variables.update(keywords)
    result = {
        'variables': variables,
        'setup': setup,
        'info': np.zeros(nbenches, 'S256'),
        'time': np.zeros(nbenches)}

    if do_memory:
        try:
            memory = memory_usage()
        except IOError:
            do_memory = False
        else:
            result.update(dict((k, np.zeros(nbenches)) for k in memory))

    if len(keywords) > 0:
        setup_init = (
            'from pybenchmarks import keyword;' +
            ';\n'.join("{0}=keyword['{0}']".format(k) for k in keywords) +
            ';\n')

    # compute column sizes
    info_nspaces = _get_info_nspaces(args, keywords)

    # iterate through the keyed inputs
    for iresult, (arg, keyword, stmt) in enumerate(iterinputs):

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

        # determine number so that 0.1 <= total time < 1.0
        for i in range(10):
            number = 10**i
            if number > maxloop:
                break
            x = t.timeit(number)
            if x >= 0.1:
                break
        number = min(number, maxloop)

        # actual runs
        gc.collect()
        if do_memory:
            memory = memory_usage()
        if number == 1:
            if x > 1:
                repeat_ = int(repeat // x)
            else:
                repeat_ = repeat - 1
            r = [x] + t.repeat(repeat_, 1)
        else:
            r = t.repeat(repeat, number)
        best = min(r)
        if do_memory:
            memory = memory_usage(since=memory)

        info = _get_info(iresult, len(stmts), arg, keyword, info_nspaces)
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

        if verbose:
            msg = '{0}{1}{2} loops, best of {3}: {4:6.2f} {5} per loop'.format(
                info, ': ' if info else '', number, len(r), value, unit)
        else:
            msg = '{0} {1:6.2f} {2}'.format(info, value, unit)
        if do_memory:
            msg += '. ' + ', '.join(k + ':' + str(v) + 'MiB'
                                    for k, v in memory.items())
        print(msg)

        result['info'][iresult] = info
        result['time'][iresult] = best / number
        if do_memory:
            for k in memory:
                result[k][iresult] = memory[k]

    result['info'] = result['info'].reshape(shape).T
    result['time'] = result['time'].reshape(shape).T
    if do_memory:
        for k in memory:
            result[k] = result[k].reshape(shape).T

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


def _get_info(istmt, nstmts, args, keywords, info_nspaces):
    if nstmts > 1:
        length_stmt = str(len(str(nstmts)))
        info = ('{0:' + length_stmt + '}: ').format(istmt % nstmts + 1)
    else:
        info = ''
    table = [repr(a) for a in args] + \
            ['{0}={1}'.format(k, _get_str(v)) for k, v in keywords.items()]
    info += ' '.join(('{0:' + str(n) + '}').format(i)
                     for i, n in zip(table, info_nspaces))
    return info


def _get_str(v):
    try:
        return v.__name__
    except AttributeError:
        pass
    if type(v).__name__ == 'fortran':
        try:
            return refortran.match(v.__doc__).group(2)
        except AttributeError:
            return 'fortran'
    return repr(v)


def _get_info_nspaces(args, keywords):
    return len(args) * [0] + [max(len('{0}={1}'.format(k, _get_str(_)))
                                  for _ in v) for k, v in keywords.items()]


def _iterkeywords(keywords):
    keys = keywords.keys()[::-1]
    itervalues = itertools.product(*keywords.values()[::-1])
    for values in itervalues:
        yield OrderedDict((k, v) for k, v in zip(keys, values)[::-1])
