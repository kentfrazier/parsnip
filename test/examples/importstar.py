from __future__ import print_function

from collections import *
from functools import *
from itertools import *

def get_file(path):
    return open(path)

this_file = partial(get_file, __file__)

def line_lengths(f):
    return imap(len, f)

def collate_by_length(f):
    vals = defaultdict(list)
    for lineno, length in izip(count(1), line_lengths(f)):
        vals[length].append(lineno)
    return vals

if __name__ == '__main__':
    for length, lines in sorted(collate_by_length(this_file()).iteritems(),
                                reverse=True):
        print('{0!s:5>}: {1}'.format(length, lines))

