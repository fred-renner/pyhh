#!/usr/bin/env python3


def GetGenerators(tree, vars, nEvents=-1):

    # construct ranges, batch_size=1000 gives e.g.
    # [[0, 999], [1000, 1999], [2000, 2999],...]
    batch_size = 10_000
    ranges = []
    batch_ranges = []
    if nEvents==-1:
        nEvents = tree.num_entries
    for i in range(0, nEvents, batch_size):
        ranges += [i]
    if nEvents not in ranges:
        ranges += [nEvents + 1]
    for i, j in zip(ranges[:-1], ranges[1:]):
        batch_ranges += [[i, j - 1]]

    # load a certain range
    for batch in batch_ranges:
        if not vars:
            vars = tree.keys()
        arr = tree.arrays(vars, entry_start=batch[0], entry_stop=batch[1], library="np")
        yield arr
        del arr
