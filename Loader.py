#!/usr/bin/env python3


def GetGenerators(tree, vars):

    # construct ranges, batch_size=1000 gives e.g.
    # [[0, 999], [1000, 1999], [2000, 2999],...]
    batch_size = 1_000
    ranges = []
    batch_ranges = []
    for i in range(0, tree.num_entries, batch_size):
        ranges += [i]
    if tree.num_entries not in ranges:
        ranges += [tree.num_entries + 1]
    for i, j in zip(ranges[:-1], ranges[1:]):
        batch_ranges += [[i, j - 1]]

    for batch in batch_ranges:
        arr = tree.arrays(
            vars, entry_start=batch[0], entry_stop=batch[1], library="np"
        )
        yield arr
