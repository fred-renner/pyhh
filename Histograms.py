import numpy as np


class FloatHistogram:
    def __init__(self, name, binrange, bins=100, compress=True):
        self._name = name
        # add overflow and underflow bins
        infvar = np.array([np.inf])
        self._bins = np.concatenate(
            [
                -infvar,
                np.linspace(*binrange, bins),
                infvar,
            ]
        )
        # book hist
        self._hist = np.zeros(self._bins.size - 1, dtype=float)
        # compression for h5 file
        self._compression = dict(compression="gzip") if compress else {}

    def fill(self, arr):
        hist = np.histogramdd(arr, bins=[self._bins])[0]
        self._hist += hist

    def write(self, group, name=None):
        hgroup = group.create_group(name or self._name)
        hgroup.attrs["type"] = "float"
        hist = hgroup.create_dataset("histogram", data=self._hist, **self._compression)
        ax = hgroup.create_dataset("edges", data=self._bins[1:-1], **self._compression)
        ax.make_scale("edges")
        hist.dims[0].attach_scale(ax)


class IntHistogram:
    def __init__(self, name, binrange):
        low, high = np.array(binrange, dtype=int)
        if not np.isclose([low, high], binrange).all():
            raise ValueError(f"interval {binrange} needs to be integer")
        self._name = name
        self._offset = -low
        # one bin for each value
        self._nbins = high - low + 1
        # plus two for under / overflow
        self._hist = np.zeros(self._nbins + 2, dtype=np.int64)

    def fill(self, arr):
        if arr.dtype.kind == "f":
            arr = arr.astype(np.int64)
        self._hist[1:-1] += np.bincount(arr, minlength=self._nbins)

    def write(self, file_, name=None):
        c = dict(compression="gzip")
        file = file_.create_group(name or self._name)
        file.attrs["type"] = "int"
        hist = file.create_dataset("histogram", data=self._hist, **c)
        edges = np.arange(self._nbins + 1) - self._offset
        ax = file.create_dataset("edges", data=edges, **c)
        ax.make_scale("edges")
        hist.dims[0].attach_scale(ax)
