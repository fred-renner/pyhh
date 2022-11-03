import numpy as np


class FloatHistogram:
    def __init__(self, name, binrange, bins=100, compress=True):
        self._name = name
        self._bins = np.linspace(*binrange, bins)
        self._hist = np.zeros(self._bins.size - 1, dtype=float)
        self._compression = dict(compression="gzip") if compress else {}

    # update histogram bin heights per iteration
    def fill(self, arr):
        hist = np.histogramdd(arr, bins=[self._bins])[0]
        self._hist += hist

    # final write out to h5
    def write(self, file_, name=None):
        # create folder in h5 for plot
        file = file_.create_group(name or self._name)
        file.attrs["type"] = "float"
        hist = file.create_dataset("histogram", data=self._hist, **self._compression)
        ax = file.create_dataset("edges", data=self._bins, **self._compression)
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
        vals = arr[self._name] + self._offset
        self._hist[0] += (vals < 0).sum()
        self._hist[-1] += (vals >= self._nbins).sum()
        valid = (vals < self._nbins) & (vals >= 0)
        vals = vals[valid]
        if vals.dtype.kind == "f":
            vals = vals.astype(np.int64)
        self._hist[1:-1] += np.bincount(vals, minlength=self._nbins)

    def write(self, file_, name=None):
        c = dict(compression="gzip")
        file = file_.create_group(name or self._name)
        file.attrs["type"] = "int"
        hist = file.create_dataset("histogram", data=self._hist, **c)
        infvar = np.array([np.inf])
        values = np.arange(self._nbins) - self._offset
        ax = file.create_dataset("values", data=values, **c)
        ax.make_scale("values")
        hist.dims[0].attach_scale(ax)
