import numpy as np
from scipy import stats


class FloatHistogram:
    def __init__(self, name, binrange, bins=100, compress=True):
        # bins translates as number of bin edges though
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
        self._histRaw = np.zeros(self._bins.size - 1, dtype=float)
        self._hist = np.zeros(self._bins.size - 1, dtype=float)
        self._w2sum = np.zeros(self._bins.size - 1, dtype=float)

        # compression for h5 file
        self._compression = dict(compression="gzip") if compress else {}

    def fill(self, values, weights):
        hist = np.histogramdd(values, bins=[self._bins], weights=weights)[0]
        histRaw = np.histogramdd(values, bins=[self._bins])[0]
        w2 = np.histogramdd(values, bins=[self._bins], weights=weights**2)[0]

        self._hist += hist
        self._histRaw += histRaw
        self._w2sum += w2

    def write(self, file_):
        hgroup = file_.create_group(self._name)
        hgroup.attrs["type"] = "float"
        hist = hgroup.create_dataset("histogram", data=self._hist, **self._compression)
        hgroup.create_dataset("histogramRaw", data=self._histRaw, **self._compression)
        hgroup.create_dataset("w2sum", data=self._w2sum, **self._compression)
        ax = hgroup.create_dataset("edges", data=self._bins[1:-1], **self._compression)
        ax.make_scale("edges")
        hist.dims[0].attach_scale(ax)


class FloatHistogram2D:
    def __init__(self, name, binrange1, binrange2, bins=100, compress=True):
        # bins translates as number of bin edges though
        self._name = name
        # add overflow and underflow bins
        infvar = np.array([np.inf])
        self._bins1 = np.concatenate(
            [
                -infvar,
                np.linspace(*binrange1, bins),
                infvar,
            ]
        )
        self._bins2 = np.concatenate(
            [
                -infvar,
                np.linspace(*binrange2, bins),
                infvar,
            ]
        )
        self._bins = np.array([self._bins1, self._bins2])
        # book hist
        self._hist = np.zeros((self._bins1.size - 1, self._bins2.size - 1), dtype=float)
        self._histRaw = np.zeros(
            (self._bins1.size - 1, self._bins2.size - 1), dtype=float
        )
        self._w2sum = np.zeros(
            (self._bins1.size - 1, self._bins2.size - 1), dtype=float
        )
        # compression for h5 file
        self._compression = dict(compression="gzip") if compress else {}

    def fill(self, values, weights):
        if values.shape[0] != 0:
            hist = np.histogramdd(values, weights=weights, bins=self._bins)[0]
            histRaw = np.histogramdd(values, bins=self._bins)[0]
            ret = stats.binned_statistic_2d(
                values[:, 0],
                values[:, 1],
                None,
                "count",
                bins=self._bins,
                expand_binnumbers=True,
            )
            binIndices = ret.binnumber - 1
            for k, indices in enumerate(binIndices.T):
                self._w2sum[indices[0], indices[1]] += weights[k] ** 2
            self._hist += hist
            self._histRaw += histRaw

    def write(self, file_):
        hgroup = file_.create_group(self._name)
        hgroup.attrs["type"] = "float"
        hist = hgroup.create_dataset("histogram", data=self._hist, **self._compression)
        histRaw = hgroup.create_dataset(
            "histogramRaw", data=self._histRaw, **self._compression
        )
        w2sum = hgroup.create_dataset("w2sum", data=self._w2sum, **self._compression)
        ax = hgroup.create_dataset("edges", data=self._bins, **self._compression)
        ax.make_scale("edges")
        hist.dims[0].attach_scale(ax)


class IntHistogram:
    def __init__(self, name, binrange):
        low, high = np.array(binrange, dtype=np.int64)
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

    def write(self, file_):
        c = dict(compression="gzip")
        file = file_.create_group(self._name)
        file.attrs["type"] = "int"
        hist = file.create_dataset("histogram", data=self._hist, **c)
        edges = np.arange(self._nbins + 1) - self._offset
        ax = file.create_dataset("edges", data=edges, **c)
        ax.make_scale("edges")
        hist.dims[0].attach_scale(ax)
