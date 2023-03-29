import glob
import json
import os
import re

from tools.logging import log
from selector.metadata import ConstructDatasetName
import pathlib
import selector.analysis

mdFile = pathlib.Path(__file__).parent / "metadata.json"

mcCampaign = {
    "r13167": ["2015", "2016"],  # "mc20a", run2, 2015-16
    "r13144": ["2017"],  # "mc20d", run2, 2017
    "r13145": ["2018"],  # "mc20e", run2, 2018
    "r13829": ["2022"],  # "mc21a", run3, 2022
}


def GetMetaDataFromFile(file):
    """
    Gets necessary metdadata from metdata file for selection, e.g. eventweight.
    Also does a ami query if metadata for file don't exist.

    Parameters
    ----------
    file : uproot.ReadOnlyFile
        root file

    Returns
    -------
    metadata : dict
        metadata
    """
    metaData = {}
    filepath = file._file._file_path

    # get r-tag for datayears
    ami = re.findall("e[0-9]{4}.s[0-9]{4}.r[0-9]{5}", filepath)
    r_tag = ami[0][-6:]
    metaData["dataYears"] = mcCampaign[r_tag]

    # get logical dataset name from ntuple name
    datasetName = ConstructDatasetName(filepath)
    log.info("Original Dataset Name: " + datasetName)

    if not os.path.exists(mdFile):
        os.mknod(mdFile)
        md = {}
    else:
        md = json.load(open(mdFile))

    if datasetName not in md:
        log.info("metaData not in json yet, will query from ami")
        import selector.metadata

        selector.metadata.get(filepath)
        md = json.load(open(mdFile))

    ds_info = md[datasetName]
    metaData["genFiltEff"] = float(ds_info["genFiltEff"])
    metaData["crossSection"] = float(ds_info["crossSection"])
    metaData["kFactor"] = float(ds_info["kFactor"])
    metaData["events"] = float(ds_info["initial_events"])
    metaData["sum_of_weights"] = float(ds_info["initial_sum_of_weights"])

    return metaData


def ConstructFilelist(
    sampleName, mergeProcessedHists=False, mergeProcessedDumps=False, verbose=False
):
    """
    Parameters
    ----------
    sampleName : str
        options : mc21_SM, mc20_SM, mc20_k2v0, mc20_ttbar, mc20_dijet, run2
    toMerge : bool, optional
        to construct filelist for processed files to merge, by default False
    mergeProcessedHists : bool, optional
        construct filelist for processed hists
    mergeProcessedDumps : bool, optional
        construct filelist for processed dumps
    Returns
    -------
    filelist : list
        list of strings with full samplepaths
    """

    if sampleName == "mc21_SM":
        topPath = "/lustre/fs24/group/atlas/freder/hh/samples/"
        pattern = "user.frenner.HH4b.2022_11_25_.601479.PhPy8EG_HH4b_cHHH01d0.e8472_s3873_r13829_p5440_TREE/*"
    if sampleName == "mc20_SM":
        topPath = "/lustre/fs24/group/atlas/freder/hh/samples/"
        pattern = "user.frenner.HH4b.2023_03_13_.502970.MGPy8EG_hh_bbbb_vbf_novhh_l1cvv1cv1.e8263_s3681_r*/*"
    if sampleName == "mc20_k2v0":
        topPath = "/lustre/fs24/group/atlas/freder/hh/samples/"
        pattern = "user.frenner.HH4b.2023_03_13_.502971.MGPy8EG_hh_bbbb_vbf_novhh_l1cvv0cv1.e8263_s3681_r*/*"
    if sampleName == "mc20_ttbar":
        topPath = "/lustre/fs24/group/atlas/dbattulga/ntup_SH_Feb2023//MC/"
        pattern = "*ttbar*/*"
    if sampleName == "mc20_dijet":
        topPath = "/lustre/fs24/group/atlas/dbattulga/ntup_SH_Feb2023//MC/"
        pattern = "*jetjet*/*"
    if sampleName == "run2":
        topPath = "/lustre/fs24/group/atlas/dbattulga/ntup_SH_Feb2023/Data/"
        pattern = "*data1*/*"

    # just changes topPath
    if mergeProcessedHists:
        topPath = "/lustre/fs22/group/atlas/freder/hh/run/histograms/"
    if mergeProcessedDumps:
        topPath = "/lustre/fs22/group/atlas/freder/hh/run/dump/"

    if "topPath" not in locals():
        verbose.error(f"{sampleName} is not defined")
        raise NameError(f"{sampleName} is not defined")

    filelist = []

    # sort by descending file size
    for file in sorted(
        glob.iglob(topPath + "/" + pattern), key=os.path.getsize, reverse=True
    ):
        filelist += [file]
    if verbose:
        for f in filelist:
            print(f)

    return filelist


def EventRanges(tree, batch_size=10_000, nEvents="All"):
    """
    construct ranges, batch_size=1000 gives e.g.
    [[0, 1000], [1000, 2000], [2000, 3000],...]

    Parameters
    ----------
    tree : uproot TTree
        tree with branches from uproot
    batch_size : int, optional
        how many events per batch, by default 10_000
    nEvents : str, optional
        total amount of events, by default "All"

    Returns
    -------
    list
        event ranges
    """
    ranges = []
    batch_ranges = []
    if nEvents == "All":
        nEvents = tree.num_entries
    for i in range(0, nEvents + 1, batch_size):
        ranges += [i]
    # add very last index to not include
    if nEvents + 1 not in ranges:
        ranges += [nEvents + 1]
    for i, j in zip(ranges[:-1], ranges[1:]):
        batch_ranges += [[i, j]]
    return batch_ranges


def write_vars(results, f):
    """
    writes variables from callback

    Parameters
    ----------
    results : dict
        see selector.callback()
    f : h5py.File
        file to write to
    """

    for varType in ["bools", "floats"]:
        if varType == "bools":
            vars = selector.analysis.boolVars
        if varType == "floats":
            vars = selector.analysis.floatVars
        for var in vars:
            var_ds = f[varType][var]
            var_result = results[varType][var]
            if var_ds.shape[0] == 0:
                idx_start = 0
            else:
                idx_start = var_ds.shape[0]
            idx_end = idx_start + var_result.shape[0]
            var_ds.resize((idx_end,))
            var_ds[idx_start:idx_end] = var_result
