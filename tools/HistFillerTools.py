import re
import json
from tools.MetaData import ConstructDatasetName
import glob

mdFile = "/lustre/fs22/group/atlas/freder/hh/hh-analysis/tools/metaData.json"


mcCampaign = {
    "r13167": ["2015", "2016"],  # "mc20a", run2, 2015-16
    "r13144": ["2017"],  # "mc20d", run2, 2017
    "r13145": ["2018"],  # "mc20e", run2, 2018
    "r13829": ["2022"],  # "mc21a", run3, 2022
    # # ptag
    # mc20 = "p5057"
}


def GetMetaDataFromFile(file):

    metaData = {}
    filepath = file._file._file_path

    # get r-tag for datayears
    ami = re.findall("e[0-9]{4}.s[0-9]{4}.r[0-9]{5}", filepath)
    r_tag = ami[0][-6:]
    metaData["dataYears"] = mcCampaign[r_tag]

    # get logical dataset name from ntuple name
    datasetName = ConstructDatasetName(filepath)
    print("Original Dataset Name: " + datasetName)

    md = json.load(open(mdFile))
    if datasetName not in md:
        print("metaData not in json yet, will query from ami")
        import tools.MetaData

        tools.MetaData.get(filepath)

    ds_info = md[datasetName]
    metaData["genFiltEff"] = float(ds_info["genFiltEff"])
    metaData["crossSection"] = float(ds_info["crossSection"])
    metaData["kFactor"] = float(ds_info["kFactor"])
    metaData["events"] = float(ds_info["initial_events"])
    metaData["sum_of_weights"] = float(ds_info["initial_sum_of_weights"])

    return metaData


def ConstructFilelist(sampleName, toMerge=False):
    """

    Parameters
    ----------
    sampleName : str
        options : mc21_cHHH01d0, mc20_l1cvv1cv1, mc20_ttbar, mc20_dijet, run2
    toMerge : bool, optional
        to construct filelist for processed files to merge, by default False

    Returns
    -------
    filelist : list
        list of strings with full samplepaths
    """

    if sampleName == "mc21_cHHH01d0":
        topPath = "/lustre/fs22/group/atlas/freder/hh/samples/"
        pattern = "user.frenner.HH4b.2022_11_25_.601479.PhPy8EG_HH4b_cHHH01d0.e8472_s3873_r13829_p5440_TREE/*"
    if sampleName == "mc20_l1cvv1cv1":
        # 1cvv1cv1
        topPath = "/lustre/fs22/group/atlas/freder/hh/samples/"
        pattern = "user.frenner.HH4b.2022_12_14.502970.MGPy8EG_hh_bbbb_vbf_novhh_l1cvv1cv1.e8263_s3681_r*/*"

    # mc20 bkg
    if sampleName == "mc20_ttbar":
        topPath = "/lustre/fs22/group/atlas/dbattulga/ntup_SH_Oct20/bkg/"
        pattern = "*ttbar*/*"

    if sampleName == "mc20_dijet":
        topPath = "/lustre/fs22/group/atlas/dbattulga/ntup_SH_Oct20/bkg/"
        pattern = "*jetjet*/*"

    if sampleName == "run2":
        topPath = "/lustre/fs22/group/atlas/freder/hh/samples/"
        pattern = "user.frenner.HH4b.2023_01_05.data*/*"

    if toMerge:
        # just changes topPath
        topPath = "/lustre/fs22/group/atlas/freder/hh/run/histograms/"

    if "topPath" not in locals():
        raise NameError(f"{sampleName} is not defined")
    filelist = []
    for file in glob.iglob(topPath + "/" + pattern):
        filelist += [file]

    return filelist


def EventRanges(tree, batch_size=10_000, nEvents="All"):
    # construct ranges, batch_size=1000 gives e.g.
    # [[0, 999], [1000, 1999], [2000, 2999],...]
    ranges = []
    batch_ranges = []
    if nEvents is "All":
        nEvents = tree.num_entries
    for i in range(0, nEvents, batch_size):
        ranges += [i]
    if nEvents not in ranges:
        ranges += [nEvents + 1]
    for i, j in zip(ranges[:-1], ranges[1:]):
        batch_ranges += [[i, j - 1]]
    return batch_ranges


def GetGenerators(tree, vars, nEvents=-1):

    batch_ranges = EventRanges
    # load a certain range
    for batch in batch_ranges:
        if not vars:
            vars = tree.keys()

        yield tree.arrays(vars, entry_start=batch[0], entry_stop=batch[1], library="np")
        # del arr
