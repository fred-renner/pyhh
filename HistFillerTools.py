import re
import json
from tools.MetaData import ConstructDatasetName

mdFile = "/lustre/fs22/group/atlas/freder/hh/hh-analysis/metaData.json"


mcCampaign = {
    "r13167": ["2015", "2016"],  # "mc20a", run2, 2015-16
    "r13144": ["2017"],  # "mc20d", run2, 2017
    "r13145": ["2018"],  # "mc20e", run2, 2018
    "r13829": ["2022"],  # "mc21a", run3, 2022
    # # ptag
    # mc20 = "p5057"
}


def getMetaData(file):

    metaData = {}
    filepath = file._file._file_path
    metaData["isSignal"] = True if "_hh_bbbb_" in filepath else False

    # cut book keeping
    for key in file.keys():
        if "CutBookkeeper" and "NOSYS" in key:
            cbk = file[key].to_numpy()
            metaData["initial_events"] = cbk[0][0]
            metaData["initial_sum_of_weights"] = cbk[0][1]
            metaData["initial_sum_of_weights_squared"] = cbk[0][2]

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

    return metaData
