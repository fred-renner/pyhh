import re
import pyAMI.client
import pyAMI_atlas.api as AtlasAPI
from enum import Enum

client = pyAMI.client.Client("atlas")
AtlasAPI.init()


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
    print(f"Get Meta-data for file: {filepath}")
    metaData["isSignal"] = True if "_hh_bbbb_" in filepath else False

    # cut book keeping
    for key in file.keys():
        if "CutBookkeeper" and "NOSYS" in key:
            cbk = file[key].to_numpy()
            metaData["initial_events"] = cbk[0][0]
            metaData["initial_sum_of_weights"] = cbk[0][1]
            metaData["initial_sum_of_weights_squared"] = cbk[0][2]

    # get dataset number
    ds = re.findall("(?<=\.)[0-9]{6}(?=\.)", filepath)
    # get ami tags until r-tag as p always changes
    ami = re.findall("e[0-9]{4}.s[0-9]{4}.r[0-9]{5}", filepath)
    # print("dataset number: ", ds[0])
    # print("partial AMI-tag: ", ami[0])
    r_tag = ami[0][-6:]
    # if r_tag in
    print("dataYears: ", mcCampaign[r_tag])
    metaData["dataYears"] = mcCampaign[r_tag]

    # construct logical dataset name from ntuple name
    ds_parts = filepath.split(".")[4:7]
    if int(r_tag[1:]) < 13829:
        project=["mc20_13TeV"]
    else:
        project=["mc21_13p6TeV"]
    ds = project + ds_parts
    ds.insert(-1, "deriv.DAOD_PHYS")
    datasetName = ".".join([str(x) for x in ds])[:-10]
    print("Original Dataset Name: " + datasetName)
    # query info
    ds_info = AtlasAPI.get_dataset_info(client, dataset=datasetName)
    metaData["genFiltEff"] = float(ds_info[0]["genFiltEff"])
    metaData["crossSection"] = float(ds_info[0]["crossSection"])

    return metaData
