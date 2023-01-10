import re
import pyAMI.client
import pyAMI_atlas.api as AtlasAPI
import os
import json

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

mdFile = "/lustre/fs22/group/atlas/freder/hh/hh-analysis/metaData.json"


def get(filepath):
    """
    queries the ami info and writes it to a json file
    Parameters
    ----------
    file : str
       filepath
    """

    datasetName = ConstructDatasetName(filepath)
    print(f"Get Meta-data for file: {filepath}")
    print("Original Dataset Name: " + datasetName)
    # query info
    # make sure file is not empty
    if os.stat(mdFile).st_size == 0:
        data = {}
    else:
        data = json.load(open(mdFile))

    if datasetName not in data:
        ds_info = AtlasAPI.get_dataset_info(client, dataset=datasetName)
        data[datasetName] = ds_info[0]
    json.dump(data, open(mdFile, "w"))


def ConstructDatasetName(filepath):
    """
    Constructs original logical datasetname from ntuples folder name

    Parameters
    ----------
    filepath : str

    Returns
    -------
    datasetName : str
    """
    # get dataset number
    ds = re.findall("(?<=\.)[0-9]{6}(?=\.)", filepath)
    # get ami tags until r-tag as p always changes
    ami = re.findall("e[0-9]{4}.s[0-9]{4}.r[0-9]{5}", filepath)
    # print("dataset number: ", ds[0])
    # print("partial AMI-tag: ", ami[0])
    r_tag = ami[0][-6:]
    # if r_tag in
    print("dataYears: ", mcCampaign[r_tag])

    # construct logical dataset name from ntuple name
    ds_parts = filepath.split(".")[4:7]
    if int(r_tag[1:]) < 13829:
        project = ["mc20_13TeV"]
    else:
        project = ["mc21_13p6TeV"]
    ds = project + ds_parts
    ds.insert(-1, "deriv.DAOD_PHYS")
    datasetName = ".".join([str(x) for x in ds])[:-10]

    return datasetName
