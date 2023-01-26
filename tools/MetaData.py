import re
import pyAMI.client
import pyAMI_atlas.api as AtlasAPI
import os
import json
import csv

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

mdFile = "/lustre/fs22/group/atlas/freder/hh/hh-analysis/tools/metaData.json"


def get(filepath):
    """
    queries the ami info and writes it to a json file
    Parameters
    ----------
    file : str
       filepath
    """

    datasetName = ConstructDatasetName(filepath)

    # query info
    # make sure file is not empty
    if os.stat(mdFile).st_size == 0:
        data = {}
    else:
        data = json.load(open(mdFile))

    if datasetName not in data:
        print(f"query metadata for: {datasetName}")
        # need to do p wildcard search as too old ones get deleted
        datasetNames = datasetName[:-4] + "%"
        datasets = AtlasAPI.list_datasets(
            client, patterns=datasetNames, type="DAOD_PHYS"
        )
        ds_info = AtlasAPI.get_dataset_info(client, dataset=datasets[0]["ldn"])
        data[datasetName] = ds_info[0]

        # add kfactor either from ami or PMG file
        if "kFactor@PMG" in ds_info:
            ds_info["kFactor"] = float(ds_info["kFactor@PMG"])
        else:
            ds_nr = re.findall("(?<=\.)[0-9]{6}(?=\.)", filepath)
            if "mc20" in datasetName:
                pmgFile = "/lustre/fs22/group/atlas/freder/hh/hh-analysis/tools/PMGxsecDB_mc16.txt"
            if "mc21" in datasetName:
                pmgFile = "/lustre/fs22/group/atlas/freder/hh/hh-analysis/tools/PMGxsecDB_mc21.txt"
            with open(pmgFile) as fd:
                # dataset_number/I:physics_short/C:crossSection/D:genFiltEff/D:kFactor/D:relUncertUP/D:relUncertDOWN/D:generator_name/C:etag/C
                rd = csv.reader(fd, delimiter="\t")
                for row in rd:
                    if row[0] == ds_nr[0]:
                        # delete empty strings
                        row = list(filter(None, row))
                        data[datasetName]["kFactor"] = row[4]

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
    folder = filepath.split("/")[-2]
    # get dataset number
    ds_nr = re.findall("(?<=\.)[0-9]{6}(?=\.)", folder)
    # get ami tags until r-tag as p always changes
    ami = re.findall("e[0-9]{4}.s[0-9]{4}.r[0-9]{5}", folder)
    # print("dataset number: ", ds_nr[0])
    # print("partial AMI-tag: ", ami[0])
    r_tag = ami[0][-6:]
    # if r_tag in
    # print("dataYears: ", mcCampaign[r_tag])

    # construct logical dataset name from ntuple name
    ds_parts = folder.split(".")
    ds_parts = ds_parts[-3:]
    # print(ds_parts)

    # remove TREE
    ds_parts[-1] = ds_parts[-1][:-5]
    if int(r_tag[1:]) < 13829:
        project = ["mc20_13TeV"]
    else:
        project = ["mc21_13p6TeV"]
    ds = project + ds_parts
    ds.insert(-1, "deriv.DAOD_PHYS")
    datasetName = ".".join([str(x) for x in ds])

    return datasetName
