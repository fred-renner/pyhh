import csv
import glob
import json
import os
import re

import pyAMI.client
import pyAMI_atlas.api as AtlasAPI
import uproot

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

mdFile = "/lustre/fs22/group/atlas/freder/hh/pyhh/tools/metaData.json"


def CombineCutBookkeepers(filelist):
    cutBookkeeper = {}
    cutBookkeeper["initial_events"] = 0
    cutBookkeeper["initial_sum_of_weights"] = 0
    cutBookkeeper["initial_sum_of_weights_squared"] = 0
    for file_ in filelist:
        with uproot.open(file_) as file:
            for key in file.keys():
                if "CutBookkeeper" and "NOSYS" in key:
                    cbk = file[key].to_numpy()
                    cutBookkeeper["initial_events"] += cbk[0][0]
                    cutBookkeeper["initial_sum_of_weights"] += cbk[0][1]
                    cutBookkeeper["initial_sum_of_weights_squared"] += cbk[0][2]
    return cutBookkeeper


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

    if not os.path.exists(mdFile):
        os.mknod(mdFile)
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
                pmgFile = "/lustre/fs22/group/atlas/freder/hh/pyhh/tools/PMGxsecDB_mc16.txt"
            if "mc21" in datasetName:
                pmgFile = "/lustre/fs22/group/atlas/freder/hh/pyhh/tools/PMGxsecDB_mc21.txt"
            with open(pmgFile) as fd:
                # dataset_number/I:physics_short/C:crossSection/D:genFiltEff/D:kFactor/D:relUncertUP/D:relUncertDOWN/D:generator_name/C:etag/C
                rd = csv.reader(fd, delimiter="\t")
                for row in rd:
                    if row[0] == ds_nr[0]:
                        # delete empty strings
                        row = list(filter(None, row))
                        data[datasetName]["kFactor"] = row[4]
        # get all files in dataset to sum up their sum_of_weights
        filelist = glob.glob(os.path.dirname(filepath) + "/*.root")
        cbk = CombineCutBookkeepers(filelist)
        data[datasetName]["initial_events"] = cbk["initial_events"]
        data[datasetName]["initial_sum_of_weights"] = cbk["initial_sum_of_weights"]

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
    # get ami tags until r-tag as p always changes
    ami = re.findall("e[0-9]{4}.s[0-9]{4}.r[0-9]{5}", folder)
    r_tag = ami[0][-6:]
    # construct logical dataset name from ntuple name
    ds_parts = folder.split(".")
    ds_parts = ds_parts[-3:]
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
