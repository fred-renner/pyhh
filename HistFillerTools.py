import re
import pyAMI.client
import pyAMI_atlas.api as AtlasAPI

client = pyAMI.client.Client("atlas")
AtlasAPI.init()


def getMetaData(file):
    metaData = {}
    for key in file.keys():
        if "CutBookkeeper" and "NOSYS" in key:
            cbk = file[key].to_numpy()
            metaData["initial_events"] = cbk[0][0]
            metaData["initial_sum_of_weights"] = cbk[0][1]
            metaData["initial_sum_of_weights_squared"] = cbk[0][2]
    print(file._file._file_path)
    filename = file._file._file_path
    # get dataset number
    ds = re.findall("(?<=\.)[0-9]{6}(?=\.)", filename)
    # get ami tags until r-tag as p always changes
    ami = re.findall("e[0-9]{4}.s[0-9]{4}.r[0-9]{5}", filename)
    print("dataset number: ", ds[0])
    print("partial AMI-tag: ", ami[0])
    # get actual datasetname from dataset number and ami tags
    datasets = AtlasAPI.list_datasets(
        client, patterns=f"%{ds[0]}%{ami[0]}%", type="DAOD_PHYS"
    )
    # use logical dataset name to get info from first dataset matching the
    # pattern
    datasetName = datasets[0]["ldn"]
    ds_info = AtlasAPI.get_dataset_info(client, dataset=datasetName)
    metaData["crossSection"] = ds_info[0]["crossSection"]
    return metaData
