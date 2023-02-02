#!/usr/bin/env python3
from tqdm.auto import tqdm
import uproot
import HistDefs
import tools.HistFillerTools
import tools.MetaData
from h5py import File
import Analysis
import multiprocessing
import argparse
import os


# TODO
# make yaml config

parser = argparse.ArgumentParser()
parser.add_argument("--cpus", type=int, default=None)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--file", type=str, default=None)

args = parser.parse_args()

if args.file:
    filelist = [args.file]
    fileParts = filelist[0].split("/")
    dataset = fileParts[-2]
    datasetPath = "/lustre/fs22/group/atlas/freder/hh/run/histograms/" + dataset
    file = fileParts[-1]
    if not os.path.isdir(datasetPath):
        os.makedirs(datasetPath)

    histOutFile = (
        "/lustre/fs22/group/atlas/freder/hh/run/histograms/"
        + dataset
        + "/"
        + file
        + ".h5"
    )
else:
    # default to mc 20 signal
    filelist = tools.HistFillerTools.ConstructFilelist("mc20_l1cvv1cv1")
    # filelist = tools.HistFillerTools.ConstructFilelist("mc20_ttbar")
    # filelist = tools.HistFillerTools.ConstructFilelist("run2")
    # make hist out file name from filename
    if "histOutFileName" not in locals():
        dataset = filelist[0].split("/")
        histOutFileName = "hists-" + dataset[-2] + ".h5"

    histOutFile = "/lustre/fs22/group/atlas/freder/hh/run/histograms/" + histOutFileName


# figure out which vars to load from analysis script
start = 'vars_arr["'
end = '"]'
vars = []
for line in open("/lustre/fs22/group/atlas/freder/hh/hh-analysis/Analysis.py", "r"):
    if "vars_arr[" in line:
        if "#" not in line:
            vars.append((line.split(start))[1].split(end)[0])


def filling_callback(results):
    """
    The filling is executed each time an Analysis.Run job finishes. According
    to this it is executed sequentially, so no data races.
    https://stackoverflow.com/questions/24770934/who-runs-the-callback-when-using-apply-async-method-of-a-multiprocessing-pool

        Parameters
        ----------
        results : list
           takes list from Analysis.ObjectSelection.returnResults()
    """
    # update bin heights per iteration
    for hist in hists:
        if hist._name not in results.keys():
            print(f"histogram with name: {hist._name} defined but not in results")
        res = results[hist._name]
        hist.fill(values=res[0], weights=res[1])
    pbar.update(batchSize)


def error_handler(e):
    print("\n\n---error_start---{}\n---error_end---\n".format(e.__cause__))
    pool.terminate()


# general settings
if args.debug:
    filelist = filelist[:3]
    histOutFile = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-debug.h5"
    nEvents = 1000
    cpus = 1
    batchSize = 1000
else:
    nEvents = "All"
    batchSize = 20_000
    if args.cpus:
        cpus = args.cpus
    else:
        cpus = multiprocessing.cpu_count() - 4

# auto setup blind if data
if any("data" in file for file in filelist):
    isData = True
else:
    isData = False
BLIND = isData

# init hists
hists = HistDefs.hists

with File(histOutFile, "w") as outfile:
    # loop over input files
    for i, file_ in enumerate(filelist):
        print("\nProcessing file " + str(i + 1) + "/" + str(len(filelist)))
        with uproot.open(file_) as file:
            # access the tree
            tree = file["AnalysisMiniTree"]
            # take only vars that exist
            existingVars = set(tree.keys()).intersection(vars)

            if isData:
                metaData = {}
                substrings = file_.split(".")
                dataCampaign = [s for s in substrings if "data" in s]
                metaData["dataYear"] = "20" + dataCampaign[0].split("_")[0][-2:]
                metaData["isData"] = True
            else:
                metaData = tools.HistFillerTools.GetMetaDataFromFile(file)
                metaData["isData"] = False
            metaData["blind"] = BLIND
            eventBatches = tools.HistFillerTools.EventRanges(
                tree, batch_size=batchSize, nEvents=nEvents
            )

            # progressbar
            pbar = tqdm(total=tree.num_entries, position=0, leave=True)
            if args.debug:
                for batch in eventBatches:
                    results = Analysis.Run(batch, metaData, tree, existingVars)
                    filling_callback(results)
            else:
                # a pool objects can start child processes on different cpu cores,
                # nicely this releases memory per batch
                pool = multiprocessing.Pool(cpus)
                for batch in eventBatches:
                    pool.apply_async(
                        Analysis.Run,
                        (batch, metaData, tree, existingVars),
                        callback=filling_callback,
                        error_callback=error_handler,
                    )
                pool.close()
                pool.join()
                pbar.close()
            print("Done")

    # write histograms to file
    print("Writing to " + histOutFile)
    for hist in hists:
        hist.write(outfile)
