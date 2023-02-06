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
import Configuration


parser = argparse.ArgumentParser()
parser.add_argument("--cpus", type=int, default=None)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--batchMode", action="store_true")
parser.add_argument("--file", type=str, default=None)

args = parser.parse_args()

# get configuration
config = Configuration.Config(args)


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
    pbar.update(config.batchSize)
    return


def error_handler(e):
    print("\n\n---error_start---{}\n---error_end---\n".format(e.__cause__))
    pool.terminate()
    return


# init hists
hists = HistDefs.hists

with File(config.histOutFile, "w") as outfile:
    # loop over input files
    for i, file_ in enumerate(config.filelist):
        print("\nProcessing file " + str(i + 1) + "/" + str(len(config.filelist)))
        with uproot.open(file_) as file:
            # access the tree
            tree = file["AnalysisMiniTree"]
            # take only vars that exist
            existingVars = set(tree.keys()).intersection(config.vars)

            if config.isData:
                metaData = {}
                substrings = file_.split(".")
                dataCampaign = [s for s in substrings if "data" in s]
                metaData["dataYear"] = "20" + dataCampaign[0].split("_")[0][-2:]
                metaData["isData"] = True
            else:
                metaData = tools.HistFillerTools.GetMetaDataFromFile(file)
                metaData["isData"] = False
            metaData["blind"] = config.BLIND
            eventBatches = tools.HistFillerTools.EventRanges(
                tree, batch_size=config.batchSize, nEvents=config.nEvents
            )

            # progressbar
            pbar = tqdm(total=tree.num_entries, position=0, leave=True)
            if args.debug:
                for batch in eventBatches:
                    results = Analysis.Run(batch, metaData, tree, existingVars)
                    filling_callback(results)
            else:
                # a pool object can start child processes on different cpu cores,
                # this properly releases memory per batch
                pool = multiprocessing.Pool(config.cpus)
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
    print("Writing to " + config.histOutFile)
    for hist in hists:
        hist.write(outfile)
