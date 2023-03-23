#!/usr/bin/env python3
import multiprocessing

import histfiller.analysis
import histfiller.configuration
import histfiller.histdefs as histdefs
import histfiller.tools
import uproot
from h5py import File
from tools.logging import log
from tqdm.auto import tqdm


def run(args):
    def filling_callback(results):
        """
        The filling is executed each time an analysis.run job finishes. This is
        executed sequentially so no data races

        Parameters
        ----------
        results : list
            takes list from analysis.ObjectSelection.returnResults()
        """
        # update bin heights per iteration
        for hist in hists:
            if hist._name not in results.keys():
                log.warning(
                    f"histogram with name: {hist._name} defined but not in results"
                )
            res = results[hist._name]
            hist.fill(values=res[0], weights=res[1])
        pbar.update(config.batchSize)
        return

    def error_handler(e):
        log.error(e.__cause__)
        pool.terminate()
        # prevents more jobs submissions
        pool.close()
        return

    # get configuration
    config = histfiller.configuration.Setup(args)
    # init hists
    hists = histdefs.hists

    with File(config.histOutFile, "w") as outfile:
        # loop over input files
        for i, file_ in enumerate(config.filelist):
            log.info("Processing file " + str(i + 1) + "/" + str(len(config.filelist)))
            with uproot.open(file_) as file:
                # access the tree
                tree = file["AnalysisMiniTree"]
                # take only vars that exist
                existingVars = set(tree.keys()).intersection(config.vars)

                if config.isData:
                    metaData = {}
                    substrings = file_.split(".")
                    dataCampaign = [s for s in substrings if "data" in s]
                    metaData["dataYear"] = "20" + dataCampaign[-1].split("_")[0][-2:]
                    metaData["isData"] = True
                else:
                    metaData = histfiller.tools.GetMetaDataFromFile(file)
                    metaData["isData"] = False
                metaData["blind"] = config.BLIND
                eventBatches = histfiller.tools.EventRanges(
                    tree, batch_size=config.batchSize, nEvents=config.nEvents
                )

                # progressbar
                pbar = tqdm(total=tree.num_entries, position=0, leave=True)
                if args.debug:
                    for batch in eventBatches:
                        results = histfiller.analysis.run(
                            batch, metaData, tree, existingVars
                        )
                        filling_callback(results)
                else:
                    # a pool object can start child processes on different cpu cores,
                    # this properly releases memory per batch
                    pool = multiprocessing.Pool(config.cpus)
                    for batch in eventBatches:
                        pool.apply_async(
                            histfiller.analysis.run,
                            (batch, metaData, tree, existingVars),
                            callback=filling_callback,
                            error_callback=error_handler,
                        )
                    pool.close()
                    pool.join()
                    pbar.close()
                log.info("Done\n")

        # write histograms to file
        log.info("Writing to " + config.histOutFile)
        for hist in hists:
            hist.write(outfile)
