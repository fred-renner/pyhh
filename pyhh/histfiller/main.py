#!/usr/bin/env python3
import multiprocessing
import histfiller.analysis
import histfiller.configuration
import histfiller.histdefs as histdefs
import histfiller.tools
import uproot
import h5py
from tools.logging import log
from tqdm.auto import tqdm


def run(args):
    """ """

    def callback(results):
        """
        The filling dumping is executed each time an analysis.run job finishes.
        This is executed sequentially so no data races.

        Parameters
        ----------
        results : dict
            dict from analysis.ObjectSelection.returnResults()
        """

        # update bin heights per iteration
        if config.fill:
            for hist in hists:
                if hist._name not in results.keys():
                    log.warning(
                        f"histogram with name: {hist._name} defined but not in results"
                    )
                res = results[hist._name]
                hist.fill(values=res[0], weights=res[1])

        # dump variables by appending to dump file
        if config.dump:
            with h5py.File(config.dumpFile, "r+") as f:
                histfiller.tools.write_vars(results, f)

        pbar.update(config.batchSize)
        return

    def error_handler(e):
        log.error(e.__cause__)
        pool.terminate()
        # prevents more jobs submissions
        pool.close()
        return

    # get configuration
    config = histfiller.configuration.setup(args)
    # init hists
    hists = histdefs.hists

    # loop over input files
    log.info("Processing file " + config.file)
    with uproot.open(config.file) as file:
        # access the tree
        tree = file["AnalysisMiniTree"]
        # take only vars that exist
        existingVars = set(tree.keys()).intersection(config.vars)

        if config.isData:
            metaData = {}
            substrings = config.file.split(".")
            dataCampaign = [s for s in substrings if "data" in s]
            metaData["dataYear"] = "20" + dataCampaign[-1].split("_")[0][-2:]
        else:
            metaData = histfiller.tools.GetMetaDataFromFile(file)
        eventBatches = histfiller.tools.EventRanges(
            tree, batch_size=config.batchSize, nEvents=config.nEvents
        )

        # progressbar
        pbar = tqdm(total=tree.num_entries, position=0, leave=True)
        if args.debug:
            for batch in eventBatches:
                results = histfiller.analysis.run(
                    batch, config, metaData, tree, existingVars
                )
                callback(results)
        else:
            # a pool object can start child processes on different cpu cores,
            # this properly releases memory per batch
            pool = multiprocessing.Pool(config.cpus)
            for batch in eventBatches:
                pool.apply_async(
                    histfiller.analysis.run,
                    (batch, config, metaData, tree, existingVars),
                    callback=callback,
                    error_callback=error_handler,
                )
            pool.close()
            pool.join()
            pbar.close()
        log.info("Done")

    # write histograms to file
    if config.fill:
        with h5py.File(config.histOutFile, "w") as outfile:
            log.info("Writing Hists to: " + config.histOutFile)
            for hist in hists:
                hist.write(outfile)
    if config.dump:
        log.info("Dumped selected vars to: " + config.dumpFile)
