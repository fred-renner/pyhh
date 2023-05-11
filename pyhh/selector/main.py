#!/usr/bin/env python3
import multiprocessing

import h5py
import selector.analysis
import selector.configuration
import selector.histdefs
import selector.tools
import uproot
from tools.logging import log
from tqdm.auto import tqdm


def run(args):
    """
    Main program for HH-->4b VBF boosted selection. It runs
    selector.analysis.run() jobs per event batch and executes the callback
    function each time it finishes. The callback does histogram filling or
    variable dumping.

    Parameters
    ----------
    args : Namespace
        args from the pyhh.main entry program
    """

    def callback(results):
        """
        The fillin/dumping is executed each time a selector.analysis.run job finishes.
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
            with h5py.File(config.dump_file, "r+") as f:
                selector.tools.write_vars(results, f)

        pbar.update(config.batchSize)
        return

    def error_callback(e):
        """
        handles error if selector.analysis.job fails

        Parameters
        ----------
        e : BaseException
            error
        """
        log.error(e.__cause__)
        pool.terminate()
        # prevents more jobs submissions
        pool.close()
        return

    ########## here the actual program starts ##########

    # get configuration
    config = selector.configuration.setup(args)
    # init hists
    hists, _, _ = selector.histdefs.get(do_systs=config.do_systematics)
    # loop over input files
    log.info("Processing file " + config.file)
    with uproot.open(config.file) as file:
        # access the tree
        tree = file["AnalysisMiniTree"]
        # more config
        selector.tools.vars_to_load(tree, config)
        if config.dump:
            selector.tools.init_dump_file(config)
        if config.isData:
            metaData = {}
            substrings = config.file.split(".")
            dataCampaign = [s for s in substrings if "data" in s]
            metaData["dataYear"] = "20" + dataCampaign[-1].split("_")[0][-2:]
        else:
            metaData = selector.tools.GetMetaDataFromFile(file)
        eventBatches = selector.tools.EventRanges(
            tree, batch_size=config.batchSize, nEvents=config.nEvents
        )
        # progressbar
        pbar = tqdm(total=tree.num_entries, position=0, leave=True)

        # RUN
        if args.debug:
            for batch in eventBatches:
                results = selector.analysis.run(batch, config, metaData, tree)
                callback(results)
        else:
            # a pool object can start child processes on different cpu cores,
            # this properly releases memory per batch
            pool = multiprocessing.Pool(config.cpus)
            for batch in eventBatches:
                pool.apply_async(
                    selector.analysis.run,
                    (batch, config, metaData, tree),
                    callback=callback,
                    error_callback=error_callback,
                )
            pool.close()
            pool.join()
            pbar.close()

    # write histograms to file
    if config.fill:
        with h5py.File(config.hist_out_file, "w") as outfile:
            log.info("Writing Hists to: " + config.hist_out_file)
            for hist in hists:
                hist.write(outfile)
    if config.dump:
        log.info("Dumped selected vars to: " + config.dump_file)

    log.info("Done")

    return
