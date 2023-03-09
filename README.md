pyhh is a python only analysis framework processing ntuples from the easyJet framework as inputs to do a boosted VBF HH->4b analysis. 

| script          | description                                                                                                                                                                                                                           |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `HistFiller.py` | Driver script to fill histograms configured with `Configuration.py`. Runs object selection code from `Analysis.py` per file. Can run over several files or just one with the `--file` option. Outputs an .h5 file with the histograms |
| `HistDefs.py`   | Definitions for the histograms to fill                                                                                                                                                                                                |
| `Merger.py`     | Merge hists from e.g. one dataset by adding up hists from given files                                                                                                                                                                 |
| `Plotter.py`    | has all the plots and is still under very heavy development                                                                                                                                                                           |
| `Fitting.py`    | yet to come                                                                                                                                                                                                                           |



```mermaid
flowchart TB

    subgraph Analysis.py
    load_vars --> ObjectSelection
    ObjectSelection --> returnResults
    end

    multiprocessing_pool --> child_process--> Analysis.py
    returnResults --> filling_callback
    filling_callback --> fillHists -.-> new_child -.-> multiprocessing_pool
    filling_callback -.- in_principle -.-> nano_ntuples

    subgraph Histfiller.py
    Configuration --> defineHists
    defineHists --> multiprocessing_pool
    multiprocessing_pool --> writeHists
    end
```

