
pyhh is a python only analysis framework processing ntuples from the easyJet framework as inputs to do a boosted VBF HH->4b analysis. It is capable of doing event selection, plotting and fitting.

```
./pyhh/pyhh/main.py -h       
usage: pyhh [-h] {select,make-submit,merge,plot,fit} ...

positional arguments:
  {select,make-submit,merge,plot,fit}
    select              run object selection
    make-submit         make HTCondor submit file
    merge               merge files of same logical dataset
    plot                run plotting
    fit                 run fitting
```