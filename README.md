
pyhh is a python only analysis framework processing ntuples from the easyJet framework as inputs to do a boosted VBF HH->4b analysis. It is capable of doing event selection, plotting and fitting. To install it do from the top pyhh folder.
```
python3 -m pip install .
```
Then it should be possible to do

```
pyhh --help
usage: pyhh [-h] {select,make-submit,merge,plot,fit} ...

positional arguments:
  {select,make-submit,merge,plot,fit}
    select              run object selection
    make-submit         make HTCondor submit file
    merge               merge files of same logical dataset
    plot                run plotting
    fit                 run fitting
```