#!/bin/bash

export X509_USER_PROXY=$1
# voms-proxy-info -all
voms-proxy-info -all -file $1

python3 /lustre/fs22/group/atlas/freder/hh/hh-analysis/HistFiller.py --cpus $2 --file $3 
