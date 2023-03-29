#!/bin/bash

# export X509_USER_PROXY=$1
# voms-proxy-info -all -file $1

python3 /lustre/fs22/group/atlas/freder/hh/submit/pyhh/pyhh/main.py select --fill --dump --batchMode --file $1 
