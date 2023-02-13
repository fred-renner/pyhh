#!/bin/bash

export X509_USER_PROXY=$1
# voms-proxy-info -all -file $1
RUCIO_ACCOUNT=frenner

_voms_proxy_long() {
    if ! type voms-proxy-info &>/dev/null; then
        echo "voms not set up!" 1>&2
        return 1
    fi
    local VOMS_ARGS="--voms atlas"
    if voms-proxy-info --exists --valid 24:00; then
        local TIME_LEFT=$(voms-proxy-info --timeleft)
        local HOURS=$(($TIME_LEFT / 3600))
        local MINUTES=$(($TIME_LEFT / 60 % 60 - 1))
        local NEW_TIME=$HOURS:$MINUTES
        VOMS_ARGS+=" --noregen --valid $NEW_TIME"
    else
        VOMS_ARGS+=" --valid 96:00"
    fi
    voms-proxy-init $VOMS_ARGS
}

source /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/user/atlasLocalSetup.sh
_voms_proxy_long
lsetup rucio

rucio download $2 --dir $3 --ndownloader 5 --replica-selection random --transfer-timeout 300
