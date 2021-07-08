#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

docker volume create cada_risk-output

docker run --rm \
        --memory=4g \
        -v $SCRIPTPATH/test/:/input/ \
        -v cada_risk-output:/output/ \
        cada_risk

docker run --rm \
        -v cada_risk-output:/output/ \
        python:3.7-slim cat /output/metrics.json | python -m json.tool

docker volume rm cada_risk-output
