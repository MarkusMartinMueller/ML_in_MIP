#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

docker volume create cada_segmentation-output

docker run --rm \
        --memory=4g \
        -v $SCRIPTPATH/test/:/input/ \
        -v cada_segmentation-output:/output/ \
        cada_segmentation

docker run --rm \
        -v cada_segmentation-output:/output/ \
        python:3.7-slim cat /output/metrics.json | python -m json.tool

docker volume rm cada_segmentation-output
