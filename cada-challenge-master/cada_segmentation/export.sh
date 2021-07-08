#!/usr/bin/env bash

./build.sh

docker save cada_segmentation | gzip -c > cada_segmentation.tar.gz
