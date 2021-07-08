#!/usr/bin/env bash

./build.sh

docker save cada_detection | gzip -c > cada_detection.tar.gz
