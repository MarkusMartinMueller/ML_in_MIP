#!/usr/bin/env bash

./build.sh

docker save cada_risk | gzip -c > cada_risk.tar.gz
