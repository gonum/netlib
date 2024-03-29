#!/bin/bash

set -ex

CACHE_DIR="/home/runner/.cache/OpenBLAS"
mkdir -p ${CACHE_DIR}

# check if cache exists
if [ -e ${CACHE_DIR}/last_commit_id ]; then
    echo "Cache $CACHE_DIR hit"
    LAST_COMMIT="$(git ls-remote https://github.com/xianyi/OpenBLAS HEAD | grep -o '^\S*')"
    CACHED_COMMIT="$(cat ${CACHE_DIR}/last_commit_id)"
    # determine current OpenBLAS master commit id and compare
    # with commit id in cache directory
    if [ "$LAST_COMMIT" != "$CACHED_COMMIT" ]; then
        echo "Cache Directory $CACHE_DIR has stale commit"
        # if commit is different, delete the cache
        rm -rf ${CACHE_DIR}/*
    fi
fi

if [ ! -e ${CACHE_DIR}/last_commit_id ]; then
    # Clear cache.
    rm -rf ${CACHE_DIR}/*

    # cache generation
    echo "Building cache at $CACHE_DIR"
    git clone --depth=1 https://github.com/xianyi/OpenBLAS

    pushd OpenBLAS
    make FC=gfortran &> /dev/null && make PREFIX=${CACHE_DIR} install
    echo $(git rev-parse HEAD) > ${CACHE_DIR}/last_commit_id
    popd
fi

# Instrument the build state
echo OpenBLAS version:$(cat ${CACHE_DIR}/last_commit_id)
cat /proc/cpuinfo

# copy the cache files into /usr
sudo cp -r ${CACHE_DIR}/* /usr/

# install gonum/blas against OpenBLAS
export CGO_LDFLAGS="-L/usr/lib -lopenblas"
go get -v -x gonum.org/v1/netlib/...
