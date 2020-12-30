#!/bin/bash

set -ex

go env
go get -d -t -v ./...
export CGO_LDFLAGS="-L/usr/lib -lopenblas"
go test -a -v ./...
