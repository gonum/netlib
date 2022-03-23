#!/bin/bash

set -ex

go env
export CGO_LDFLAGS="-L/usr/lib -lopenblas"
go test -a -v ./...
