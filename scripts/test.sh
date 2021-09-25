#!/usr/bin/env bash

set -e

_FLOATX=${FLOATX:=float64}
AESARA_FLAGS="floatX=${_FLOATX},gcc__cxxflags='-march=core2'" pytest -v --cov=pymc --cov-report=xml "$@" --cov-report term
