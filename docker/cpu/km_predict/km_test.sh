#!/bin/bash

trap 'echo "Cancelled by user"; exit' INT

cd /home/km_predict
pytest
