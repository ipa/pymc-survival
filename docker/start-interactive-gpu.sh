#!/bin/bash

bsub -Is -q gpu-medium -M 16 -W 3:00 -n 5 /bin/bash


