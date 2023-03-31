#!/bin/bash

bsub -Is -q short -M 8 -W 2:00 -n 5 /bin/bash


