#!/usr/bin/env sh
set -e
/home/eryuan/soft/caffe-mobilenet_senet/build/tools/caffe train \
          --solver=/home/eryuan/projects/Pycharm_projects/eyeState_detect/train/solver.prototxt -gpu 0 \
         2>&1 | tee $LOG

