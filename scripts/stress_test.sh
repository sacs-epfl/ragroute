#!/bin/bash

root_dir=/home/dhasade/ragroute
save_dir=mtl
log_dir=$root_dir/results/$save_dir
mkdir -p $log_dir 
nfs_dir=/scratch/home/dhasade/ragroute

echo "==> Updating code from nfs"

cp -r ${nfs_dir}/ragroute $root_dir
cp -r ${nfs_dir}/*.py $root_dir

env_python=/home/dhasade/.conda/envs/rag/bin/python

echo "==> Running server"

$env_python $root_dir/stress_test.py
