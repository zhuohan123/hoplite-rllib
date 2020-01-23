#!/usr/bin/env bash

for i in generated-*.yaml; do
    /home/ubuntu/efs/zhuohan/object_store/restart_all_workers.sh
    sleep 5
    echo "=====" $i "====="
    ./test_rllib_hoplite.sh $i
    sleep 10
done
/home/ubuntu/efs/zhuohan/object_store/restart_all_workers.sh