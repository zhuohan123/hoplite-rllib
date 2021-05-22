#!/usr/bin/env bash

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM SIGHUP EXIT

sudo fuser -k 6666/tcp -s &> /dev/null
sudo fuser -k 50055/tcp -s &> /dev/null

my_address=$(ifconfig | grep 'inet.*broadcast' | awk '{print $2}')

hoplite_path=/home/ubuntu/efs/hoplite

pkill notification
sleep 1
export RAY_BACKEND_LOG_LEVEL=fatal
$hoplite_path/build/notification $my_address &
sleep 1

log_file=log/$1-$(date +"%Y%m%d-%H%M%S").log
log_latest=log/$1-latest.log

mkdir -p log

rllib train -v -f $1 --ray-address "auto" 2>&1 | tee -a ${log_file} &

ln -sfn $(realpath ${log_file}) ${log_latest}

sleep 1500
