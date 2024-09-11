#!/bin/bash

THIS_UID=`id -u`
THIS_GID=`id -g`
THIS_USER=$USER

docker build \
--build-arg MYUID=$THIS_UID \
--build-arg MYGID=$THIS_GID \
--build-arg MYUSER=$THIS_USER \
-t hack_eage_hpc_2024 .
