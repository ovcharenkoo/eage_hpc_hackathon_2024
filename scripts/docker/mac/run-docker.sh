#!/bin/bash

        # --rm \
docker run -it \
	-v ${PWD}:/workspace \
    --name hack_eage_hpc_2024 \
	-p 8887:8888 \
	hack_eage_hpc_2024 \
	/bin/bash
