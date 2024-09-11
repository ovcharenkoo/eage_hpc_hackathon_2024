#!/bin/bash


# -v /scratch/llms:/workspace/llms \
# -v /var/tmp:/var/tmp \
#-u $(id -un) \

        # --rm \
docker run -it \
	--gpus=all \
	-v ${PWD}:/workspace \
	-p 8888:8888 \
        --name hack_eage_hpc_2024 \
	hack_eage_hpc_2024 \
	/bin/bash
