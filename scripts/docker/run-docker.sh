#!/bin/bash


# -v /scratch/llms:/workspace/llms \
# -v /var/tmp:/var/tmp \
#-u $(id -un) \

        # --rm \
docker run -it \
	--gpus=all \
	-v ${PWD}:/workspace \
	-p 8888:8888 \
        --name nim_copilot \
	nim_copilot \
	/bin/bash
