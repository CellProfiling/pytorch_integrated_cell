#!/usr/bin/env bash

python inception_score.py \
	--parent_dir '/root/results/nucleoli_numt_latent16_1' \
	--batch_size 100 \
	--gpu_ids 0 1
