#!/bin/bash

# cd /cfs/klemming/nobackup/t/trangle/pytorch_integrated_cell
cd ..
python /root/projects/train_model.py \
	--lrEnc 2E-4 --lrDec 2E-4 --lrEncD 2E-2 --lrDecD 2E-4 \
	--encDRatio 1E-4 --decDRatio 1E-5 \
	--model_name aaegan_256v3 \
	--save_dir ./results/nucleoli_1/ \
	--train_module aaegan_trainv6 \
	--noise 1E-2 \
	--imdir /root/data/PNG \
	--dataProvider DataProvider2DPNG \
	--saveStateIter 1 --saveProgressIter 1 \
	--channels_pt1 0 2 --channels_pt2 0 1 2 \
	--gpu_ids 0 1 \
	--batch_size 50  \
	--nlatentdim 16 \
	--nepochs 150 \
	--nepochs_pt2 150
