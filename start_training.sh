#!/bin/bash

# cd /cfs/klemming/nobackup/t/trangle/pytorch_integrated_cell

python pytorch_integrated_cell/train_model.py \
	--lrEnc 2E-4 --lrDec 2E-4 --lrEncD 1E-2 --lrDecD 2E-4 \
	--encDRatio 1E-4 --decDRatio 1E-5 \
	--model_name aaegan3Dv6-exp \
	--save_dir ./nucleoli_test/ \
	--train_module aaegan_trainv6 \
	--noise 1E-2 \
	--imdir /cfs/klemming/nobackup/t/trangle/nucleoli_v18_U2OS_noccd/h5_nuclei \
	--dataProvider DataProvider2Dfake3D \
	--saveStateIter 1 --saveProgressIter 1 \
	--channels_pt1 0 2 --channels_pt2 0 1 2 \
	--gpu_ids 0 3 \
	--batch_size 20  \
	--nlatentdim 32 \
	--nepochs 50 \
	--nepochs_pt2 50