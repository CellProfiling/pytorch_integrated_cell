python train_model.py \
	--gpu_ids 3 \
	--batch_size 32 \
	--imsize 128 \
	--nlatentdim 16 \
	--nepochs 250 \
	--nepochs_pt2 250 \
	--lrEnc 2E-4 \
	--lrDec 2E-4 \
	--lrEncD 2E-4 \
	--lrDecD 2E-4 \
	--encDRatio 1E-3 \
	--decDRatio 1E-5 \
	--model_name aaegan_v3 \
	--save_dir ./test_aaegan/aaegan_v3_single_backprop/ \
	--train_module aaegan_train_single_backprop \
	--noise=0.01
