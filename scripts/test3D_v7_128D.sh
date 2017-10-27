cd ..
python train_model.py --gpu_ids 0 1 2 --batch_size 30 --data_save_path ./test_aaegan/aaegan3Dv7_128D/data.pyt --nlatentdim 128 --nepochs 250 --nepochs_pt2 300 --lrEnc 2E-4 --lrDec 2E-4 --lrEncD 1E-2 --lrDecD 2E-4 --encDRatio 5E-4 --decDRatio 5E-4 --model_name aaegan3Dv6 --save_dir ./test_aaegan/aaegan3Dv7_128D/ --train_module aaegan_trainv3 --noise=0 --imdir /root/results/ipp_dataset_cellnuc_seg_curated_8_24_17 --dataProvider DataProvider3Dh5 --saveStateIter 1 --saveProgressIter 1 --channels_pt1 0 2 --channels_pt2 0 1 2
