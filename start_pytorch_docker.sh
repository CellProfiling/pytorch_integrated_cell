nvidia-docker run -it \
	-v /home/trang/3D-cell-model/pytorch_integrated_cell:/root/pytorch_integrated_cell \
        -v /home/trang/3D-cell-model/results:/root/results \
        -v /home/trang/2D-cell-model/Nuclei_nucleoli/h5:/root/data/h5 \
        rorydm/pytorch_extras:jupyter \
	bash 
