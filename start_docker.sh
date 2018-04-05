nvidia-docker run -it \
	-v /home/trang/Desktop/3Dcell/pytorch_integrated_cell:/root/projects \
        -v /home/trang/Desktop/results:/root/results \
        -v /home/trang/Desktop/nucleoli_v18_U2OS_noccd:/root/data \
	rorydm/pytorch_extras:jupyter \
	bash 
