## TODO Figure out why this compilation doesn't work.

NVCC = nvcc
INCLUDES := -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include"
NVCC_FLAGS := -gencode=arch=compute_20,code=\"sm_20,compute_20\" --use-local-env --cl-version 2015 --compile -cudart static
CCBIN := -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin"
#CUDA_LIBS := -L$(CUDA_INSTALL_PATH)\lib -lcudart

rectify.o:
	$(NVCC) $(NVCC_FLAGS) $(CCBIN) $(INCLUDES) rectify.cu