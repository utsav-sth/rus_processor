all: utils/libevent_reducer.so utils/libxz_tracking.so utils/libyz_tracking.so utils/libglobal_tracking.so utils/libvertexing.so utils/libdimuon_building.so

NVCC ?= nvcc

CUDA_ARCH = \
 -gencode arch=compute_52,code=sm_52 \
 -gencode arch=compute_61,code=sm_61 \
 -gencode arch=compute_70,code=sm_70 \
 -gencode arch=compute_80,code=sm_80 \
 -gencode arch=compute_86,code=sm_86 \
 -gencode arch=compute_52,code=compute_52

NVCCFLAGS := $(CUDA_ARCH) -Xcompiler -fPIC -std=c++17 -DDEBUG=1
INCDIRS := -Ikernels/include
CUDA_LIBDIR ?= /usr/local/cuda/lib64
LIBS := -L$(CUDA_LIBDIR) -lcudadevrt -lcudart

EVENT_OBJ = kernels/event_reducer.o
$(EVENT_OBJ): kernels/event_reducer.cu
	@echo "CU to O $@"
	@$(NVCC) $(NVCCFLAGS) -dc -rdc=true $(INCDIRS) -o $@ $<

utils/libevent_reducer.so: $(EVENT_OBJ)
	@echo "O to SO $@"
	@$(NVCC) $(NVCCFLAGS) -shared -rdc=true -o $@ $^ $(LIBS)
	@echo "$@ built"

XZ_OBJ = kernels/gkernel_xz_tracking.o
$(XZ_OBJ): kernels/gkernel_xz_tracking.cu kernels/include/gHits.cuh kernels/include/gPlane.cuh kernels/include/gTracklet.cuh
	@echo "CU to O $@"
	@$(NVCC) $(NVCCFLAGS) -dc -rdc=true $(INCDIRS) -o $@ $<

utils/libxz_tracking.so: $(XZ_OBJ)
	@echo "O to SO $@"
	@$(NVCC) $(NVCCFLAGS) -shared -rdc=true -o $@ $^ $(LIBS)
	@echo "$@ built"

YZ_OBJ = kernels/gkernel_yz_tracking.o
$(YZ_OBJ): kernels/gkernel_yz_tracking.cu kernels/include/gHits.cuh kernels/include/gPlane.cuh kernels/include/gTracklet.cuh
	@echo "CU to O $@"
	@$(NVCC) $(NVCCFLAGS) -dc -rdc=true $(INCDIRS) -o $@ $<

utils/libyz_tracking.so: $(YZ_OBJ)
	@echo "O to SO $@"
	@$(NVCC) $(NVCCFLAGS) -shared -rdc=true -o $@ $^ $(LIBS)
	@echo "$@ built"

GLB_OBJ = kernels/gkernel_global_tracking.o
$(GLB_OBJ): kernels/gkernel_global_tracking.cu
	@echo "CU to O $@"
	@$(NVCC) $(NVCCFLAGS) -dc -rdc=true $(INCDIRS) -o $@ $<

utils/libglobal_tracking.so: $(GLB_OBJ)
	@echo "O to SO $@"
	@$(NVCC) $(NVCCFLAGS) -shared -rdc=true -o $@ $^ $(LIBS)
	@echo "$@ built"

VTX_OBJ = kernels/gkernel_vertexing.o
$(VTX_OBJ): kernels/gkernel_vertexing.cu kernels/include/gPlane.cuh
	@echo "CU to O $@"
	@$(NVCC) $(NVCCFLAGS) -dc -rdc=true $(INCDIRS) -o $@ $<

utils/libvertexing.so: $(VTX_OBJ)
	@echo "O to SO $@"
	@$(NVCC) $(NVCCFLAGS) -shared -rdc=true -o $@ $^ $(LIBS)
	@echo "$@ built"

DIMU_OBJ = kernels/gkernel_dimuon_building.o
$(DIMU_OBJ): kernels/gkernel_dimuon_building.cu kernels/include/gHits.cuh kernels/include/gPlane.cuh
	@echo "CU to O $@"
	@$(NVCC) $(NVCCFLAGS) -dc -rdc=true $(INCDIRS) -o $@ $<

utils/libdimuon_building.so: $(DIMU_OBJ)
	@echo "O to SO $@"
	@$(NVCC) $(NVCCFLAGS) -shared -rdc=true -o $@ $^ $(LIBS)
	@echo "$@ built"

clean:
	@rm -f kernels/*.o utils/libevent_reducer.so utils/libxz_tracking.so utils/libyz_tracking.so utils/libglobal_tracking.so utils/libvertexing.so utils/libdimuon_building.so
	@echo "Cleaned"

.PHONY: all clean
