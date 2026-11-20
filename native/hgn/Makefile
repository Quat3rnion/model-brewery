CXX ?= c++
NVCC ?= nvcc
HIPCXX ?= hipcc
CUDA_ARCH ?= sm_120
ROCM_ARCH ?= gfx1151
BUILD ?= .build
SRC := cpu.cpp probe.cpp
HDR := hgn.h packed.h
CPU_FLAGS := -std=c++17 -O2 -Wall -Wextra -Werror

.PHONY: cpu cuda rocm
cpu: $(BUILD)/probe-cpu
cuda: $(BUILD)/probe-$(CUDA_ARCH)
rocm: $(BUILD)/probe-$(ROCM_ARCH)

$(BUILD):
	mkdir -p $@

$(BUILD)/probe-cpu: $(SRC) $(HDR) | $(BUILD)
	$(CXX) $(CPU_FLAGS) $(CXXFLAGS) $(SRC) -o $@

$(BUILD)/probe-$(CUDA_ARCH): $(SRC) $(HDR) device.cu | $(BUILD)
	$(NVCC) -std=c++17 -O2 -arch=$(CUDA_ARCH) $(NVCCFLAGS) -DHGN_DEVICE_ENABLED -DHGN_CUDA $(SRC) device.cu -o $@

$(BUILD)/probe-$(ROCM_ARCH): $(SRC) $(HDR) device.cu | $(BUILD)
	$(HIPCXX) -std=c++17 -O2 --offload-arch=$(ROCM_ARCH) $(HIPFLAGS) -DHGN_DEVICE_ENABLED -DHGN_ROCM -x hip $(SRC) device.cu -o $@
