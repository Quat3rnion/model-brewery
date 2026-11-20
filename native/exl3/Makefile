# The core library uses only C++17. Only the checkpoint CLI needs json-c.
BUILD ?= build
NVCC ?= nvcc
HIPCC ?= hipcc
CUDA_ARCH ?= sm_120
HIP_ARCH ?= gfx1151
CXXFLAGS ?= -O2 -Wall -Wextra
CPUFLAGS = $(CPPFLAGS) $(CXXFLAGS) -std=c++17
JSON_C = $(shell pkg-config --cflags --libs json-c)
CUDAFLAGS = -O2 -std=c++17 -arch=$(CUDA_ARCH)
HIPFLAGS = -O2 -std=c++17 --offload-arch=$(HIP_ARCH)
HIPLDLIBS ?= -lamdhip64
CORE = $(BUILD)/libbrewery-exl3.a

.PHONY: all cpu cuda rocm test test-cpu test-cuda test-rocm
all: cpu
cpu: $(CORE) $(BUILD)/exl3-checkpoint
cuda: $(BUILD)/exl3-checkpoint-cuda
rocm: $(BUILD)/exl3-checkpoint-rocm
$(BUILD):
	mkdir -p "$@"
$(BUILD)/%.o: %.cpp exl3.h | $(BUILD)
	$(CXX) $(CPUFLAGS) -c $< -o "$@"
$(CORE): $(BUILD)/cpu.o $(BUILD)/encode.o
	$(AR) rcs "$@" $^
$(BUILD)/cuda.o: cuda.cu exl3.h gpu_forward.cuh | $(BUILD)
	$(NVCC) $(CUDAFLAGS) -c $< -o "$@"
$(BUILD)/rocm.o: rocm.hip exl3.h gpu_forward.cuh | $(BUILD)
	$(HIPCC) $(HIPFLAGS) -x hip -c $< -o "$@"
$(BUILD)/exl3-checkpoint: checkpoint.cpp $(CORE) exl3.h
	$(CXX) $(CPUFLAGS) $< $(CORE) $(JSON_C) -o "$@"
$(BUILD)/checkpoint-cuda.o: checkpoint.cpp exl3.h | $(BUILD)
	$(CXX) $(CPUFLAGS) -DEXL3_HAVE_CUDA $(shell pkg-config --cflags json-c) -c $< -o "$@"
$(BUILD)/checkpoint-rocm.o: checkpoint.cpp exl3.h | $(BUILD)
	$(CXX) $(CPUFLAGS) -DEXL3_HAVE_ROCM $(shell pkg-config --cflags json-c) -c $< -o "$@"
$(BUILD)/exl3-checkpoint-cuda: $(BUILD)/checkpoint-cuda.o $(BUILD)/cuda.o $(CORE)
	$(NVCC) $(CUDAFLAGS) $^ $(shell pkg-config --libs json-c) -o "$@"
$(BUILD)/exl3-checkpoint-rocm: $(BUILD)/checkpoint-rocm.o $(BUILD)/rocm.o $(CORE)
	$(HIPCC) $(HIPFLAGS) $(LDFLAGS) $^ $(shell pkg-config --libs json-c) $(HIPLDLIBS) -o "$@"
$(BUILD)/test_cpu: test_cpu.cpp $(CORE) exl3.h
	$(CXX) $(CPUFLAGS) $< $(CORE) -o "$@"
$(BUILD)/test_encode: test_encode.cpp $(CORE) exl3.h
	$(CXX) $(CPUFLAGS) $< $(CORE) -o "$@"
$(BUILD)/test_cuda.o: test_cuda.cpp exl3.h | $(BUILD)
	$(NVCC) $(CUDAFLAGS) -c $< -o "$@"
$(BUILD)/test_rocm.o: test_rocm.cpp exl3.h | $(BUILD)
	$(CXX) $(CPUFLAGS) -c $< -o "$@"
$(BUILD)/test_cuda: $(BUILD)/test_cuda.o $(BUILD)/cuda.o $(CORE)
	$(NVCC) $(CUDAFLAGS) $^ -o "$@"
$(BUILD)/test_rocm: $(BUILD)/test_rocm.o $(BUILD)/rocm.o $(CORE)
	$(HIPCC) $(HIPFLAGS) $(LDFLAGS) $^ $(HIPLDLIBS) -o "$@"
test: test-cpu
test-cpu: cpu $(BUILD)/test_cpu $(BUILD)/test_encode
	$(BUILD)/test_cpu
	$(BUILD)/test_encode
	sh test_checkpoint.sh "$(abspath $(BUILD))/exl3-checkpoint"
test-cuda: cuda $(BUILD)/test_cuda
	$(BUILD)/test_cuda
	sh test_checkpoint.sh "$(abspath $(BUILD))/exl3-checkpoint-cuda" cuda
test-rocm: rocm $(BUILD)/test_rocm
	$(BUILD)/test_rocm
	sh test_checkpoint.sh "$(abspath $(BUILD))/exl3-checkpoint-rocm" rocm
