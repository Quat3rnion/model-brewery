BUILD ?= build
UPSTREAM ?=
NVCC ?= nvcc
HIPCC ?= hipcc
CUDA_ARCH ?= sm_120
HIP_ARCH ?= gfx1151
CXXFLAGS ?= -O2 -Wall -Wextra
CFLAGS ?= -O2
HOST = $(CPPFLAGS) $(CXXFLAGS) -std=c++17 -I$(BUILD)/pinned
CUDA = $(NVCC) -O2 -std=c++17 -arch=$(CUDA_ARCH) -I$(BUILD)/pinned
HIP = $(HIPCC) -O2 -std=c++17 --offload-arch=$(HIP_ARCH) -I$(BUILD)/pinned
PIN = $(BUILD)/pinned/gyro_codebook.inc
CORE = $(BUILD)/gyro.o
ORACLE = $(BUILD)/tq_ref.o $(BUILD)/tq_codebook.o

.PHONY: all cpu cuda rocm test-cpu test-cuda test-rocm verify-pin
all: cpu
cpu: $(BUILD)/libgyro.a $(BUILD)/gyro-probe-cpu
cuda: $(BUILD)/libgyro-cuda.a $(BUILD)/gyro-probe-cuda
rocm: $(BUILD)/libgyro-rocm.a $(BUILD)/gyro-probe-rocm
test-cpu: cpu
	$(BUILD)/gyro-probe-cpu
test-cuda: cuda
	$(BUILD)/gyro-probe-cuda
test-rocm: rocm
	$(BUILD)/gyro-probe-rocm

# Stage and verify before compiling. Keep timestamps when the pin is unchanged.
verify-pin:
	@test -n "$(UPSTREAM)" || { echo "Set UPSTREAM to the pinned Agention archive root" >&2; exit 1; }
	sh stage_pin.sh "$(UPSTREAM)" "$(BUILD)/pinned"
$(PIN): | verify-pin
	@test -f "$@"
$(CORE): gyro.cpp gyro.h layout.h $(PIN)
	$(CXX) $(HOST) -c $< -o $@
$(BUILD)/tq_ref.o: $(PIN)
	$(CC) $(CFLAGS) -std=c99 -ffp-contract=off -c $(BUILD)/pinned/tq_ref.c -o $@
$(BUILD)/tq_codebook.o: $(PIN)
	$(CC) $(CFLAGS) -std=c99 -c $(BUILD)/pinned/tq_codebook.c -o $@
$(BUILD)/probe.o: probe.cpp fixtures.h gyro.h $(PIN)
	$(CXX) $(HOST) -c $< -o $@
$(BUILD)/probe-gpu.o: probe.cpp fixtures.h gyro.h $(PIN)
	$(CXX) $(HOST) -DGYRO_GPU -c $< -o $@
$(BUILD)/gyro-cuda.o: gyro_gpu.cu gpu_backend.h gyro.h layout.h $(PIN)
	$(CUDA) -c $< -o $@
$(BUILD)/gyro-rocm.o: gyro_gpu.cu gpu_backend.h gyro.h layout.h $(PIN)
	$(HIP) -x hip -c $< -o $@
$(BUILD)/probe-cuda.o: probe_gpu.cu gpu_backend.h gyro.h $(PIN)
	$(CUDA) -c $< -o $@
$(BUILD)/probe-rocm.o: probe_gpu.cu gpu_backend.h gyro.h $(PIN)
	$(HIP) -x hip -c $< -o $@
$(BUILD)/libgyro.a: $(CORE)
	$(AR) rcs $@ $^
$(BUILD)/libgyro-cuda.a: $(CORE) $(BUILD)/gyro-cuda.o
	$(AR) rcs $@ $^
$(BUILD)/libgyro-rocm.a: $(CORE) $(BUILD)/gyro-rocm.o
	$(AR) rcs $@ $^
$(BUILD)/gyro-probe-cpu: $(BUILD)/probe.o $(ORACLE) $(BUILD)/libgyro.a
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -lm -o $@
$(BUILD)/gyro-probe-cuda: $(BUILD)/probe-gpu.o $(BUILD)/probe-cuda.o $(ORACLE) $(BUILD)/libgyro-cuda.a
	$(CUDA) $(LDFLAGS) $^ $(LDLIBS) -o $@
$(BUILD)/gyro-probe-rocm: $(BUILD)/probe-gpu.o $(BUILD)/probe-rocm.o $(ORACLE) $(BUILD)/libgyro-rocm.a
	$(HIP) $(LDFLAGS) $^ $(LDLIBS) -o $@
