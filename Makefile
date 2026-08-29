# bytes-in-flight -- build the HBM bandwidth ladder benchmark.
#
#   make                 # -arch=native (needs a visible GPU at build time)
#   make ARCH=sm_90      # Hopper  (H100 / H200, HBM3 / HBM3e)
#   make ARCH=sm_100     # Blackwell
#   make clean

ARCH      ?= native
NVCC      ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(ARCH) --ptxas-options=-v -lineinfo

BIN := bin/bif
SRC := src/main.cu
HDR := src/kernels.cuh

.PHONY: all clean
all: $(BIN)

$(BIN): $(SRC) $(HDR)
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $@

clean:
	rm -rf bin
