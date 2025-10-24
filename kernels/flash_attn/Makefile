GPU_TARGET=H100

# Compiler
NVCC?=nvcc

NVCCFLAGS_BASE=-Xcompiler=-fPIE -Xcompiler -fopenmp --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing -forward-unknown-to-host-compiler -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++20 -MD -MT -MF -x cu -lrt -lpthread -ldl -DKITTENS_HOPPER -arch=sm_90a -lcuda -lcudadevrt -lcudart_static -lcublas -lgomp -I ./../../ThunderKittens/include -I ./../../ThunderKittens/prototype

ifeq ($(DEBUG),1)
	NVCCFLAGS=$(NVCCFLAGS_BASE) -g -G -O0
else
	NVCCFLAGS=$(NVCCFLAGS_BASE) -DNDEBUG --use_fast_math -O3
endif

TARGET=h100
SRC=h100.cu

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(TARGET)

debug:
	$(MAKE) DEBUG=1

# Clean target
clean:
	rm -f $(TARGET)