# Makefile for CUDA matrix multiplication project

# Compiler and flags
NVCC = nvcc
NVCC_FLAGS = -arch=sm_86 -std=c++17

# Target
TARGET = ./bin/matrixmult

# Files
SRC_FILES = ./src/matrixmult.cu ./src/kernel.cu
HEADER_FILES = ./src/dev_array.h ./src/kernel.h

# Build rules
all: $(TARGET)

$(TARGET): $(SRC_FILES) $(HEADER_FILES)
	$(NVCC) $(NVCC_FLAGS) $(SRC_FILES) -o $(TARGET)

clean:
	rm -f $(TARGET)

run:
	./bin/matrixmult.exe