# Compiler and flags
NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_60  # Adjust sm_60 to match your GPU architecture

# Source files and object files
SOURCES = $(wildcard *.cu)
OBJECTS = $(SOURCES:.cu=.o)

# Executable name
EXECUTABLE = ml_algorithms

# Default target
all: $(EXECUTABLE)

# Link object files to create executable
$(EXECUTABLE): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $(OBJECTS) -o $@

# Compile CUDA source files into object files
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Clean up object files and executable
clean:
	rm -f $(OBJECTS) $(EXECUTABLE)

# Phony targets
.PHONY: all clean