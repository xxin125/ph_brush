NVCC ?= nvcc
CXX  ?= g++

ARCH ?= sm_120
CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -Werror
NVFLAGS  := -O3 -std=c++17 -arch=$(ARCH) -Xcompiler "-Wall,-Wextra" \
            -lineinfo

PRECISION ?= float
DEFINES :=
ifeq ($(PRECISION),float)
DEFINES += -DPME_USE_FLOAT_PRECISION
endif

CXXFLAGS += $(DEFINES)
NVFLAGS  += $(DEFINES)

INCLUDES := -Iinclude
LIBS := -lcufft

SRC_CPP := $(wildcard src/*.cpp)
SRC_CU  := $(wildcard src/*.cu)

OBJ_CPP := $(patsubst src/%.cpp, build/%.o, $(SRC_CPP))
OBJ_CU  := $(patsubst src/%.cu,  build/%.o, $(SRC_CU))

BIN := bin/MC

.PHONY: all debug clean

all: $(BIN)

debug: NVFLAGS += -G -O0
debug: CXXFLAGS += -O0 -g
debug: clean all

$(BIN): $(OBJ_CPP) $(OBJ_CU)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVFLAGS) $(INCLUDES) -o $@ $^ $(LIBS)

build/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

build/%.o: src/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c -o $@ $<

clean:
	rm -rf build bin
