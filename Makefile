CXX        := nvc++
CXXSTD     := -std=c++17
OPT        := -O2 -mp=gpu -gpu=cc89
WARN       :=
DEBUG      := #-g

ROOT       := $(shell pwd)
QGINC      := $(ROOT)/quadgrid/include
QGSRC      := $(ROOT)/quadgrid/src
JSONINC    := $(ROOT)/json/single_include/nlohmann
TIMERINC   := $(ROOT)/timer/src
TIMERSRC   := $(ROOT)/timer/src
MAINDIR    := $(ROOT)/main
BUILDDIR   := $(ROOT)/build

# includes, compilation flags and libraries
INCLUDES   := -I$(QGINC) -I$(JSONINC) -I$(TIMERINC) -I$(MAINDIR)
CXXFLAGS   := $(CXXSTD) $(OPT) $(WARN) $(DEBUG) $(INCLUDES)
LDFLAGS    := -mp=gpu -gpu=cc89
LDLIBS     :=

LIBOBJS    := $(BUILDDIR)/particles.o $(BUILDDIR)/timer.o $(BUILDDIR)/gpu_kernels.o

TARGETS    := gpu_offload
EXES       := $(addprefix $(BUILDDIR)/, $(TARGETS))

.PHONY: all clean libs

all: libs $(EXES)

libs: $(LIBOBJS)

$(BUILDDIR):
	@mkdir -p $(BUILDDIR)

$(BUILDDIR)/particles.o: $(QGSRC)/particles.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/timer.o: $(TIMERSRC)/timer.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/gpu_kernels.o: $(MAINDIR)/gpu_kernels.cpp $(MAINDIR)/gpu_kernels.h | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/gpu_offload: $(MAINDIR)/gpu_offload.cpp $(LIBOBJS) | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $< $(LIBOBJS) -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -rf $(BUILDDIR)
