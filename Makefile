CXX        ?= g++
CXXSTD     := -std=c++17
OPT        := -O3 -DNDEBUG
WARN       := -Wall -Wextra -Wno-unused-parameter
DEBUG      := -g

ROOT       := $(shell pwd)
QGINC      := $(ROOT)/quadgrid/include
QGSRC      := $(ROOT)/quadgrid/src
JSONINC    := $(ROOT)/json/single_include/nlohmann
TIMERINC   := $(ROOT)/timer/src
TIMERSRC   := $(ROOT)/timer/src
MAINDIR    := $(ROOT)/main
BUILDDIR   := $(ROOT)/build

INCLUDES   := -I$(QGINC) -I$(JSONINC) -I$(TIMERINC) -I$(MAINDIR)
CXXFLAGS   := $(CXXSTD) $(OPT) $(WARN) $(DEBUG) $(INCLUDES)
LDFLAGS    := 
LDLIBS     :=

LIBOBJS    := $(BUILDDIR)/particles.o $(BUILDDIR)/timer.o

# wbal exam optim_par
TARGETS    := godamp optim merge_split gpu_offload
EXES       := $(addprefix $(BUILDDIR)/, $(TARGETS))

# NVHPC compiler specific flags
NVCXX      ?= nvc++
NV_OPT     := -O4 -DNDEBUG
NV_GPU_ARCH:= # -gpu=cc89 # equivalente a -arch=sm_89 per nvc++


.PHONY: all clean libs

all: libs $(EXES)

libs: $(LIBOBJS)

$(BUILDDIR):
	@mkdir -p $(BUILDDIR)

$(BUILDDIR)/particles.o: $(QGSRC)/particles.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/timer.o: $(TIMERSRC)/timer.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/merge_split_bfs.o: $(MAINDIR)/merge_split_bfs.cpp | $(BUILDDIR)
	nvc++ $(CXXSTD) -O2 -g $(INCLUDES) -c $< -o $@


$(BUILDDIR)/godamp: $(MAINDIR)/godamp.cpp $(LIBOBJS) | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $< $(LIBOBJS) -o $@ $(LDFLAGS) $(LDLIBS)

#$(BUILDDIR)/wbal: $(MAINDIR)/wbal.cpp $(LIBOBJS) | $(BUILDDIR)
#	$(CXX) $(CXXFLAGS) $< $(LIBOBJS) -o $@ $(LDFLAGS) $(LDLIBS)

#$(BUILDDIR)/exam: $(MAINDIR)/exam.cpp $(LIBOBJS) | $(BUILDDIR)
#	$(CXX) $(CXXFLAGS) $< $(LIBOBJS) -o $@ $(LDFLAGS) $(LDLIBS)

$(BUILDDIR)/optim: $(MAINDIR)/optim.cpp $(LIBOBJS) | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $< $(LIBOBJS) -o $@ $(LDFLAGS) $(LDLIBS)

$(BUILDDIR)/merge_split: $(MAINDIR)/merge_split.cpp $(LIBOBJS) $(BUILDDIR)/merge_split_bfs.o | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $< $(LIBOBJS) $(BUILDDIR)/merge_split_bfs.o -o $@ $(LDFLAGS) $(LDLIBS)

$(BUILDDIR)/gpu_offload: $(MAINDIR)/gpu_offload.cpp $(LIBOBJS) $(BUILDDIR)/merge_split_bfs.o | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $< $(LIBOBJS) $(BUILDDIR)/merge_split_bfs.o -o $@ $(LDFLAGS) $(LDLIBS)


clean:
	rm -rf $(BUILDDIR)