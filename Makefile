CXX        := nvc++
CXXSTD     := -std=c++17
OPT        := -O2 -mp=gpu -gpu=cc89 -Minfo=accel
WARN       := 
DEBUG      := 

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
LDFLAGS    := -mp=gpu -gpu=cc89
LDLIBS     :=

LIBOBJS    := $(BUILDDIR)/particles.o $(BUILDDIR)/timer.o

TARGETS    := godamp wbal exam optim optim_par merge_split gpu_offload
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

$(BUILDDIR)/merge_split_bfs.o: $(MAINDIR)/merge_split_bfs.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@


$(BUILDDIR)/godamp: $(MAINDIR)/godamp.cpp $(LIBOBJS) | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $< $(LIBOBJS) -o $@ $(LDFLAGS) $(LDLIBS)

$(BUILDDIR)/wbal: $(MAINDIR)/wbal.cpp $(LIBOBJS) | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $< $(LIBOBJS) -o $@ $(LDFLAGS) $(LDLIBS)

$(BUILDDIR)/exam: $(MAINDIR)/exam.cpp $(LIBOBJS) | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $< $(LIBOBJS) -o $@ $(LDFLAGS) $(LDLIBS)

$(BUILDDIR)/optim: $(MAINDIR)/optim.cpp $(LIBOBJS) | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $< $(LIBOBJS) -o $@ $(LDFLAGS) $(LDLIBS)

$(BUILDDIR)/merge_split: $(MAINDIR)/merge_split.cpp $(LIBOBJS) $(BUILDDIR)/merge_split_bfs.o | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $< $(LIBOBJS) $(BUILDDIR)/merge_split_bfs.o -o $@ $(LDFLAGS) $(LDLIBS)

$(BUILDDIR)/gpu_offload: $(MAINDIR)/gpu_offload.cpp $(LIBOBJS) $(BUILDDIR)/merge_split_bfs.o | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $< $(LIBOBJS) $(BUILDDIR)/merge_split_bfs.o -o $@ $(LDFLAGS) $(LDLIBS)


clean:
	rm -rf $(BUILDDIR)
