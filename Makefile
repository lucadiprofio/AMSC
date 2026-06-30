CXX        := nvc++ #g++
CXXSTD     := -std=c++17
OPT        := -O2 -mp=gpu -gpu=cc89 -Minfo=accel #-O3 -DNDEBUG
WARN       := 
DEBUG      := #-g

# paths
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

LIBOBJS    := $(BUILDDIR)/particles.o $(BUILDDIR)/timer.o

TARGETS    := optim_par gpu_offload # wbal optim
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

# $(BUILDDIR)/wbal: $(MAINDIR)/wbal.cpp $(LIBOBJS) | $(BUILDDIR)
#	$(CXX) $(CXXFLAGS) $< $(LIBOBJS) -o $@ $(LDFLAGS) $(LDLIBS)

# $(BUILDDIR)/optim: $(MAINDIR)/optim.cpp $(LIBOBJS) | $(BUILDDIR)
#	$(CXX) $(CXXFLAGS) $< $(LIBOBJS) -o $@ $(LDFLAGS) $(LDLIBS)

$(BUILDDIR)/optim_par: $(MAINDIR)/optim_par.cpp $(LIBOBJS) | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $< $(LIBOBJS) -o $@ $(LDFLAGS) $(LDLIBS)

$(BUILDDIR)/gpu_offload: $(MAINDIR)/gpu_offload.cpp $(LIBOBJS) | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $< $(LIBOBJS) -o $@ $(LDFLAGS) $(LDLIBS)


clean:
	rm -rf $(BUILDDIR)
