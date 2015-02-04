# To build a gibbs sampler for the phone-learning project

CC = g++
#CC=icpc
CFLAGS = -c -Wall -O3 -fomit-frame-pointer -msse2 -mssse3 
# CFLAGS = -c -O3
#CFLAGS = -c -xhost -parallel -O3 
#CFLAGS = -c -Wall -g
SOURCES = src/main.cc src/manager.cc src/sampler.cc src/sample_boundary_info.cc src/cluster.cc src/gmm.cc src/mixture.cc src/segment.cc src/bound.cc src/calculator.cc src/storage.cc
OBJECTS=$(SOURCES:.cc=.o)
EXECUTABLE = bin/dnn-phone-learning
#EXECUTABLE = gibbs-icpc
#EXECUTABLE = gibbs-icpc-quad

ifeq ($(INTEL_TARGET_ARCH), ia32)
MKL_LINKS=-Wl,--start-group -lmkl_intel -lmkl_intel_thread -lmkl_core -Wl,--end-group -liomp5 -lpthread
else
MKL_LINKS=-Wl,--start-group -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -Wl,--end-group -liomp5 -lpthread
endif

MKL_FLAGS=-I$(MKLROOT)/include -L$(MKLROOT)/lib/$(INTEL_ARCH) $(MKL_LINKS)
IPP_PATHS=-I$(IPPROOT)/include -L$(IPPROOT)/lib/$(INTEL_ARCH)

all: $(SOURCES) $(EXECUTABLE) 

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(MKL_LINKS) 

.cc.o:
	$(CC) $(CFLAGS)  $< -o $@ $(MKL_LINKS) 

clean:
	rm -rf *.o 
