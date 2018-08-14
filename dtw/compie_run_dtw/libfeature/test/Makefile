CC = g++
CXX = g++

CXXFLAGS = -I ../include/ -I../../libutility/include
LDPATH = -L../lib/$(MACHINE) -L../../libutility/lib/$(MACHINE)
LDFLAGS = -lfeature -lutility 

MACHINE = $(shell uname -m)

.PHONY: test

test: test.cpp
	$(CXX) $^ $(CXXFLAGS) $(LDPATH) $(LDFLAGS) -o $@

clean:
	-rm -rf test
