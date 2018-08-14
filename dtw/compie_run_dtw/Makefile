CC = g++
libutility_path = libutility
libdtw_path = libdtw
libfeature_path = libfeature
libsegtree_path = libsegtree
CFLAGS = -Wall -Werror -ansi -pedantic \
				 -I $(libdtw_path)/include/ -I $(libutility_path)/include \
				 -I $(libsegtree_path)/include -I $(libfeature_path)/include

MACHINE = $(shell uname -m)
LDFLAGS = -ldtw \
					-lfeature \
					-lsegtree \
					-lutility

LDPATH = -L $(libdtw_path)/lib/$(MACHINE) -L $(libutility_path)/lib/$(MACHINE) \
				 -L $(libsegtree_path)/lib/$(MACHINE) -L $(libfeature_path)/lib/$(MACHINE)

.PHONY: all clean


all: CFLAGS +=  -O2
debug: CFLAGS += -g

all:run_dtw


run_dtw: run_dtw.cpp
	${CC} ${CFLAGS} $(LDPATH) $< ${LDFLAGS} -o $@

clean:
	rm -f example example2
