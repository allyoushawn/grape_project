CC = g++
CFLAGS = -Wall -Werror -ansi -pedantic -I /home/ugoc/include/

MACHINE = $(shell uname -m)
LDFLAGS = -lfeature \
					-lsegtree \
					-lutility

LDPATH = -L /home/ugoc/lib

.PHONY: all clean


all: CFLAGS +=  -O2
debug: CFLAGS += -g

all: example
debug: example

example: example.cpp
	${CC} ${CFLAGS} $(LDPATH) $< ${LDFLAGS} -o $@

clean:
	rm -f example
