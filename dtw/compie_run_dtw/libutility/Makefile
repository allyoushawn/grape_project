CC=g++
CXX=g++

CXXFLAGS=-Iinclude

MACHINE=$(shell uname -m)

SRC = utility.cpp thread_util.cpp std_common.cpp
OBJ = $(addprefix obj/$(MACHINE)/,$(SRC:.cpp=.o))

TARGET = lib/$(MACHINE)/libutility.a

vpath %.cpp src
vpath %.o obj/$(MACHINE)
vpath %.a lib/$(MACHINE)

.PHONY: mk_machine_dir all clean allclean

all: CXXFLAGS:=-Wall -Werror -O2 $(CXXFLAGS)

all: mk_machine_dir $(TARGET)

debug: CXXFLAGS += -DDEBUG -g

debug: $(TARGET)

%.d: %.cpp
	$(CC) -M $(CXXFLAGS) $< > $@

lib/$(MACHINE)/libutility.a: \
	obj/$(MACHINE)/utility.o \
	obj/$(MACHINE)/thread_util.o \
	obj/$(MACHINE)/std_common.o
	$(AR) rucs $@ $^

obj/$(MACHINE)/%.o: src/%.cpp
	$(CC) -c $(CXXFLAGS) -o $@ $^

mk_machine_dir:
	@mkdir -p obj/$(MACHINE)
	@mkdir -p lib/$(MACHINE)

allclean: clean
	$(RM) $(TARGET)

clean:
	$(RM) $(OBJ)
