#####################################################################################################################
# All the source files in `minisat` folder were initially copied and later modified from https://github.com/feiwang3311/minisat #
# (which was taken from the MiniSat source at https://github.com/niklasso/minisat). The MiniSAT license is below.   #
#####################################################################################################################
# MiniSat -- Copyright (c) 2003-2006, Niklas Een, Niklas Sorensson
#            Copyright (c) 2007-2010  Niklas Sorensson
# 
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
# 
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# 
###################################################################################################

.PHONY:	r d p sh cr cd cp csh lr ld lp lsh config all install install-headers install-lib\
        install-bin clean distclean
all:	r lr lsh

## Load Previous Configuration ####################################################################

-include config.mk

## Configurable options ###########################################################################

# Directory to store object files, libraries, executables, and dependencies:
BUILD_DIR      ?= build

# Include debug-symbols in release builds
MINISAT_RELSYM ?= -g

# Sets of compile flags for different build types
MINISAT_REL    ?= -O3 -D NDEBUG
MINISAT_DEB    ?= -O0 -D DEBUG 
MINISAT_PRF    ?= -O3 -D NDEBUG
MINISAT_FPIC   ?= -fpic

# GNU Standard Install Prefix
prefix         ?= /usr/local

## Write Configuration  ###########################################################################

config:
	@( echo 'BUILD_DIR?=$(BUILD_DIR)'           ; \
	   echo 'MINISAT_RELSYM?=$(MINISAT_RELSYM)' ; \
	   echo 'MINISAT_REL?=$(MINISAT_REL)'       ; \
	   echo 'MINISAT_DEB?=$(MINISAT_DEB)'       ; \
	   echo 'MINISAT_PRF?=$(MINISAT_PRF)'       ; \
	   echo 'MINISAT_FPIC?=$(MINISAT_FPIC)'     ; \
	   echo 'prefix?=$(prefix)'                 ) > config.mk

## Configurable options end #######################################################################

INSTALL ?= install

# GNU Standard Install Variables
exec_prefix ?= $(prefix)
includedir  ?= $(prefix)/include
bindir      ?= $(exec_prefix)/bin
libdir      ?= $(exec_prefix)/lib
datarootdir ?= $(prefix)/share
mandir      ?= $(datarootdir)/man

# Target file names
MINISAT      = minisat#       Name of MiniSat main executable.
MINISAT_CORE = minisat_core#  Name of simplified MiniSat executable (only core solver support).
MINISAT_SLIB = lib$(MINISAT).a#  Name of MiniSat static library.
MINISAT_DLIB = lib$(MINISAT).so# Name of MiniSat shared library.

# Shared Library Version
SOMAJOR=2
SOMINOR=1
SORELEASE?=.0#   Declare empty to leave out from library file name.

MINISAT_CXXFLAGS = -I. -D __STDC_LIMIT_MACROS -D __STDC_FORMAT_MACROS -Wall -Wno-parentheses -Wextra
MINISAT_LDFLAGS  = -Wall -lz

ECHO=@echo
ifeq ($(VERB),)
VERB=@
else
VERB=
endif

SRCS = $(wildcard minisat/core/*.cc) $(wildcard minisat/simp/*.cc) $(wildcard minisat/utils/*.cc) $(wildcard minisat/gym/*.cc)
HDRS = $(wildcard minisat/mtl/*.h) $(wildcard minisat/core/*.h) $(wildcard minisat/simp/*.h) $(wildcard minisat/utils/*.h) $(wildcard minisat/gym/*.h)
OBJS = $(filter-out %Main.o, $(SRCS:.cc=.o))

r:	$(BUILD_DIR)/release/bin/$(MINISAT)
d:	$(BUILD_DIR)/debug/bin/$(MINISAT)
p:	$(BUILD_DIR)/profile/bin/$(MINISAT)
sh:	$(BUILD_DIR)/dynamic/bin/$(MINISAT)

cr:	$(BUILD_DIR)/release/bin/$(MINISAT_CORE)
cd:	$(BUILD_DIR)/debug/bin/$(MINISAT_CORE)
cp:	$(BUILD_DIR)/profile/bin/$(MINISAT_CORE)
csh:	$(BUILD_DIR)/dynamic/bin/$(MINISAT_CORE)

lr:	$(BUILD_DIR)/release/lib/$(MINISAT_SLIB)
ld:	$(BUILD_DIR)/debug/lib/$(MINISAT_SLIB)
lp:	$(BUILD_DIR)/profile/lib/$(MINISAT_SLIB)
lsh:	$(BUILD_DIR)/dynamic/lib/$(MINISAT_DLIB).$(SOMAJOR).$(SOMINOR)$(SORELEASE)

## Build-type Compile-flags:
$(BUILD_DIR)/release/%.o:			MINISAT_CXXFLAGS +=$(MINISAT_REL) $(MINISAT_RELSYM)
$(BUILD_DIR)/debug/%.o:				MINISAT_CXXFLAGS +=$(MINISAT_DEB) -g
$(BUILD_DIR)/profile/%.o:			MINISAT_CXXFLAGS +=$(MINISAT_PRF) -pg
$(BUILD_DIR)/dynamic/%.o:			MINISAT_CXXFLAGS +=$(MINISAT_REL) $(MINISAT_FPIC)

## Build-type Link-flags:
$(BUILD_DIR)/profile/bin/$(MINISAT):		MINISAT_LDFLAGS += -pg
$(BUILD_DIR)/release/bin/$(MINISAT):		MINISAT_LDFLAGS += --static $(MINISAT_RELSYM)
$(BUILD_DIR)/profile/bin/$(MINISAT_CORE):	MINISAT_LDFLAGS += -pg
$(BUILD_DIR)/release/bin/$(MINISAT_CORE):	MINISAT_LDFLAGS += --static $(MINISAT_RELSYM)

## Executable dependencies
$(BUILD_DIR)/release/bin/$(MINISAT):	 	$(BUILD_DIR)/release/minisat/simp/Main.o $(BUILD_DIR)/release/lib/$(MINISAT_SLIB)
$(BUILD_DIR)/debug/bin/$(MINISAT):	 	$(BUILD_DIR)/debug/minisat/simp/Main.o $(BUILD_DIR)/debug/lib/$(MINISAT_SLIB)
$(BUILD_DIR)/profile/bin/$(MINISAT):	 	$(BUILD_DIR)/profile/minisat/simp/Main.o $(BUILD_DIR)/profile/lib/$(MINISAT_SLIB)
# need the main-file be compiled with fpic?
$(BUILD_DIR)/dynamic/bin/$(MINISAT):	 	$(BUILD_DIR)/dynamic/minisat/simp/Main.o $(BUILD_DIR)/dynamic/lib/$(MINISAT_DLIB)

## Executable dependencies (core-version)
$(BUILD_DIR)/release/bin/$(MINISAT_CORE):	$(BUILD_DIR)/release/minisat/core/Main.o $(BUILD_DIR)/release/lib/$(MINISAT_SLIB)
$(BUILD_DIR)/debug/bin/$(MINISAT_CORE):	 	$(BUILD_DIR)/debug/minisat/core/Main.o $(BUILD_DIR)/debug/lib/$(MINISAT_SLIB)
$(BUILD_DIR)/profile/bin/$(MINISAT_CORE):	$(BUILD_DIR)/profile/minisat/core/Main.o $(BUILD_DIR)/profile/lib/$(MINISAT_SLIB)
# need the main-file be compiled with fpic?
$(BUILD_DIR)/dynamic/bin/$(MINISAT_CORE): 	$(BUILD_DIR)/dynamic/minisat/core/Main.o $(BUILD_DIR)/dynamic/lib/$(MINISAT_DLIB)

## Library dependencies
$(BUILD_DIR)/release/lib/$(MINISAT_SLIB):	$(foreach o,$(OBJS),$(BUILD_DIR)/release/$(o))
$(BUILD_DIR)/debug/lib/$(MINISAT_SLIB):		$(foreach o,$(OBJS),$(BUILD_DIR)/debug/$(o))
$(BUILD_DIR)/profile/lib/$(MINISAT_SLIB):	$(foreach o,$(OBJS),$(BUILD_DIR)/profile/$(o))
$(BUILD_DIR)/dynamic/lib/$(MINISAT_DLIB).$(SOMAJOR).$(SOMINOR)$(SORELEASE)\
 $(BUILD_DIR)/dynamic/lib/$(MINISAT_DLIB).$(SOMAJOR)\
 $(BUILD_DIR)/dynamic/lib/$(MINISAT_DLIB):	$(foreach o,$(OBJS),$(BUILD_DIR)/dynamic/$(o))

## Compile rules (these should be unified, buit I have not yet found a way which works in GNU Make)
$(BUILD_DIR)/release/%.o:	%.cc
	$(ECHO) Compiling: $@
	$(VERB) mkdir -p $(dir $@)
	$(VERB) $(CXX) $(MINISAT_CXXFLAGS) $(CXXFLAGS) -c -o $@ $< -MMD -MF $(BUILD_DIR)/release/$*.d

$(BUILD_DIR)/profile/%.o:	%.cc
	$(ECHO) Compiling: $@
	$(VERB) mkdir -p $(dir $@)
	$(VERB) $(CXX) $(MINISAT_CXXFLAGS) $(CXXFLAGS) -c -o $@ $< -MMD -MF $(BUILD_DIR)/profile/$*.d

$(BUILD_DIR)/debug/%.o:	%.cc
	$(ECHO) Compiling: $@
	$(VERB) mkdir -p $(dir $@)
	$(VERB) $(CXX) $(MINISAT_CXXFLAGS) $(CXXFLAGS) -c -o $@ $< -MMD -MF $(BUILD_DIR)/debug/$*.d

$(BUILD_DIR)/dynamic/%.o:	%.cc
	$(ECHO) Compiling: $@
	$(VERB) mkdir -p $(dir $@)
	$(VERB) $(CXX) $(MINISAT_CXXFLAGS) $(CXXFLAGS) -c -o $@ $< -MMD -MF $(BUILD_DIR)/dynamic/$*.d

## Linking rule
$(BUILD_DIR)/release/bin/$(MINISAT) $(BUILD_DIR)/debug/bin/$(MINISAT) $(BUILD_DIR)/profile/bin/$(MINISAT) $(BUILD_DIR)/dynamic/bin/$(MINISAT)\
$(BUILD_DIR)/release/bin/$(MINISAT_CORE) $(BUILD_DIR)/debug/bin/$(MINISAT_CORE) $(BUILD_DIR)/profile/bin/$(MINISAT_CORE) $(BUILD_DIR)/dynamic/bin/$(MINISAT_CORE):
	$(ECHO) Linking Binary: $@
	$(VERB) mkdir -p $(dir $@)
	$(VERB) $(CXX) $^ $(MINISAT_LDFLAGS) $(LDFLAGS) -o $@

## Static Library rule
%/lib/$(MINISAT_SLIB):
	$(ECHO) Linking Static Library: $@
	$(VERB) mkdir -p $(dir $@)
	$(VERB) $(AR) -rcs $@ $^

## Shared Library rule
$(BUILD_DIR)/dynamic/lib/$(MINISAT_DLIB).$(SOMAJOR).$(SOMINOR)$(SORELEASE)\
 $(BUILD_DIR)/dynamic/lib/$(MINISAT_DLIB).$(SOMAJOR)\
 $(BUILD_DIR)/dynamic/lib/$(MINISAT_DLIB):
	$(ECHO) Linking Shared Library: $@
	$(VERB) mkdir -p $(dir $@)
	$(VERB) $(CXX) $(MINISAT_LDFLAGS) $(LDFLAGS) -o $@ -shared -Wl,-soname,$(MINISAT_DLIB).$(SOMAJOR) $^
	$(VERB) ln -sf $(MINISAT_DLIB).$(SOMAJOR).$(SOMINOR)$(SORELEASE) $(BUILD_DIR)/dynamic/lib/$(MINISAT_DLIB).$(SOMAJOR)
	$(VERB) ln -sf $(MINISAT_DLIB).$(SOMAJOR) $(BUILD_DIR)/dynamic/lib/$(MINISAT_DLIB)

install:	install-headers install-lib install-bin
install-debug:	install-headers install-lib-debug

install-headers:
#       Create directories
	$(INSTALL) -d $(DESTDIR)$(includedir)/minisat
	for dir in mtl utils core simp gym; do \
	  $(INSTALL) -d $(DESTDIR)$(includedir)/minisat/$$dir ; \
	done
#       Install headers
	for h in $(HDRS) ; do \
	  $(INSTALL) -m 644 $$h $(DESTDIR)$(includedir)/$$h ; \
	done

install-lib-debug: $(BUILD_DIR)/debug/lib/$(MINISAT_SLIB)
	$(INSTALL) -d $(DESTDIR)$(libdir)
	$(INSTALL) -m 644 $(BUILD_DIR)/debug/lib/$(MINISAT_SLIB) $(DESTDIR)$(libdir)

install-lib: $(BUILD_DIR)/release/lib/$(MINISAT_SLIB) $(BUILD_DIR)/dynamic/lib/$(MINISAT_DLIB).$(SOMAJOR).$(SOMINOR)$(SORELEASE)
	$(INSTALL) -d $(DESTDIR)$(libdir)
	$(INSTALL) -m 644 $(BUILD_DIR)/dynamic/lib/$(MINISAT_DLIB).$(SOMAJOR).$(SOMINOR)$(SORELEASE) $(DESTDIR)$(libdir)
	ln -sf $(MINISAT_DLIB).$(SOMAJOR).$(SOMINOR)$(SORELEASE) $(DESTDIR)$(libdir)/$(MINISAT_DLIB).$(SOMAJOR)
	ln -sf $(MINISAT_DLIB).$(SOMAJOR) $(DESTDIR)$(libdir)/$(MINISAT_DLIB)
	$(INSTALL) -m 644 $(BUILD_DIR)/release/lib/$(MINISAT_SLIB) $(DESTDIR)$(libdir)

install-bin: $(BUILD_DIR)/dynamic/bin/$(MINISAT)
	$(INSTALL) -d $(DESTDIR)$(bindir)
	$(INSTALL) -m 755 $(BUILD_DIR)/dynamic/bin/$(MINISAT) $(DESTDIR)$(bindir)

python-wrap: $(BUILD_DIR)/dynamic/lib/$(MINISAT_DLIB).$(SOMAJOR).$(SOMINOR)$(SORELEASE) $(SRCS)
    # Graph-Q-SAT UPD: bump python version
	g++ -O2 -fPIC -c minisat/gym/GymSolver_wrap.cxx -o minisat/gym/GymSolver_wrap.o -I. -I/usr/include/python3.6
	g++ -shared -o minisat/gym/_GymSolver.so $(foreach o,$(OBJS),$(BUILD_DIR)/dynamic/$(o)) minisat/gym/GymSolver_wrap.o /usr/lib/x86_64-linux-gnu/libz.so

clean:
	rm -f $(foreach t, release debug profile dynamic, $(foreach o, $(SRCS:.cc=.o), $(BUILD_DIR)/$t/$o)) \
          $(foreach t, release debug profile dynamic, $(foreach d, $(SRCS:.cc=.d), $(BUILD_DIR)/$t/$d)) \
	  $(foreach t, release debug profile dynamic, $(BUILD_DIR)/$t/bin/$(MINISAT_CORE) $(BUILD_DIR)/$t/bin/$(MINISAT)) \
	  $(foreach t, release debug profile, $(BUILD_DIR)/$t/lib/$(MINISAT_SLIB)) \
	  $(BUILD_DIR)/dynamic/lib/$(MINISAT_DLIB).$(SOMAJOR).$(SOMINOR)$(SORELEASE)\
	  $(BUILD_DIR)/dynamic/lib/$(MINISAT_DLIB).$(SOMAJOR)\
	  $(BUILD_DIR)/dynamic/lib/$(MINISAT_DLIB)

distclean:	clean
	rm -f config.mk

## Include generated dependencies
-include $(foreach s, $(SRCS:.cc=.d), $(BUILD_DIR)/release/$s)
-include $(foreach s, $(SRCS:.cc=.d), $(BUILD_DIR)/debug/$s)
-include $(foreach s, $(SRCS:.cc=.d), $(BUILD_DIR)/profile/$s)
-include $(foreach s, $(SRCS:.cc=.d), $(BUILD_DIR)/dynamic/$s)
