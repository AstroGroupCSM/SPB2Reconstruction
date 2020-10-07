#
# General Makefile for the OfflineUser package
#
#
# If the environment variable JEMEUSOOFFLINEROOT is not set
# AND the executable 'jemeuso-offline-config' is not in your PATH
# the definition of the following variable is required
#
# JEMEUSOOFFLINEROOT = @prefix@
#
# Replace the wildcard expression with .cc file list if you do
# not want to compile all .cc files in this directory
#
USER_SRCS = $(wildcard *.cc)
#
# All .xml.in files will be transformed into .xml with correct
# config and schema locations
#
USER_XMLS = $(patsubst %.xml.in,%.xml,$(wildcard *.xml.in))
USER_XMLS += $(patsubst %.xml.in,%.xml,$(wildcard xml/*.xml.in))
#
# Give your executable a name
#
EXE = JemEusoOffline
#
#############################################################

## You should not need to change anything below this line ###

# Authors: T. Paul, S. Argiro, L. Nellen, D. Veberic
# $Id: Makefile.in 14717 2009-09-17 20:24:36Z lukas $
# Send bug reports to http://www.jemeuso.unam.mx/bugzilla/

.PHONY: all depend clean

ifdef JEMEUSOOFFLINEROOT
  JEMEUSOOFFLINECONFIG = $(JEMEUSOOFFLINEROOT)/bin/jemeuso-offline-config
else
  JEMEUSOOFFLINECONFIG = jemeuso-offline-config
endif

OBJS = $(USER_SRCS:.cc=.o)

CPPFLAGS    = $(shell $(JEMEUSOOFFLINECONFIG) --cppflags)
CXXFLAGS    = $(shell $(JEMEUSOOFFLINECONFIG) --cxxflags)
LDFLAGS     = $(shell $(JEMEUSOOFFLINECONFIG) --ldflags)
MAIN        = $(shell $(JEMEUSOOFFLINECONFIG) --main)
CONFIGFILES = $(shell $(JEMEUSOOFFLINECONFIG) --config)
XMLSCHEMALOCATION = $(shell $(JEMEUSOOFFLINECONFIG) --schema-location)

all: $(EXE) $(USER_XMLS)

$(EXE): $(OBJS)
	$(CXX) -o $@ $^ $(MAIN) $(CXXFLAGS) $(LDFLAGS) -lMinuit

%: %.in
	@echo -n "Generating $@ file..."
	@sed -e 's!@''CONFIGDIR@!$(CONFIGFILES)!g;s!@''SCHEMALOCATION@!$(XMLSCHEMALOCATION)!g' $< >$@
	@echo "done"

#############################################################
# gcc can generate the dependency list

depend: Make-depend

Make-depend: $(USER_SRCS)
	$(CPP) $(CPPFLAGS) -MM $^ > $@

clean:
	- rm -f *.o  *.so *.ps *.wrl core $(USER_XMLS) Make-depend

#############################################################
# 'make run' will run the thing

run: $(EXE) $(USER_XMLS)
	./$(EXE) -b bootstrap.xml && touch $@

#############################################################
# the lines below are for running with debugger 'make run_gdb'

.INTERMEDIATE: gdb.cmdl

# batch mode gdb needs a file with commands
gdb.cmdl:
	echo "r -b bootstrap.xml" > $@

run_gdb: gdb.cmdl $(EXE) $(USER_XMLS)
	gdb -batch -x $< ./$(EXE) && touch $@

-include Make-depend
