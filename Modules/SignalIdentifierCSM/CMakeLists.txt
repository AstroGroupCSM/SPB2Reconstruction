LIST (APPEND JEMEUSO_SOURCES
  ${srcdir}/SignalIdentifier.cc
  ${srcdir}/SignalIdentifier.h
)
LIST (APPEND config_DATA
  ${builddir}/SignalIdentifier.xml
  ${builddir}/SignalIdentifier.xsd
)


JEMEUSO_CREATE_CONFIGURE_FILES (  
${srcdir}/SignalIdentifier.xml.in
${srcdir}/SignalIdentifier.xsd.in
)
