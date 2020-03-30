LIST (APPEND JEMEUSO_SOURCES
  ${srcdir}/Prioritizer.cc
  ${srcdir}/Prioritizer.h
  )

LIST (APPEND config_DATA
  ${builddir}/Prioritizer.xml
  ${builddir}/Prioritizer.xsd
)

JEMEUSO_CREATE_CONFIGURE_FILES (  
${srcdir}/Prioritizer.xml.in
${srcdir}/Prioritizer.xsd.in
)
