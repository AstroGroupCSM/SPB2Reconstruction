LIST (APPEND JEMEUSO_SOURCES
  ${srcdir}/BackgroundSubtractor.cc
  ${srcdir}/BackgroundSubtractor.h
  )

LIST (APPEND config_DATA
  ${builddir}/BackgroundSubtractor.xml
  ${builddir}/BackgroundSubtractor.xsd
)

JEMEUSO_CREATE_CONFIGURE_FILES (  
${srcdir}/BackgroundSubtractor.xml.in
${srcdir}/BackgroundSubtractor.xsd.in
)
