LIST (APPEND JEMEUSO_SOURCES
  ${srcdir}/BackgroundSimulator.cc
  ${srcdir}/BackgroundSimulator.h
)

LIST (APPEND config_DATA
  ${builddir}/BackgroundSimulatorCSM.xml
  ${builddir}/BackgroundSimulatorCSM.xsd
  ${srcdir}/TAbg10deg.txt
  ${srcdir}/TAbg15deg.txt
  ${srcdir}/TAbg23deg.txt
)

JEMEUSO_CREATE_CONFIGURE_FILES (
  ${srcdir}/BackgroundSimulatorCSM.xml.in
  ${srcdir}/BackgroundSimulatorCSM.xsd.in
)
