LIST (APPEND JEMEUSO_SOURCES
  ${srcdir}/NewSPB2TriggerModule.cc
  ${srcdir}/NewSPB2TriggerModule.h
  )

LIST (APPEND config_DATA
	${builddir}/NewSPB2TriggerModuleCSM.xml
	${builddir}/NewSPB2TriggerModuleCSM.xsd
)

JEMEUSO_CREATE_CONFIGURE_FILES (  
	${srcdir}/NewSPB2TriggerModuleCSM.xml.in
${srcdir}/NewSPB2TriggerModuleCSM.xsd.in
)
