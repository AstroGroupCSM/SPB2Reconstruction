#ifndef _PythonOutput_PythonOutput_h_
#define _PythonOutput_PythonOutput_h_

/**
 * \file
 * \author G. Filippatos
 * \date 04 Mar 2020
 */

#include <fwk/VModule.h>
#include <boost/utility.hpp>
#include <utl/ErrorLogger.h>
#include <utl/Trace.h>
#include <evt/Event.h>
#include <evt/Header.h>
#include <fevt/FEvent.h>
#include <fevt/Eye.h>
#include <fevt/Telescope.h>
#include <fdet/FDetector.h>
#include <det/Detector.h>
#include <fwk/CentralConfig.h>



namespace PythonOutput {

  class PythonOutput : 
      public boost::noncopyable,
      public fwk::VModule {
  public:
    PythonOutput();
    ~PythonOutput();
    VModule::ResultFlag Init();
    VModule::ResultFlag Run(evt::Event& e);
    VModule::ResultFlag Finish();
  private:
    void Input(evt::Event& event); //Reads event
    int pmtMapperY(int pmt);
    int eventNumber=0;
    int fPDMid;
    int nSignal=0;
    int nNoise=0;
    int nSigPE;
    int iphe[128][144][48];
    REGISTER_MODULE("PythonOutput",PythonOutput);
  };
}

#endif // _PythonOutput_PythonOutput_h_
