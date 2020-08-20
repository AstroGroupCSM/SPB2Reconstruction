#ifndef _TriggerSPB2CSM_TriggerSPB2CSM_h_
#define _TriggerSPB2CSM_TriggerSPB2CSM_h_

/**
 * \file
 * \author G. Filippatos
 * \date 10 Feb 2020
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
#include <fstream>
#include <fdet/FDetector.h>
#include <det/Detector.h>
#include <fwk/CentralConfig.h>


namespace TriggerSPB2CSM {

  class TriggerSPB2CSM :
    public boost::noncopyable,
    public fwk::VModule {
  public:
      TriggerSPB2CSM();
      ~TriggerSPB2CSM();
      VModule::ResultFlag Init();
      VModule::ResultFlag Run(evt::Event& e);
      VModule::ResultFlag Finish();
  private:
      void Input(evt::Event& event); //Reads event
      void Clear();
      int irow;  //Used to save data in offline format
      int icol;
      int icell; // Place to keep cell number
      int fPDMid;
      int iEvent;
      std::string fPath;
      int iphe[3][36][8][8]; //Photo-electron trace
      REGISTER_MODULE("SPB1Extractor",TriggerSPB2CSM);
    };
}

#endif // _TriggerSPB2CSM_TriggerSPB2CSM_h_
