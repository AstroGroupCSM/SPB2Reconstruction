#ifndef _Trigger_TriggerCSM_h_
#define _Trigger_TriggerCSM_h_

/**
 * \file
 * \author G. Filippatos
 * \date 08 Apr 2020
 */

#include <fwk/VModule.h>
#include <boost/utility.hpp>
#include <evt/Event.h>
#include <utl/ErrorLogger.h>
#include <utl/Trace.h>
#include <evt/Header.h>
#include <fevt/FEvent.h>
#include <fevt/Eye.h>
#include <fevt/Telescope.h>
#include <fdet/FDetector.h>
#include <det/Detector.h>

#include <fwk/CentralConfig.h>



namespace Trigger {

  class TriggerCSM : 
      public boost::noncopyable,
      public fwk::VModule {
  public:
    TriggerCSM();
    ~TriggerCSM();
    VModule::ResultFlag Init();
    VModule::ResultFlag Run(evt::Event& e);
    VModule::ResultFlag Finish();
  private:
    REGISTER_MODULE("TriggerCSM",TriggerCSM);
  };
}

#endif // _Trigger_TriggerCSM_h_
