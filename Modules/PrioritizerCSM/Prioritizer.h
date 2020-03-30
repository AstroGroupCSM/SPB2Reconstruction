 #ifndef _Prior_PrioritizerCSM_h_
#define _Prior_PrioritizerCSM_h_

/**
 * \file
 * \author G. Filippatos
 * \date 12 Mar 2020
 */

#include <fwk/VModule.h>
#include <boost/utility.hpp>
#include <utl/Trace.h>
#include <evt/Event.h>
#include <evt/Header.h>
#include <fevt/FEvent.h>
#include <fevt/Eye.h>
#include <fevt/Telescope.h>
#include <fdet/FDetector.h>
#include<fwk/CentralConfig.h>
#include <det/Detector.h>
#include <evt/ShowerSimData.h>

namespace Prior {

  class PrioritizerCSM : 
      public boost::noncopyable,
      public fwk::VModule {
  public:
    PrioritizerCSM();
    ~PrioritizerCSM();
    VModule::ResultFlag Init();
    VModule::ResultFlag Run(evt::Event& e);
    VModule::ResultFlag Finish();
  private:
    void Input(evt::Event& e); //Reads event
    int xMapper(int ipmt,int ipix);
    int yMapper(int ipmt,int ipix);
    
    double sCell;
    long total[2];
    int rt;
    int nHot;
    REGISTER_MODULE("PrioritizerCSM",PrioritizerCSM);
  };
}

#endif // _Prior_PrioritizerCSM_h_
