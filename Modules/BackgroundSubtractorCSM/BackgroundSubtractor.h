#ifndef _BGSub_BackgroundSubtractorCSM_h_
#define _BGSub_BackgroundSubtractorCSM_h_

/**
 * \author G. Filippatos
 * \date 24 Jan 2020
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



namespace BGSub {

  class BackgroundSubtractorCSM : 
      public boost::noncopyable,
      public fwk::VModule {
  public:
    BackgroundSubtractorCSM();
    ~BackgroundSubtractorCSM();
    VModule::ResultFlag Init();
    VModule::ResultFlag Run(evt::Event& e);
    VModule::ResultFlag Finish();
  private:
    void Input(evt::Event& event, int16_t iphe[3][36][8][8][128]); //Reads event
    void Output(evt::Event& event, utl::TraceI TraceBS[3][36][8][8] ); //Saves BS event 
    int pmtMapperY(int pmt); //Function for mapping pmt number to Y Location
    int irow;  //Used to save data in offline format
    int icol;
    int icell; // Place to keep cell number 
    int fPDMid;
    int nHot;
    double sCell;
 
  REGISTER_MODULE("BackgroundSubtractorCSM",BackgroundSubtractorCSM);
  };
}

#endif // _BGSub_BackgroundSubtractor_h_
