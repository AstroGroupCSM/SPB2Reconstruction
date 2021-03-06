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
      void ReadThresholds();
      void SaveTriggerData(evt::Event& event);
      int pmtMapperY(int pmt); //Function for mapping pmt number to Y Location
      int irow;  //Used to save data in offline format
      int icol;
      int icell; // Place to keep cell number
      int fPDMid;
      int nHot;
      bool fSignalOnly;
      bool fReadFromFile;
      int nActive;
      int fnumPDMs;
      int triggerState;
      int total;
      int triggerGTU;
      int lenThreshold;
      int fLengthTrigger;
      int fVerbosityLevel;
      int nPersist=0;
      double nSigma;
      std::string fPath;
      int iphe[3][36][8][8][128]; //Photo-electron trace
      int ipheSig[3][36][16][128]; //Photo-electron trace
      int valCells[3][36][16][128]; //pe trace for cells
      int sumCells[3][36][16]; //Integrated cell pe value
      int sumPixels[3][36][8][8];
      int HotOrNot[24][24][128];
      int triggerData[3][128];
      int dummy[100];
      REGISTER_MODULE("TriggerSPB2CSM",TriggerSPB2CSM);
    };
}

#endif // _TriggerSPB2CSM_TriggerSPB2CSM_h_
