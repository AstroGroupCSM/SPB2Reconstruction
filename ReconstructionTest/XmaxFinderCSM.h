#ifndef _XmaxFinderCSM_XmaxFinderCSM_h_
#define _XmaxFinderCSM_XmaxFinderCSM_h_

/**
 * \file
 * \author G. Filippatos
 * \date 16 Nov 2020
 */

#include <fwk/VModule.h>
#include <boost/utility.hpp>
#include <fevt/FEvent.h>
#include <fevt/Eye.h>
#include <fevt/Telescope.h>
#include <fevt/TelescopeSimData.h>
#include <fevt/TelescopeRecData.h>
#include <fevt/CoordinateData.h>

#include <fevt/CoordinatesRecData.h>
#include <det/Detector.h>

#include <evt/ShowerSimData.h>
#include <fdet/FDetector.h>
#include <fdet/Eye.h>
#include <fdet/Telescope.h>

#include <atm/Atmosphere.h>
#include <atm/ProfileResult.h>
#include <atm/InclinedAtmosphericProfile.h>


#include <utl/PhysicalConstants.h>
#include "TF1.h"
#include "TH1.h"
#include "TH3.h"
#include "TCanvas.h"
#include <algorithm>
/*
 * Avoid using using namespace declarations in your headers,
 * doing so makes all symbols from each namespace visible
 * to the client which includes your header.
 */

namespace XmaxFinderCSM {

  class XmaxFinderCSM :
      public boost::noncopyable,
      public fwk::VModule {
  public:
    XmaxFinderCSM();
    ~XmaxFinderCSM();
    VModule::ResultFlag Init();
    VModule::ResultFlag Run(evt::Event& e);
    VModule::ResultFlag Finish();
  private:
    double fR0;
    double fPsi0;
    double fTheta;
    double fPhi;
    double rho_0;
    double dz;
    double dl;
    double scale_height;
    double hd;
    double dist;
    double fCorePosReco[3];
    double slope[3];
  std::vector<double>   fTi;
  std::vector<double>   fPsii;
  std::vector<double>   fSignal;
  std::vector<double>   hi;
  std::vector<double>   Ri;
  std::vector<double>   dE;
  std::vector<double>   X;
  std::vector<double>   dEdX;
    REGISTER_MODULE("XmaxFinder",XmaxFinderCSM);
  };
}

#endif // _XmaxFinderCSM_XmaxFinderCSM_h_
