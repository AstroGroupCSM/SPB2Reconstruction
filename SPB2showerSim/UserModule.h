#ifndef _UserModule_h
#define _UserModule_h

#include <fwk/VModule.h>
#include <vector>

class TFile;
class TTree;

class UserModule : public fwk::VModule {

public:
  UserModule();
  virtual ~UserModule();

  fwk::VModule::ResultFlag Init();
  fwk::VModule::ResultFlag Run(evt::Event& event);
  fwk::VModule::ResultFlag Finish();


private :

  TFile* fOutFile;
  TTree* fTree;

  static const unsigned int fMaxPhoton = 1000000;
  unsigned int fNPhotons;
  double fTime[fMaxPhoton] = {0};
  double fTheta[fMaxPhoton] = {0};
  double fPhi[fMaxPhoton] = {0};
  double fTheta_shower[fMaxPhoton] = {0};
  double fPhi_shower[fMaxPhoton] = {0};
  double fWeight[fMaxPhoton] = {0};
  int fType[fMaxPhoton] = {0};
  double wavelength[fMaxPhoton] = {0};
///new declarations
  double fEnergy;
  double fZenith, fAzimuth;
  double x,y,z;
  double core[2];


  REGISTER_MODULE("UserModule", UserModule);
};


#endif //_UserModule_h

// End:
