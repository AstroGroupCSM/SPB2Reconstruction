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
  double fTime[fMaxPhoton];
  double fTheta[fMaxPhoton];
  double fPhi[fMaxPhoton];
  double fWeight[fMaxPhoton];
  int fType[fMaxPhoton];
///new declarations
  double fEnergy;
  double fZenith, fAzimuth;
  double x,y,z;
  double core[2];


  REGISTER_MODULE("UserModule", UserModule);
};


#endif //_UserModule_h

// End:
