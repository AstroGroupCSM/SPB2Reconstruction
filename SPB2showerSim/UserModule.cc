#include "UserModule.h"
#include <utl/ErrorLogger.h>
#include <evt/Event.h>
#include <fevt/FEvent.h>
#include <fevt/Eye.h>
#include <fevt/TelescopeSimData.h>
#include <fevt/Telescope.h>
#include <evt/ShowerSimData.h>
#include <det/Detector.h>
#include <fdet/FDetector.h>
#include <fdet/Eye.h>

//Allows local definitions of coordinates
#include <fwk/LocalCoordinateSystem.h>

//Commonly used points are hardcoded into a registry to avoid
//having to repeatly create the locations
#include <fwk/CoordinateSystemRegistry.h>

//Tools
#include <utl/JemEusoUnits.h>
#include <utl/TimeStamp.h>
#include <utl/UTCDateTime.h>

//Geometry Related Headers
#include <utl/ReferenceEllipsoid.h>
#include <utl/CoordinateSystemPtr.h>
#include <utl/UTMPoint.h>
#include <utl/Point.h>
#include <utl/AxialVector.h>
#include <utl/Vector.h>
#include <utl/TransformationMatrix.h>

//Use of the boost c++ package to extract
//coorindates from vectors
#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>
#include <boost/tuple/tuple_io.hpp>

#include <det/Detector.h>
#include <fdet/FDetector.h>
#include <fdet/Eye.h>
#include <fdet/Telescope.h>
#include <fdet/Camera.h>

#include <TTree.h>
#include <TFile.h>

#include <iostream>

#include <fwk/CentralConfig.h>


using namespace fwk;
using namespace utl;
using namespace fevt;
using namespace det;
using namespace std;
using namespace boost;

UserModule::UserModule() {

}

UserModule::~UserModule()
{
}


VModule::ResultFlag
UserModule::Init()
{
   string filename;
   CentralConfig* cc = CentralConfig::GetInstance();
   Branch topBranch = cc->GetTopBranch("DataWriter");
   topBranch.GetChild("FileName").GetData(filename);
  string test = "fSim" + filename;

  fOutFile = new TFile(test.c_str(), "RECREATE");
  fTree = new TTree("Photons","");
  fTree->Branch("nPhoton", &fNPhotons, "nPhoton/i");
  fTree->Branch("theta", fTheta, "theta[nPhoton]/D");
  fTree->Branch("phi", fPhi, "phi[nPhoton]/D");
  fTree->Branch("theta_shower", fTheta_shower, "theta_shower[nPhoton]/D");
  fTree->Branch("phi_shower", fPhi_shower, "phi_shower[nPhoton]/D");
  fTree->Branch("type", fType, "type[nPhoton]/I");
  fTree->Branch("weight", fWeight, "weight[nPhoton]/D");
  fTree->Branch("time", fTime, "time[nPhoton]/D");
  fTree->Branch("wavelength", wavelength, "wavelength[nPhoton]/D");

  //Branches with general shower info
  fTree->Branch("energy",&fEnergy,"energy/D");
  fTree->Branch("zenith",&fZenith,"zenith/D");
  fTree->Branch("azimuth",&fAzimuth,"azimuth/D");

  fTree->Branch("xcore",&x,"x/D");
  fTree->Branch("ycore",&y,"y/D");
  fTree->Branch("zcore",&z,"z/D");

  return eSuccess;
}


VModule::ResultFlag
UserModule::Run(evt::Event& event)
{

  fOutFile->cd();
  Detector& detector = Detector::GetInstance();
  const fdet::FDetector& detFD = detector.GetFDetector();

//////////////Begin additions to UserModule.cc added by J.Fenn, 05/2015

  //Get event data
  FEvent& fEvent = event.GetFEvent();
  const fdet::Eye& myeye = detFD.GetEye(1);
  string fDetName = myeye.GetName();


  //Get event parameters from FEvent class
  INFO(" UserModule->Passing on shower info: ENERGY, ZENITH, AZIMUTH, CORE LOCATION");
  evt::ShowerSimData& SimDat = event.GetSimShower();
  fEnergy = log10(SimDat.GetEnergy());
  fAzimuth = SimDat.GetAzimuth() / degree;
  fZenith = SimDat.GetZenith() / degree;


  //From the fDetector class setup a coordinate system (CS) at Eye-center
  CoordinateSystemPtr jemeusoCS = detFD.GetEye(fDetName).GetEyeCoordinateSystem();
  //Acquire core location in eye-centric CS
  Point detectorPOSITION = detFD.GetEye(fDetName).GetPosition();
  //  cout << "Detector Position = " << detectorPOSITION.GetCoordinates(jemeusoCS)  << endl;

  //Create a point with the shower position
  Point showerPOSITION = SimDat.GetPosition();
  Triple showercore = showerPOSITION.GetCoordinates(jemeusoCS);
  cout << "Shower Position = " << showercore << endl;
  boost::tie(x, y, z) = showercore;
  cout << "x = " << x << ", y = " << y << ", z = " << z << endl;
  cout << " Shower core assigned = " << x << ", " << y << endl;




//////////////End code additions by J.Fenn

  unsigned int currPhoton = 0;
  for (FEvent::EyeIterator iEye =
         fEvent.EyesBegin(ComponentSelector::eInDAQ);
       iEye != fEvent.EyesEnd(ComponentSelector::eInDAQ) ; ++iEye) {
    for (Eye::TelescopeIterator iTel = iEye->TelescopesBegin(ComponentSelector::eInDAQ);
         iTel != iEye->TelescopesEnd(ComponentSelector::eInDAQ); ++iTel) {

      const fdet::Telescope& detTel = detFD.GetTelescope(*iTel);
      const CoordinateSystemPtr telCS = detTel.GetTelescopeCoordinateSystem();
      const CoordinateSystemPtr showerCS = SimDat.GetShowerCoordinateSystem();;

      const TelescopeSimData& telSim = iTel->GetSimData();

      for (TelescopeSimData::ConstPhotonIterator iPhoton =
             telSim.PhotonsBegin(); iPhoton != telSim.PhotonsEnd(); ++iPhoton) {
        if (currPhoton < fMaxPhoton) {
          const Vector& direction = iPhoton->GetDirection();
          fTheta[currPhoton] = direction.GetTheta(telCS) / degree;
          fPhi[currPhoton] = direction.GetPhi(telCS) / degree;
          fTheta_shower[currPhoton] = direction.GetTheta(showerCS) / degree;
          fPhi_shower[currPhoton] = direction.GetPhi(showerCS) / degree;
          fType[currPhoton] = iPhoton->GetSource();
          fWeight[currPhoton] = iPhoton->GetWeight();
          fTime[currPhoton] = iPhoton->GetTime().GetNanoSecond();
          wavelength[currPhoton] = iPhoton->GetWavelength();
          ++currPhoton;
        }
        else {
          WARNING("photons in TTree truncated!");
          break;
        }
      }
    }
  }
  fNPhotons = currPhoton;
  fTree->Fill();
  return eSuccess;
}


VModule::ResultFlag
UserModule::Finish()
{
  fOutFile->cd();
  fOutFile->Write();
  fOutFile->Close();

  return eSuccess;

}

