#include "PythonOutput.h"
#include <fstream>
#include <stdlib.h>

using namespace fwk;
using namespace utl;
using namespace fevt;
using namespace det;
using namespace std;
namespace PythonOutput {


PythonOutput::PythonOutput() {}

PythonOutput::~PythonOutput(){}

VModule::ResultFlag
PythonOutput::Init()
{
  fMetaData=true;
  fMinSignal=0;
  fOutputDir="./";

  CentralConfig* cc = CentralConfig::GetInstance();
  Branch topBranch = cc->GetTopBranch("PythonOutput");

  if (topBranch.GetChild("metaData")){
    topBranch.GetChild("metaData").GetData(fMetaData);
  }
  if (topBranch.GetChild("minSignal")){
    topBranch.GetChild("minSignal").GetData(fMinSignal);
  }
  if (topBranch.GetChild("outputDir")){
    topBranch.GetChild("outputDir").GetData(fOutputDir);
  }

  return eSuccess;
}

VModule::ResultFlag
PythonOutput::Run(evt::Event& event)
{
  nSigPE=0;
  fill(&iphe[0][0][0], &iphe[0][0][0] + (128*144*48), 0);
  Input(event);
  if (nSigPE<fMinSignal)
    return eSuccess;
  eventNumber++;
  ofstream myfile;
  for (int igtu=0;igtu<128;igtu++){
    myfile.open("example-"+to_string(igtu)+".txt");
    for (int iy=0;iy<48;iy++){
      for (int ix=0;ix<144;ix++){
	myfile<<iphe[igtu][ix][iy]<<' ';
      }
      myfile<<endl;
    }
    myfile.close();
  }
  string s="python PythonOutput.py "+fOutputDir+" "+to_string(eventNumber);
  system(s.c_str());

  double fZenith =-1.;
  double fAzimuth=-1.;
  double fEnergy=-1.;
  double fCorePos[3];
  for (int i=0;i<3;i++)
    fCorePos[i]=-1.;
  if (event.HasSimShower()) {
    evt::ShowerSimData& SimShower = event.GetSimShower();
    const fdet::FDetector& detFD = Detector::GetInstance().GetFDetector();
    CoordinateSystemPtr telCS = detFD.GetEye(1).GetTelescope(1).GetTelescopeCoordinateSystem();
    CoordinateSystemPtr showerCS = SimShower.GetShowerCoordinateSystem();
    Point showerCore(0, 0, 0, showerCS);
    fCorePos[0] = showerCore.GetX(telCS);
    fCorePos[1] = showerCore.GetY(telCS);
    fCorePos[2] = showerCore.GetZ(telCS);
    fZenith = SimShower.GetZenith();
    fAzimuth= SimShower.GetAzimuth();
    fEnergy = SimShower.GetEnergy();
  }

  if (fMetaData){
    ofstream metaDataFile;
    string s2=fOutputDir+"/event-"+to_string(eventNumber)+".txt";
    metaDataFile.open(s2.c_str());
    metaDataFile<<"Energy:\t\t"<<fEnergy<<endl;
    metaDataFile<<"Zenith:\t\t"<<fZenith<<endl;
    metaDataFile<<"Azimuth:\t"<<fAzimuth<<endl;
    metaDataFile<<"Core x:\t\t"<<fCorePos[0]<<endl;
    metaDataFile<<"Core y:\t\t"<<fCorePos[1]<<endl;
    metaDataFile<<"Core z:\t\t"<<fCorePos[2]<<endl;
    metaDataFile<<"Signal PE:\t"<<nSigPE<<endl;
    metaDataFile.close();
  }


  return eSuccess;
}

VModule::ResultFlag
PythonOutput::Finish()
{
  return eSuccess;
}


  void PythonOutput::Input(evt::Event& event){

    FEvent& fdEvent = event.GetFEvent();

    for (fevt::FEvent::ConstEyeIterator eye = fdEvent.EyesBegin(ComponentSelector::eUnknown);
	 eye != fdEvent.EyesEnd(ComponentSelector::eUnknown); ++eye) {
      for (fevt::Eye::ConstTelescopeIterator tel = eye->TelescopesBegin(ComponentSelector::eUnknown);
	   tel != eye->TelescopesEnd(ComponentSelector::eUnknown); ++tel) {

	for (fevt::Telescope::ConstCCBIterator ccb = tel->CCBsBegin(ComponentSelector::eUnknown);
	     ccb != tel->CCBsEnd(ComponentSelector::eUnknown); ++ccb) {

	  for (fevt::CCB::ConstPDMIterator pdm = ccb->PDMsBegin(ComponentSelector::eUnknown);
	       pdm != ccb->PDMsEnd(ComponentSelector::eUnknown); ++pdm) {
	    int ipdm = pdm->GetId();
	    if (fPDMid < ipdm) fPDMid = ipdm;

	    for (fevt::PDM::ConstECIterator ec = pdm->ECsBegin(ComponentSelector::eUnknown);
		 ec != pdm->ECsEnd(ComponentSelector::eUnknown); ++ec) {
	      int iec = ec->GetId();

	      for (fevt::EC::ConstPMTIterator pmt = ec->PMTsBegin(ComponentSelector::eUnknown);
		   pmt != ec->PMTsEnd(ComponentSelector::eUnknown); ++pmt) {
		int ipmt = pmt->GetId();

		for (fevt::PMT::ConstPixelIterator pix = pmt->PixelsBegin(ComponentSelector::eUnknown);
		     pix != pmt->PixelsEnd(ComponentSelector::eUnknown); ++pix) {
		  int ipix = pix->GetId();
		  if(pix->HasRecData()){
		    const PixelRecData& pr = pix->GetRecData();

		    for (PixelRecData::ConstFADCTraceIterator trIt = pr.FADCTracesBegin(); trIt != pr.FADCTracesEnd(); ++trIt) {
		      if (static_cast<FdConstants::LightSource>(trIt->GetLabel()) == FdConstants::eTotal){
			const TraceI& trace = pix->GetRecData().GetFADCTrace(FdConstants::eTotal);
			int irow=0,icol=0;

			for(int igtu=0;igtu<128; igtu++){
			  irow = (ipix-1)/8;// irow per PMT
			  icol = ((ipix-1)%8);// icol per PMT
			  int ipmt_global = ((iec-1)) * 4 + (ipmt-1);
			  int xLoc=48*(ipdm-1) +(ipmt_global/12)*16+(((ipmt_global%12)/2)%2)*8  + irow;
			  int yLoc= pmtMapperY(ipmt_global%12)*8  + icol; 
			  iphe[igtu][xLoc][yLoc] = trace[igtu];
			}
		      }
		    }
		  }
		  	if(pix->HasSimData()){
		  const PixelSimData& ps = pix->GetSimData();

		  for (PixelRecData::ConstFADCTraceIterator trIts = ps.FADCTracesBegin(); trIts != ps.FADCTracesEnd(); ++trIts) {

		    if (static_cast<FdConstants::LightSource>(trIts->GetLabel()) == FdConstants::eSignalPE ){

		      const TraceI& tracePE = pix->GetSimData().GetFADCTrace(FdConstants::eSignalPE);
		      for(int igtu=0;igtu<128; igtu++){
			nSigPE += tracePE[igtu];
		      }
		    }
		  }
			}
		}
	      }
	    }
	  }
	}
      }
    }
  }




  int PythonOutput::pmtMapperY(int pmt){
  if (pmt==0 || pmt ==2)
    return 0;
  if (pmt==1 || pmt ==3)
    return 1;
  if (pmt==4 || pmt ==6)
    return 2;
  if (pmt==5 || pmt ==7)
    return 3;
  if (pmt==8 || pmt ==10)
    return 4;
  if (pmt==9 || pmt ==11)
    return 5;
  return 0;
  }
}
