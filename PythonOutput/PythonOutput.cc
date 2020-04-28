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
  return eSuccess;
}

VModule::ResultFlag 
PythonOutput::Run(evt::Event& event)
{
  nSigPE=0;
  fill(&iphe[0][0][0], &iphe[0][0][0] + (128*144*48), 0);
  Input(event);
  eventNumber++;
  ofstream myfile;
  for (int igtu=0;igtu<128;igtu++){
    myfile.open("txts/example-"+to_string(igtu)+".txt");
    for (int iy=0;iy<48;iy++){
      for (int ix=0;ix<144;ix++){
	myfile<<iphe[igtu][ix][iy]<<' ';
      }
      myfile<<endl;
    }
    myfile.close();
  }
  


  
  string s="python dummy.py "+to_string(nSigPE)+" "+to_string(eventNumber);
  system(s.c_str());
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
			  int xLoc=48*(ipdm-1) + pmtMapperY(ipmt_global%12)*8  + icol; 
			  int yLoc=(ipmt_global/12)*16+(((ipmt_global%12)/2)%2)*8  + irow;
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
		      int irow=0,icol=0;
		      for(int igtu=0;igtu<128; igtu++){
			irow = (ipix-1)/8;// irow per PMT
			icol = ((ipix-1)%8);// icol per PMT
			int ipmt_global = ((iec-1)) * 4 + (ipmt-1);
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
  }
}

