/* 
THIS IS A DUMMY MODULE

only use it if you are 
1.) Re-simulating showers, and
2.) your input showers have no background to start with 
*/


#include "SignalIdentifier.h"

using namespace fwk;
using namespace utl;
using namespace fevt;
using namespace det;
using namespace std;

namespace SignalId {



SignalIdentifierCSM::SignalIdentifierCSM() {}

SignalIdentifierCSM::~SignalIdentifierCSM(){}

VModule::ResultFlag 
SignalIdentifierCSM::Init()
{
 
  
  return eSuccess;
}

VModule::ResultFlag 
SignalIdentifierCSM::Run(evt::Event& event)
{
  INFO("IDENTIFYING SIGNAL");
  Input(event);
  return eSuccess;
}

VModule::ResultFlag 
SignalIdentifierCSM::Finish() 
{
  return eSuccess;
}

void SignalIdentifierCSM::Input(evt::Event& event){
  
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
		if(pix->HasSimData()){
		  const PixelSimData& pr = pix->GetSimData();

		  for (PixelSimData::ConstFADCTraceIterator trIt = pr.FADCTracesBegin(); trIt != pr.FADCTracesEnd(); ++trIt) {
		    if (static_cast<FdConstants::LightSource>(trIt->GetLabel()) == FdConstants::eTotal){
		      const TraceI& trace = pix->GetSimData().GetFADCTrace(FdConstants::eTotal);
		      pix->GetSimData().MakeFADCTrace(trace, FdConstants::eSignalPE);

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

