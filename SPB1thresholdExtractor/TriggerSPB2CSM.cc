#include "TriggerSPB2CSM.h"

using namespace fwk;
using namespace utl;
using namespace fevt;
using namespace det;
using namespace std;

namespace TriggerSPB2CSM {



TriggerSPB2CSM::TriggerSPB2CSM() {}

TriggerSPB2CSM::~TriggerSPB2CSM(){}

VModule::ResultFlag
TriggerSPB2CSM::Init()
{
  INFO("Running TriggerSPB2CSM");
  iEvent=0;
  return eSuccess;
}

VModule::ResultFlag
TriggerSPB2CSM::Run(evt::Event& event)
{

  INFO("Running TriggerSPB2CSM");
  //Initialize arrays to zero
  Clear();

  //Read event in
  Input(event);
  fPath="event-"+to_string(iEvent)+".dat";
  ofstream myfile;
  myfile.open(fPath.c_str());
  for (int ipmt=0;ipmt<36;ipmt++){
    for (int ipixx=0;ipixx<8;ipixx++){
      for (int ipixy=0;ipixy<8;ipixy++){
        myfile<<iphe[0][ipmt][ipixx][ipixy]<<"\t"<<0<<"\t"<<ipmt<<"\t"<<ipixx<<"\t"<<ipixy<<endl;
      }
    }
  }




  return eSuccess;
}

VModule::ResultFlag
TriggerSPB2CSM::Finish()
{
  INFO("TriggerSPB2CSM");
  return eSuccess;
}
void TriggerSPB2CSM::Clear(){
  fill(&iphe[0][0][0][0], &iphe[0][0][0][0] + (3*36*8*8), 0);
}

  void TriggerSPB2CSM::Input(evt::Event& event){
    iEvent++;
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
			iphe[ipdm-1][ipmt_global][irow][icol] += trace[igtu];
      //if (trace[igtu]>0) cout<< trace[igtu]<<endl;
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
}
