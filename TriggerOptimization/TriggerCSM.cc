#include "TriggerCSM.h"

using namespace fwk;
using namespace utl;
using namespace det;
using namespace fevt;
namespace Trigger {

TriggerCSM::TriggerCSM() {}

TriggerCSM::~TriggerCSM(){}

VModule::ResultFlag
TriggerCSM::Init()
{
  VModule* m1= VModuleFactory::Create("TriggerSPB2CSM");
  m1->Init();
  VModule* m3= VModuleFactory::Create("TriggerSPB2cells3x3TG");
  m3->Init();

  return eSuccess;
}

VModule::ResultFlag
TriggerCSM::Run(evt::Event& event)
{
  Input(event);
  double fZenith =-1.;
  double fAzimuth=-1.;
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
  }
  event.SetTriggerState(0);
  VModule* m1= VModuleFactory::Create("TriggerSPB2CSM");
  m1->Run(event);
  int triggerState0=event.HasTrigger();
  event.SetTriggerState(0);
  VModule* m3= VModuleFactory::Create("TriggerSPB2cells3x3TG");
  m3->Run(event);
  int triggerState2=event.HasTrigger();
  event.SetTriggerState(0);
  std::cout<<"TRIGGERS\tTriggerSPB2CSM\t"<<triggerState0<<"\tTriggerSPB2cells3x3TG\t" <<triggerState2<<"\tZenith\t"<<fZenith<<"\tAzimuth\t"<<fAzimuth<<"\tLocation\t"<<fCorePos[0]<<"\t"<<fCorePos[1]<<"\t"<<fCorePos[2]<<"\t"<<nSigPE<<std::endl;
  return eSuccess;
}

VModule::ResultFlag
TriggerCSM::Finish()
{
  VModule* m1= VModuleFactory::Create("TriggerSPB2CSM");
  m1->Finish();
  return eSuccess;
}

    void TriggerCSM::Input(evt::Event& event){
      nSigPE=0;
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


	      for (fevt::EC::ConstPMTIterator pmt = ec->PMTsBegin(ComponentSelector::eUnknown);
		   pmt != ec->PMTsEnd(ComponentSelector::eUnknown); ++pmt) {


		for (fevt::PMT::ConstPixelIterator pix = pmt->PixelsBegin(ComponentSelector::eUnknown);
		     pix != pmt->PixelsEnd(ComponentSelector::eUnknown); ++pix) {

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


}
