#include "Prioritizer.h"

using namespace fwk;
using namespace fevt;
using namespace utl;
using namespace det;
using namespace std;
using namespace evt;

namespace Prior {


PrioritizerCSM::PrioritizerCSM() {}

PrioritizerCSM::~PrioritizerCSM(){}

VModule::ResultFlag 
PrioritizerCSM::Init()
{
  INFO("PRORITiZING");
  rt=0;
  CentralConfig* cc = CentralConfig::GetInstance();
  Branch topBranch = cc->GetTopBranch("PrioritizerCSM");
  if (topBranch.GetChild("ReqTrig")){
    topBranch.GetChild("ReqTrig").GetData(rt);
  }

  return eSuccess;
}

VModule::ResultFlag 
PrioritizerCSM::Run(evt::Event& event)
{
  
  total[0]=0;total[1]=1;
  Input(event);
  int ts=0;
  ts = event.HasTrigger();
  cout <<event.GetHeader().GetTime();
  float z =-1;
  if (event.HasSimShower()){		 
    evt::ShowerSimData& SimShower = event.GetSimShower();
    float z ;
    z= SimShower.GetZenith();
  }
  
  cout <<"\tSimulated PE\t"<<ts<<"\tZENITH\t"<<z<<"\tTOTAL:\t"<<(float)total[0]/((float)total[1]*128.)<<"\tPIXELS\t"<<total[1]<<endl;
  
  return eSuccess;
}

VModule::ResultFlag 
PrioritizerCSM::Finish() 
{
  return eSuccess;
}
  
  void PrioritizerCSM::Input(evt::Event& event){
  
    FEvent& fdEvent = event.GetFEvent();
    int fPDMid;

  
    for (FEvent::ConstEyeIterator eye = fdEvent.EyesBegin(ComponentSelector::eUnknown);
	 eye != fdEvent.EyesEnd(ComponentSelector::eUnknown); ++eye) {
      for (Eye::ConstTelescopeIterator tel = eye->TelescopesBegin(ComponentSelector::eUnknown);
	   tel != eye->TelescopesEnd(ComponentSelector::eUnknown); ++tel) {
      
	for (Telescope::ConstCCBIterator ccb = tel->CCBsBegin(ComponentSelector::eUnknown);
	     ccb != tel->CCBsEnd(ComponentSelector::eUnknown); ++ccb) {
	
	  for (CCB::ConstPDMIterator pdm = ccb->PDMsBegin(ComponentSelector::eUnknown);
	       pdm != ccb->PDMsEnd(ComponentSelector::eUnknown); ++pdm) {
	    int ipdm = pdm->GetId();
	    if (fPDMid < ipdm) fPDMid = ipdm;
	  
	    for (PDM::ConstECIterator ec = pdm->ECsBegin(ComponentSelector::eUnknown);
		 ec != pdm->ECsEnd(ComponentSelector::eUnknown); ++ec) {
	      int iec = ec->GetId();
	    
	      for (EC::ConstPMTIterator pmt = ec->PMTsBegin(ComponentSelector::eUnknown);
		   pmt != ec->PMTsEnd(ComponentSelector::eUnknown); ++pmt) {
		int ipmt = pmt->GetId();
	      
		for (PMT::ConstPixelIterator pix = pmt->PixelsBegin(ComponentSelector::eUnknown);
		     pix != pmt->PixelsEnd(ComponentSelector::eUnknown); ++pix) {
		  int ipix = pix->GetId();
		
		  if(pix->HasRecData()){
		    const PixelRecData& pr = pix->GetRecData();
		  
		    for (PixelRecData::ConstFADCTraceIterator trIt = pr.FADCTracesBegin(); trIt != pr.FADCTracesEnd(); ++trIt) {
		      if (static_cast<FdConstants::LightSource>(trIt->GetLabel()) == FdConstants::eTotal){
			const utl::TraceI& trace = pix->GetRecData().GetFADCTrace(FdConstants::eTotal);
			int irow=0,icol=0;
			int temp =0;
			for(int igtu=0;igtu<128; igtu++){
			  total[0]+=trace[igtu];
			  if (trace[igtu]>0&&temp==0){
			    total[1]++;
			    temp=1;
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


  /*  Offline mapping of pixels in each PMT, PMTs in each EC, and EC in each PDM,
   as visible with the viewer etos

    Pixels                                   PMTs                       ECs (correct?)
    |0,7|1,7|2,7|3,7|4,7|5,7|6,7|7,7|        | 9|11|21|23|33|35|        |2|5|8|
    |0,6|1,6|2,6|3,6|4,6|5,6|6,6|7,6|        | 8|10|20|22|32|34|        |1|4|7|
    |0,5|1,5|2,5|3,5|4,5|5,5|6,5|7,5|        | 5| 7|17|19|29|31|        |0|3|6|
    |0,4|1,4|2,4|3,4|4,4|5,4|6,4|7,4|        | 4| 6|16|18|28|30|
    |0,3|1,3|2,3|3,3|4,3|5,3|6,3|7,3|        | 1| 3|13|15|25|27|
    |0,2|1,2|2,2|3,2|4,2|5,2|6,2|7,2|        | 0| 2|12|14|24|26|
    |0,1|1,1|2,1|3,1|4,1|5,1|6,1|7,1|
    |0,0|1,0|2,0|3,0|4,0|5,0|6,0|7,0|

*/

  
  int PrioritizerCSM::xMapper(int ipmt, int ipix){
    return (((ipmt%12))/2%2)*8+(ipix-1)/8;  
  }
  int PrioritizerCSM::yMapper(int ipmt,int ipix){
    int pmt = ipmt%12;
    int temp =0;
    if (pmt==0 || pmt ==2)
      temp= 0;
    if (pmt==1 || pmt ==3)
      temp= 1;
    if (pmt==4 || pmt ==6)
      temp= 2;
    if (pmt==5 || pmt ==7)
      temp= 3;
    if (pmt==8 || pmt ==10)
      temp= 4;
    if (pmt==9 || pmt ==11)
      temp= 5;
    return temp*8 + ((ipix-1)%8);
  }
  
}
