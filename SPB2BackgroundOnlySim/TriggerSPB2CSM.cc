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

  CentralConfig* cc = CentralConfig::GetInstance();
  Branch topBranch = cc->GetTopBranch("TriggerSPB2CSM");

  //Default Values 
  nSigma=6.75;
  nHot=3;
  nPersist=3;
  nActive=3;
  fVerbosityLevel=0;
  fSignalOnly=true;
  //Optional xml parameters
  if (topBranch.GetChild("nHot")){
    topBranch.GetChild("nHot").GetData(nHot);
  }
   if (topBranch.GetChild("nActive")){
    topBranch.GetChild("nActive").GetData(nActive);
  }
  if (topBranch.GetChild("nSigma")){
    topBranch.GetChild("nSigma").GetData(nSigma);
  }
  if (topBranch.GetChild("nPersist")){
    topBranch.GetChild("nSigma").GetData(nPersist);
  }
  if (topBranch.GetChild("Verbosity")){
    topBranch.GetChild("Verbosity").GetData(fVerbosityLevel);
  }

  if (!topBranch.GetChild("S") && !topBranch.GetChild("N"))
    INFO("Using Default values for TriggerSPB2CSM");
  if (topBranch.GetChild("signalOnly")){
    topBranch.GetChild("signalOnly").GetData(fSignalOnly);
  }
  nSigma=(nSigma-4.0)*0.25 + 4.0;
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
  
  triggerState=0;
  int sigTotal=0;
  for (int ipdm=0; ipdm<3; ipdm++){ //Do everythin on a per-PDM basis
    Clear2();
    //Sums value in each cell
    for (int ipmt=0; ipmt<36; ipmt++){
      for (int ipixx=0; ipixx<8; ipixx++){
	for (int ipixy=0; ipixy<8; ipixy++){
	  icell = (ipixx/2) + 4*(ipixy/2);
	  for (int igtu=0;igtu<128;igtu++){
	    sigTotal+= ipheSig[ipdm][ipmt][ipixx][ipixy][igtu];
	    sumCells[ipdm][ipmt][icell] +=iphe[ipdm][ipmt][ipixx][ipixy][igtu];
	    if ((fSignalOnly && ipheSig[ipdm][ipmt][ipixx][ipixy][igtu]>0)||!(fSignalOnly)){
	      //if(ipheSig[ipdm][ipmt][ipixx][ipixy][igtu]>0)
	      valCells[ipdm][ipmt][icell][igtu] +=iphe[ipdm][ipmt][ipixx][ipixy][igtu];
	    }
	    //else{
	    //valCells[ipdm][ipmt][icell][igtu] += iphe[ipdm][ipmt][ipixx][ipixy][igtu];
	      //}
	  }
	}
      }
    }

    //Decide which cells are hot 
    for (int ipmt=0; ipmt<36; ipmt++){
      for (icell=0;icell<16;icell++){
	int thresh =int(float(sumCells[ipdm][ipmt][icell])/128.0+ nSigma*sqrt(sumCells[ipdm][ipmt][icell]/128.0));
	for (int igtu=0;igtu<128;igtu++){
	  if(valCells[ipdm][ipmt][icell][igtu]>=thresh){
	    hotCells[ipdm][ipmt][icell][igtu] =1;
	  }
	}
      }
    }
    
    //Makes single 3D array of cell values
    for (int ipmt=0; ipmt<36; ipmt++){
      for(icell=0;icell<16;icell++){
	int xLoc= (ipmt/12)*8+(((ipmt%12)/2)%2)*4  + icell%4;
	int yLoc= pmtMapperY(ipmt%12)*4  + (4-(icell/4));
	for (int igtu=0;igtu<128;igtu++){
	  HotOrNot[xLoc][yLoc][igtu]=hotCells[ipdm][ipmt][icell][igtu];
	}
      }
    }
    
    //Trigger Logic 
    total=0;
    int gtuMin=128,gtuMax=0;
    for (int igtu=1;igtu<127;igtu++){ //Look through the event skipping first and last frames
      for (int iLocX=1;iLocX<23;iLocX++){// Look at whole camera (1 PDM) skipping 
	for (int iLocY=1;iLocY<23;iLocY++){ //top, bottom, rightmost, leftmost pixels
	  for (int ix=-1;ix<2;ix++){ //Look at the MacroPixels to  left and right 
	    for(int iy=-1;iy<2;iy++){// And look at the MacroPixels above and below 
	      for (int it=-1;it<2;it++){ //Look at the GTU before and after 
		if(HotOrNot[iLocX+ix][iLocY+iy][igtu+it] !=0) { // Check if the adjacent MacroPixel is "hot"
		  HotNeighborsCount[iLocX][iLocY][igtu]++;//Count the hot neighbors 
		}
	      }
	    }
	  } 
	  if  (HotNeighborsCount[iLocX][iLocY][igtu]>=nHot){ //Count how many active MacroPixels
	    total++;
	    if(igtu<gtuMin)
	      gtuMin=igtu;
	    if(igtu>gtuMax)
	      gtuMax=igtu;
	  }
	}
      }
    }    
    if (total >=nActive && (gtuMax-gtuMin)>=nPersist)
      triggerState=1;
  }
  if (triggerState==1){
      if (fVerbosityLevel>0)
	INFO("CSM_TRIGGER:\t1");
      event.SetTriggerState(1);
    }
  if (triggerState==0){
    if (fVerbosityLevel >0)
      INFO("CSM_TRIGGER:\t0");
    event.SetTriggerState(0);
  }
  cout <<sigTotal<<endl;
  return eSuccess;
}

VModule::ResultFlag 
TriggerSPB2CSM::Finish() 
{
  INFO("TriggerSPB2CSM");
  return eSuccess;
}
void TriggerSPB2CSM::Clear(){
  fill(&iphe[0][0][0][0][0], &iphe[0][0][0][0][0] + (3*36*8*8*128), 0);
  fill(&ipheSig[0][0][0][0][0], &ipheSig[0][0][0][0][0] + (3*36*8*8*128), 0);
  fill(&sumCells[0][0][0], &sumCells[0][0][0] + (3*36*16), 0);
  fill(&valCells[0][0][0][0], &valCells[0][0][0][0] + (3*36*16*128), 0);
  fill(&hotCells[0][0][0][0], &hotCells[0][0][0][0] + (3*36*16*128), 0);
  fill(&HotOrNot[0][0][0], &HotOrNot[0][0][0] + (24*24*128), 0);
  fill(&HotNeighborsCount[0][0][0], &HotNeighborsCount[0][0][0] + (24*24*128), 0);
}

  void TriggerSPB2CSM::Clear2(){
      fill(&HotOrNot[0][0][0], &HotOrNot[0][0][0] + (24*24*128), 0);
      fill(&HotNeighborsCount[0][0][0], &HotNeighborsCount[0][0][0] + (24*24*128), 0);
  }
  
void TriggerSPB2CSM::Input(evt::Event& event){
  
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
			iphe[ipdm-1][ipmt_global][irow][icol][igtu] = trace[igtu];
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
			ipheSig[ipdm-1][ipmt_global][irow][icol][igtu] = tracePE[igtu];
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

/*
   Offline mapping of pixels in each PMT, PMTs in each PDM, and EC in each PDM,
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

  //Need this to easily get a pmt number to y location on the grid 
int TriggerSPB2CSM::pmtMapperY(int pmt){
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

