#include "BackgroundSubtractor.h"
#include <utl/Trace.h>

using namespace fwk;
using namespace utl;
using namespace fevt;
using namespace det;
using namespace std;

namespace BGSub {


BackgroundSubtractorCSM::BackgroundSubtractorCSM(){}

BackgroundSubtractorCSM::~BackgroundSubtractorCSM(){}

VModule::ResultFlag 
BackgroundSubtractorCSM::Init()
{

  INFO("Running BackgroundSubtractor");

  CentralConfig* cc = CentralConfig::GetInstance();
  Branch topBranch = cc->GetTopBranch("BackgroundSubtractorCSM");

  //Default Values 
  sCell=2.67;
  nHot=5;

  //Optional xml parameters
  if (topBranch.GetChild("N")){
    topBranch.GetChild("N").GetData(nHot);
  }
  if (topBranch.GetChild("S")){
    topBranch.GetChild("S").GetData(sCell);
  }

  if (!topBranch.GetChild("S") && !topBranch.GetChild("N"))
    INFO("Using Default valus for BackgroundSubractor");
  
  return eSuccess;
}

VModule::ResultFlag 
BackgroundSubtractorCSM::Run(evt::Event& event)
{
  INFO("Running BackgroundSubtractor");
  int sumiphe[3][36][8][8];
  int sumCells[3][36][16];
  int16_t iphe[3][36][8][8][128];
  int16_t valCells[3][36][16][128];
  int16_t hotCells[3][36][16][128];
  int16_t hotCells1[3][36][16][128];
  int16_t cellLoc[72][24][128];
  utl::TraceI TraceBS[3][36][8][8];
  int16_t cellLoc1[72][24][128];
  //Initialize arrays to zero 
  fill(&iphe[0][0][0][0][0], &iphe[0][0][0][0][0] + (3*36*8*8*128), 0);
  fill(&sumiphe[0][0][0][0], &sumiphe[0][0][0][0] + (3*36*8*8), 0);
  fill(&sumCells[0][0][0], &sumCells[0][0][0] + (3*36*16), 0);
  fill(&valCells[0][0][0][0], &valCells[0][0][0][0] + (3*36*16*128), 0);
  fill(&hotCells[0][0][0][0], &hotCells[0][0][0][0] + (3*36*16*128), 0);
  fill(&hotCells1[0][0][0][0], &hotCells1[0][0][0][0] + (3*36*16*128), 0);
  fill(&cellLoc[0][0][0], &cellLoc[0][0][0] + (72*24*128), 0);
  fill(&cellLoc1[0][0][0], &cellLoc1[0][0][0] + (72*24*128), 0);

  //Read event in 
  Input(event,iphe);


  //Sums value in each pixel and cell
  for (int ipdm=0; ipdm<3; ipdm++){
    for (int ipmt=0; ipmt<36; ipmt++){
      for (int ipixx=0; ipixx<8; ipixx++){
	for (int ipixy=0; ipixy<8; ipixy++){
	  icell = (ipixx/2) + 4*(ipixy/2);
	  for (int igtu=0;igtu<128;igtu++){
	    if (iphe[ipdm][ipmt][ipixx][ipixy][igtu]<4){
	      sumiphe[ipdm][ipmt][ipixx][ipixy] +=iphe[ipdm][ipmt][ipixx][ipixy][igtu]; 
	      sumCells[ipdm][ipmt][icell] +=iphe[ipdm][ipmt][ipixx][ipixy][igtu];
	    }
	    else{
	      sumiphe[ipdm][ipmt][ipixx][ipixy] +=3; 
	      sumCells[ipdm][ipmt][icell] +=3;
	    }
	    valCells[ipdm][ipmt][icell][igtu] +=iphe[ipdm][ipmt][ipixx][ipixy][igtu];
	  }
	}
      }
    }
  }

  //Decide which cells are hot 
  for (int ipdm=0; ipdm<3; ipdm++){
    for (int ipmt=0; ipmt<36; ipmt++){
      for (icell=0;icell<16;icell++){
	for (int igtu=0;igtu<128;igtu++){
	  if(int(128.0/sCell)*valCells[ipdm][ipmt][icell][igtu]>=sumCells[ipdm][ipmt][icell]){
	    hotCells[ipdm][ipmt][icell][igtu] =1;
	    hotCells1[ipdm][ipmt][icell][igtu] =1;
	  }
	}
      }
    }
  }

  //Makes single 3D array of cell values
  for (int ipdm=0; ipdm<3; ipdm++){
    for (int ipmt=0; ipmt<36; ipmt++){
      for(icell=0;icell<16;icell++){
	int xLoc=24*ipdm + (ipmt/12)*8+(((ipmt%12)/2)%2)*4  + icell%4;
	int yLoc= pmtMapperY(ipmt%12)*4  + (4-(icell/4));
	for (int igtu=0;igtu<128;igtu++){
	  cellLoc[xLoc][yLoc][igtu]=hotCells[ipdm][ipmt][icell][igtu];
	  cellLoc1[xLoc][yLoc][igtu]=hotCells[ipdm][ipmt][icell][igtu];
	}
      }
    }
  }

  //Decide how many nearby cells are hot 
  for (int igtu=0;igtu<128;igtu++){
    for (int iLocx=1;iLocx<71;iLocx++){
      if (cellLoc1[iLocx][1][igtu]!=0)
	cellLoc[iLocx][0][igtu]++;
      if (cellLoc1[iLocx][-2][igtu]!=0)
	cellLoc[iLocx][-1][igtu]++;
      for (int iLocy=1;iLocy<23;iLocy++){
	if(cellLoc1[iLocx][iLocy+1][igtu]!=0)
	  cellLoc[iLocx][iLocy][igtu]++;
	if(cellLoc1[iLocx+1][iLocy+1][igtu]!=0)
	  cellLoc[iLocx][iLocy][igtu]++;
	if(cellLoc1[iLocx-1][iLocy+1][igtu]!=0)
	  cellLoc[iLocx][iLocy][igtu]++;
	if(cellLoc1[iLocx+1][iLocy][igtu]!=0)
	  cellLoc[iLocx][iLocy][igtu]++;
	if(cellLoc1[iLocx-1][iLocy][igtu]!=0)
	  cellLoc[iLocx][iLocy][igtu]++;
	if(cellLoc1[iLocx][iLocy-1][igtu]!=0)
	  cellLoc[iLocx][iLocy][igtu]++;
	if(cellLoc1[iLocx+1][iLocy-1][igtu]!=0)
	  cellLoc[iLocx][iLocy][igtu]++;
	if(cellLoc1[iLocx-1][iLocy-1][igtu]!=0)
	  cellLoc[iLocx][iLocy][igtu]++;
      }
    }
    //Check the border cells
    for (int iLocy=0;iLocy<24;iLocy++){
      if (cellLoc1[1][iLocy][igtu]!=0)
	cellLoc[0][iLocy][igtu]++;
      if (cellLoc1[-2][iLocy][igtu]!=0)
	cellLoc[-1][iLocy][igtu]++;
    }
  }
  for (int ipdm=0; ipdm<3; ipdm++){
    for (int ipmt=0; ipmt<36; ipmt++){
      for(icell=0;icell<16;icell++){
	int xLoc=24*ipdm + (ipmt/12)*8+(((ipmt%12)/2)%2)*4  + icell%4;
	int yLoc= pmtMapperY(ipmt%12)*4  + (4-(icell/4));
	for (int igtu=0;igtu<128;igtu++){
	  hotCells[ipdm][ipmt][icell][igtu]=cellLoc[xLoc][yLoc][igtu];
	}
      }
    }
  }

  //Check if the GTU before or after is hot 
  for (int ipdm=0; ipdm<3; ipdm++){
    for (int ipmt=0; ipmt<36; ipmt++){
      for (int ipixx=0; ipixx<8; ipixx++){
	for (int ipixy=0; ipixy<8; ipixy++){
	  icell = (ipixx/2) + 4*(ipixy/2);
	  if (hotCells1[ipdm][ipmt][icell][1]!=0)
	    hotCells[ipdm][ipmt][icell][0]++;
	  if (hotCells1[ipdm][ipmt][icell][-2]!=0)
	    hotCells[ipdm][ipmt][icell][-1]++;
	  for (int igtu=1;igtu<127;igtu++){
	    if (hotCells1[ipdm][ipmt][icell][igtu+1]!=0)
	      hotCells[ipdm][ipmt][icell][igtu]++;
	    if(hotCells1[ipdm][ipmt][icell][igtu-1]!=0)
	      hotCells[ipdm][ipmt][icell][igtu]++;
	  }  
	}
      }
    }
  }

  //Check if pixels are to remain active, if not set to zero
  for (int ipdm=0; ipdm<3; ipdm++){
    for (int ipmt=0; ipmt<36; ipmt++){
      for (int ipixx=0; ipixx<8; ipixx++){
	for (int ipixy=0; ipixy<8; ipixy++){
	  icell = (ipixx/2) + 4*(ipixy/2);
	  for(int igtu=0;igtu<128;igtu++){
	    if(hotCells[ipdm][ipmt][icell][igtu]<nHot)
	      iphe[ipdm][ipmt][ipixx][ipixy][igtu]=0;
	    if (iphe[ipdm][ipmt][ipixx][ipixy][igtu]*128 <= sumiphe[ipdm][ipmt][ipixx][ipixy])
	      iphe[ipdm][ipmt][ipixx][ipixy][igtu]=0;
	  }
	}
      }
    }
  }

  //Get it in a nice Format
  for (int ipdm=0; ipdm<3; ipdm++){
    for (int ipmt=0; ipmt<36; ipmt++){
      for (int ipixx=0; ipixx<8; ipixx++){
	for (int ipixy=0; ipixy<8; ipixy++){
	  TraceBS[ipdm][ipmt][ipixx][ipixy].Adopt(iphe[ipdm][ipmt][ipixx][ipixy],128);
	}
      }
    }
  }

  //Output
  Output(event,TraceBS);
  return eSuccess;
}

VModule::ResultFlag 
BackgroundSubtractorCSM::Finish() 
{
  INFO("Finish BackgroundSubtractor");
  return eSuccess;
}

  
  void BackgroundSubtractorCSM::Input(evt::Event& event,int16_t iphe[3][36][8][8][128]){
  
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
	      }
	    }
	  }
	}
      }
    }
  }
}

void BackgroundSubtractorCSM::Output(evt::Event& event,  TraceI TraceBS[3][36][8][8]){
  
  
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
		  int ipmt_global = ((iec-1)) * 4 + (ipmt-1);
		  irow = (ipix-1)/8;// irow per PMT
		  icol = ((ipix-1)%8);// icol per PMT
		  pix->GetRecData().MakeFADCTrace(TraceBS[ipdm-1][ipmt_global][irow][icol], FdConstants::eBackgroundSubtracted);
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
int BackgroundSubtractorCSM::pmtMapperY(int pmt){
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

