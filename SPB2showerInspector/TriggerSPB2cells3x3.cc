/*
  Implementation of the cosmic ray trigger algorithm for EUSO-SPB2, version sith cells 3x3 pixels

  author Francesca Bisconti (trigger algorithm, version with 3x3 pixels cells inside MAPMTs, developed by Matteo Battisti, Mario Bertaina)

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

#include "TriggerSPB2cells3x3.h"

#include <utl/ErrorLogger.h>
#include <utl/Trace.h>
#include <evt/Event.h>
#include <evt/Header.h>
#include <fevt/FEvent.h>
#include <fevt/Eye.h>
#include <fevt/Telescope.h>
#include <fdet/FDetector.h>
#include <det/Detector.h>
#include <fwk/CentralConfig.h>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

using namespace fwk;
using namespace utl;
using namespace fevt;
using namespace det;
using namespace std;
using namespace TriggerSPB2cells3x3TG;

#define initial_thrsld 9999

TriggerSPB2cells3x3::TriggerSPB2cells3x3(){}

TriggerSPB2cells3x3::~TriggerSPB2cells3x3(){}

//#################################### INIT
VModule::ResultFlag
TriggerSPB2cells3x3::Init()
{
  INFO("Initiating TriggerSPB2cells3x3TG");        

  CentralConfig* cc = CentralConfig::GetInstance();
  
  Branch topBranch = cc->GetTopBranch("TriggerSPB2cells3x3TG");
  topBranch.GetChild("nPixelInCell").GetData(fNPixelInCell);
  topBranch.GetChild("nCellInPMT").GetData(fNCellInPMT);
  topBranch.GetChild("nGTUpersistence").GetData(fNGTUpersistence);
  topBranch.GetChild("nSigma").GetData(fNSigma);
  if (topBranch.GetChild("verbosityLevel")){
    topBranch.GetChild("verbosityLevel").GetData(fVerbosityLevel);
  }
  if (topBranch.GetChild("signalOnly")){
    topBranch.GetChild("signalOnly").GetData(fSignalOnly);
  }
  if (topBranch.GetChild("wantTriggerOutput")){
    topBranch.GetChild("wantTriggerOutput").GetData(fWantTriggerOutput);
  }
  
  cout <<"n pixels in cell = " <<fNPixelInCell <<", n cells in PMT = " <<fNCellInPMT <<", n GTU persistence = " <<fNGTUpersistence <<", n sigma = " <<fNSigma 
       <<", with trigger on signal only = " <<fSignalOnly <<" used by trigger algorithm" <<endl;

  ievent = 0;
  RunNum = 0;

  trigger=0;
  trigger_flag=0;
  
  sum_pixel[nPDM][nPMT][nPixX][nPixY]={0}; // sum counts of every pixel over 128 GTUs for 3 PDMs 
  
  buffer[nPDM][nPMT][nPixX][nPixY] = {0}; // buffer containing 3 PDMs at the time. Shape is 3 PDMs, 36 PMTs, 8x8 pixel
  boolean_buffer[nPDM][nPMT][nPixX][nPixY] = {0}; // 1 if a pixel is above threshold, 0 otherwise. Shape is 3 PDMs, 36 PMTs, 8x8 pixel

  pixel_OT_in_cell[nPDM][nPMT][nCell] = {0}; // number of pixels over thresholdd in each cell (between 0 and 9). Shape is 3 PDMs, 36 PMTs, 36 cells
  boolean_active_cell[fNGTUpersistence][nPDM][nPMT][nCell] = {0};  // 1 if a cell is active, 0 otherwise. A cell is active if there are more than 'fNPixelInCell' pixel above threshold. Holds in memory 'fNGTUpersistence' GTUs
 
  return eSuccess;
}//init

//#################################### RUN
VModule::ResultFlag
TriggerSPB2cells3x3::Run(evt::Event& event)
{
  INFO("Running TriggerSPB2cells3x3TG");
  gtuMin=128;
  gtuMax=0;
  gtuTrigMin=128;
  gtuTrigMax=0;
  for (int i =0;i<3;i++)
    pdmCounts[i]=0;
  RunNum = event.GetHeader().GetRunNumber();
  Input(event);

  if (ievent==0){
    for(int ipdm=0; ipdm<nPDM; ipdm++){
      for(int ipmt=0; ipmt<nPMT; ipmt++){
        for(int ix=0; ix<nPixX; ix++){
          for(int iy=0; iy<nPixY; iy++){
            threshold[ipdm][ipmt][ix][iy] = initial_thrsld; // initialize threshold values
	  }
        }
      }
    }
  }
  
  cout <<"Run: ievent=" <<ievent <<endl; // starts from ievent=0
 
  ////
  for (int igtu=0; igtu<nGTU; igtu++){
    for (int ipdm=0; ipdm<nPDM; ipdm++){
      for (int ipmt=0; ipmt<nPMT; ipmt++){
        for (int ipixx=0; ipixx<nPixX; ipixx++){
          for (int ipixy=0; ipixy<nPixY; ipixy++){
            sum_pixel[ipdm][ipmt][ipixx][ipixy] += iphe[igtu][ipdm][ipmt][ipixx][ipixy]; // sum pixel counts /pixel /PMT /PDM over 128 GTUs 
            if (fSignalOnly){
              if (ipheSig[igtu][ipdm][ipmt][ipixx][ipixy]!=0)
                buffer[ipdm][ipmt][ipixx][ipixy] = iphe[igtu][ipdm][ipmt][ipixx][ipixy]; // fill the buffer with  pixel counts /pixel /PMT /PDM /GTU 
              else
                buffer[ipdm][ipmt][ipixx][ipixy] = 0;
            }
            else{
              buffer[ipdm][ipmt][ipixx][ipixy] = iphe[igtu][ipdm][ipmt][ipixx][ipixy];
            }
            if (fVerbosityLevel > 0) {
              cout <<"ievent=" <<ievent <<", igtu=" <<igtu <<" - gtu_absolute=" <<ievent*nGTU+igtu <<", ipdm=" <<ipdm <<", ipmt=" <<ipmt <<", ipixx=" <<ipixx <<", ipixy=" <<ipixy <<", iphe=" <<iphe[igtu][ipdm][ipmt][ipixx][ipixy] <<endl;
            }
          }
        }
      }
    }
    // START LOOPED ANALYSIS every GTU
    ProcessBuffer(buffer, threshold, boolean_buffer); 
    DefineAndSumCells (boolean_buffer, pixel_OT_in_cell);
    WriteBooleanActiveCell (boolean_active_cell, pixel_OT_in_cell, fNGTUpersistence, igtu);
    AnalyzeBooleanActiveCell (boolean_active_cell, trigger_flag, fNGTUpersistence, ievent, igtu, outString);
  } // close loop over gtus

  // calculate thresholds for each event (every 128 GTUs)
  for (int ipdm=0; ipdm<nPDM; ipdm++){
    for (int ipmt=0; ipmt<nPMT; ipmt++){
      for (int ipixx=0; ipixx<nPixX; ipixx++){
        for (int ipixy=0; ipixy<nPixY; ipixy++){
	  
          threshold[ipdm][ipmt][ipixx][ipixy] = trunc((sum_pixel[ipdm][ipmt][ipixx][ipixy]/float(nGTU)) + fNSigma*(sqrt(sum_pixel[ipdm][ipmt][ipixx][ipixy]/float(nGTU))));
	  
	  if (fVerbosityLevel > 0) {
            cout <<"ievent=" <<ievent <<" - ipdm=" <<ipdm <<", ipmt=" <<ipmt <<", ipixx=" <<ipixx <<", ipixy=" <<ipixy <<" sum_pixel=" <<sum_pixel[ipdm][ipmt][ipixx][ipixy] <<" - threshold for next event="<<threshold[ipdm][ipmt][ipixx][ipixy] <<endl;
          }
	}
      }
    }
  }// it was used to update the thresholds every 128 GTUs. Now we keep the same threshold for 0.5 seconds, so it is not needed to update it
      
  //reinitialize to 0
  for (int ipdm=0; ipdm<nPDM; ipdm++){
    for (int ipmt=0; ipmt<nPMT; ipmt++){
      for (int ipixx=0; ipixx<nPixX; ipixx++){ 
        for (int ipixy=0; ipixy<nPixY; ipixy++){ 
          sum_pixel[ipdm][ipmt][ipixx][ipixy]= 0;
        }
      }
    }
  }// the variable sum_pixel is no longeer used, no need to reinitialization

  // set trigger result in event
  cout <<"Trigger result for event " <<ievent <<": ";
  if (trigger_flag > 0){
    cout <<"1" <<endl;
    event.SetTriggerState(1);
    trigger_flag=0;
    trigger++;
  }
  else {
    cout <<"0" <<endl;
    event.SetTriggerState(0);

    // text file where to write details on triggers
    if (fWantTriggerOutput) {
      outString.open (outFile.c_str(), ios::app);
      outString <<"event " <<ievent <<": "  <<trigger_flag <<" trigger" <<endl;
      outString.close();
    }
  }

  // reset the iphe to 0
  fill(&iphe[0][0][0][0][0], &iphe[0][0][0][0][0] + (nGTU*nPDM*nPMT*nPixX*nPixY), 0);
  // reset the ipheSig to 0
  fill(&ipheSig[0][0][0][0][0], &ipheSig[0][0][0][0][0] + (nGTU*nPDM*nPMT*nPixX*nPixY), 0);
  return eSuccess;
} // Run - loop over 128 GTUs

//#################################### FINISH
VModule::ResultFlag
TriggerSPB2cells3x3::Finish()
{ 
  INFO("Finish TriggerSPB2cells3x3TG");
  
  return eSuccess;
}

//#################################### FUNCTIONS

// analyze the GTU in the buffer. It contains the number of pixel over threshold in every PMT
void TriggerSPB2cells3x3::ProcessBuffer( int buffer[nPDM][nPMT][nPixX][nPixY], int threshold[nPDM][nPMT][nPixX][nPixY], int boolean_buffer[nPDM][nPMT][nPixX][nPixY]) {
  for (int ipdm=0; ipdm<nPDM; ipdm++){  //PDMs
    for (int ipmt=0; ipmt<nPMT; ipmt++){ //PMTs
      for (int ix=0; ix<nPixX; ix++){ //pixels
        for (int iy=0; iy<nPixY; iy++){
          if (buffer[ipdm][ipmt][ix][iy] > threshold[ipdm][ipmt][ix][iy] ){  // strictly >, and not >= because threshold was truncated
            boolean_buffer[ipdm][ipmt][ix][iy]=1;
          } 
          else {
            boolean_buffer[ipdm][ipmt][ix][iy]=0;
          }
        }
      }//close pixels
    }// close PMTs
  }// close PDMs
}// close function


// for each PMTs it defines the 36 cells and counts how many pixels over threshold there are in each cell (between 0 and 9) 
void TriggerSPB2cells3x3::DefineAndSumCells(int boolean_buffer[nPDM][nPMT][nPixX][nPixY], int pixel_OT_in_cell[nPDM][nPMT][nCell]) { 
  int icell;
  for (int ipdm=0; ipdm<nPDM; ipdm++){ // 3 PDMs
    for (int ipmt=0; ipmt<nPMT; ipmt++){ // PMTs
      icell=0;
      for (int ix=1; ix<7; ix++){ // center of a cell in a PMT
        for (int iy=1; iy<7; iy++){
          for (int iix=ix-1; iix<ix+2; iix++){ // define cells 3x3 pixels , centered at ix,iy
            for (int iiy=iy-1; iiy<iy+2; iiy++){
              pixel_OT_in_cell[ipdm][ipmt][icell] += boolean_buffer[ipdm][ipmt][iix][iiy];
	    }
          }
	  icell++;
        }
      }
    }
  }
}


// moves the rows of boolean_active_cell and fill the last row
void TriggerSPB2cells3x3::WriteBooleanActiveCell(int boolean_active_cell[][nPDM][nPMT][nCell], int pixel_OT_in_cell[nPDM][nPMT][nCell], int fNGTUpersistence, int gtu) {
  for (int igtu=fNGTUpersistence-1; igtu>0; igtu--){ //n_GTU
    for (int ipdm=0; ipdm<nPDM; ipdm++){ // 3 PDMs
      for (int ipmt=0; ipmt<nPMT; ipmt++){ // PMTs
        for (int icell=0; icell<nCell; icell++){ // cells
          boolean_active_cell[igtu][ipdm][ipmt][icell]=boolean_active_cell[igtu-1][ipdm][ipmt][icell]; 
        } 
      } 
    } 
  } 

  for (int ipdm=0; ipdm<nPDM; ipdm++){ // 3 PDMs
    for (int ipmt=0; ipmt<nPMT; ipmt++){  // 36 PMTs
      for ( int icell=0; icell<nCell; icell++){ // cells
        if (pixel_OT_in_cell [ipdm][ipmt][icell] >= fNPixelInCell ) {
          boolean_active_cell [0][ipdm][ipmt][icell] = 1;
	}
        else {
          boolean_active_cell [0][ipdm][ipmt][icell] = 0;
        }
        pixel_OT_in_cell[ipdm][ipmt][icell]=0;
      }
    }
  }
}


// checks if the trigger condition is satisfied
void TriggerSPB2cells3x3::AnalyzeBooleanActiveCell (int boolean_active_cell[][nPDM][nPMT][nCell], int &trigger_flag, int fNGTUpersistence, int event, int gtu, ofstream &outString) {

  int sum_cell[nPDM][nCell] = {0}; // variable where to store the number of active cells over fNGTUpersistence GTUs in one PMT
  for (int igtu=0; igtu<fNGTUpersistence; igtu++){ // GTUs
    for (int ipdm=0; ipdm<nPDM; ipdm++){ // 3 PDMs
      for (int ipmt=0; ipmt<nPMT; ipmt++){ // PMTs
        for ( int icell=0; icell<nCell; icell++){ // cells
          sum_cell[ipdm][ipmt]+=boolean_active_cell[igtu][ipdm][ipmt][icell];
        } 
      } 
    }
  }

  for (int ipdm=0; ipdm<nPDM; ipdm++){ // 3 PDMs
    for (int ipmt=0; ipmt<nPMT; ipmt++){ // PMTs
      if (sum_cell[ipdm][ipmt] >= fNCellInPMT ) { // if TRUE, the PMT has enough cells active, therefore there is a TRIGGER 
        trigger_flag++;
        cout <<"event " <<event <<": trigger at gtu=" <<gtu <<", gtu_abs=" <<event*nGTU+gtu <<", pdm=" <<ipdm <<", pmt=" <<ipmt <<", active cells=" <<sum_cell[ipdm][ipmt] <<endl;
	
        if (fWantTriggerOutput){
          outString.open (outFile.c_str(), ios::app); 
          /*outString*/cout <<"QWERTYevent " <<event <<": trigger at gtu " <<gtu <<", gtu_abs=" <<event*nGTU+gtu <<", pdm " <<ipdm <<", pmt " <<ipmt <<", active cells=" <<sum_cell[ipdm][ipmt] <<" GTUMin "<<gtuMin<<" GTUMax "<<gtuMax<<" " <<pdmCounts[0]<<" " <<pdmCounts[1]<<" "<<pdmCounts[2]<<endl;
          outString.close();
        }
      }
    }
  }
}


// data access and loading into container
void TriggerSPB2cells3x3::Input(evt::Event& event){
  
  ievent = event.GetHeader().GetEventNumber();
  
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
                      
                      for(int igtu=0;igtu<nGTU; igtu++){
			irow = (ipix-1)/8;// irow per PMT
			icol = ((ipix-1)%8);// icol per PMT
			int ipmt_global = ((iec-1)) * 4 + (ipmt-1);
			iphe[igtu][ipdm-1][ipmt_global][irow][icol] = trace[igtu];
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
		      for(int igtu=0;igtu<nGTU; igtu++){
			irow = (ipix-1)/8;// irow per PMT
			icol = ((ipix-1)%8);// icol per PMT
			int ipmt_global = ((iec-1)) * 4 + (ipmt-1);
			ipheSig[igtu][ipdm-1][ipmt_global][irow][icol] = tracePE[igtu];
			//cout <<"trace=" <<trace[igtu] <<", iphe=" <<iphe[igtu][ipdm-1][ipmt_global][irow][icol] <<", igtu=" <<igtu <<", ipdm=" <<ipdm <<", ipmt_global=" <<ipmt_global <<", irow=" <<irow <<", icol=" <<icol  <<endl; 
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
			if (tracePE[igtu]!=0){
			  pdmCounts[ipdm-1]++;
			  if(igtu>gtuMax)
			    gtuMax=igtu;
			  if (igtu<gtuMin)
			    gtuMin=igtu;
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
