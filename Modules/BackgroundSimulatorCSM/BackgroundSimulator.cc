#include "BackgroundSimulator.h"

#include <fevt/FEvent.h>
#include <fevt/Eye.h>
#include <fevt/Telescope.h>

#include <det/Detector.h>

#include <fdet/Telescope.h>
#include <fdet/FDetector.h>
#include <fdet/Eye.h>
#include <fevt/TelescopeTriggerData.h>

#include <fwk/CentralConfig.h>
#include <utl/CoordinateSystemPtr.h>
#include <utl/RandomEngine.h>
#include <fwk/RandomEngineRegistry.h>
#include <CLHEP/Random/Randomize.h>

#include <TMath.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <iostream>


using namespace evt;
using namespace fwk;
using namespace utl;
using namespace fevt;
using namespace det;
using namespace std;
using namespace BackgroundSimulatorCSM;


BackgroundSimulator::BackgroundSimulator() :
  fReadFromFile(false), ///Read in average background per pixel from file
  fPath(""), ///Path for file for option a 
  fIsSPB1(false), ///flag for SPB1 simulations
  fTiltDependent(false), ///Flag for tilt dependent background
  fNonUniform(false),
  fAverageBlobSize(5.0),
  fAverageBlobBrightness(20.0),
  fBlobFrac(0.0),
  numPDMs(0),
  fLengthBloob(1),
  fVarriedModifier(1.0),
  fVarriedBackground(false)
{
}

BackgroundSimulator::~BackgroundSimulator()
{ }

VModule::ResultFlag
BackgroundSimulator::Init()
{
  ///read in xml config
  Branch topB, branch;
  topB = fwk::CentralConfig::GetInstance()->GetTopBranch("BackgroundSimulatorCSM");
  if (!topB) {
    ERROR("No XML configuration found for BackgroundSimulatorCSM!");
    return eFailure;
  }
 
  branch = topB.GetChild("ReadFromFile");
  if(branch) branch.GetData(fReadFromFile);

  branch = topB.GetChild("Path");
  if(branch) branch.GetData(fPath);
  
  branch = topB.GetChild("IsSPB1");
  if(branch) branch.GetData(fIsSPB1);
  ///Begin data read for tilted background 
  branch = topB.GetChild("TiltDependent");
  if (branch) branch.GetData(fTiltDependent);

  branch = topB.GetChild("BaseValue");
  if (branch) branch.GetData(fBaseValue);
  
  branch = topB.GetChild("NonUniform");
  if (branch) branch.GetData(fNonUniform);
  branch = topB.GetChild("MakeBlobs");
  if (branch)branch.GetData(fBlobFrac);

  
  branch = topB.GetChild("AverageBlobSize");
  if (branch) branch.GetData(fAverageBlobSize);
  branch = topB.GetChild("AverageBlobBrightness");
  if (branch) branch.GetData(fAverageBlobBrightness);
  branch = topB.GetChild("LengthBloob");
  if (branch) branch.GetData(fLengthBloob);
  
  ///get tilt angle from dtd file, same eveywhere! Tilt needed for tilt dependent BG 
  Detector& detector = Detector::GetInstance();
  detector.Update(GetCurrentSystemTime());
  const fdet::Telescope& detTel = detector.GetFDetector().GetEye(1).GetTelescope(1);
  string pointingId = detTel.GetTelescopePointingId();
  fdet::Telescope::PointingAngleSet paE = detTel.GetTelescopePointingElevation();
  for (fdet::Telescope::PointingAngleSet::iterator pIt = paE.begin() ; pIt != paE.end() ; ++pIt) {
      if (pIt->first == pointingId) {
          fTilt = pIt->second;
      }
  }
  
  int pxRow = 0;
  int pxCol = 0;
  int pxId = 0;
  double pixTilt;
  double vals[48]; ///Empty array that will hold the value of background at that row. Hardcoded to current PDM size 
  fTilt = 90.0 - fTilt / deg; ///change tilt to degrees that make more sense. Zero is straight down, 90 is horizontal. Not the same as used in the telescope simulator
  for (int i =0; i<48; i++){
    pixTilt = (5.875+fTilt-(i/4.0)); ///Calculates the tilt on a per pixel basis, assumes that center is at tilt angle. FoV and pixel size harcoded. 
    if (pixTilt<90.0){
      vals[i] = -0.999896 + 4.02946*pixTilt*degree; ///Caluclates the linear Portion of tilt dependent BG (i.e. when the pixel looks above the horizontal. Fits done in radians, hence conversion to degrees
    }
    else{
      vals[i] = 1.00024 + 0.075005*TMath::Exp(-4.77*pixTilt*degree +11.52 ); ///Exponetial portion. Looking below orizontal. Normalized so that the value looking straight down is the base value. 
    }
   
  }
  
  double bgValue = fBaseValue;
  
  ///load average background value of single pixel from file, pixelId fills corresponding BG value 
  if(fReadFromFile){
    	ifstream myfile (fPath.c_str());
  	if (myfile.is_open()){
        	INFO("Using bg file!!");
        	while(!myfile.eof( )){
                        if(fIsSPB1){
           			myfile >> pxRow >> pxCol >> bgValue;
           			AvgBackroundXY[std::make_pair(pxRow,pxCol)]= bgValue;
                        }else{
                                myfile >> pxId >> bgValue;
     				AvgBackround[pxId]= bgValue;
                        }
           	}
       	}else{
        	ERROR("Backgroundfile could not be opened. It does not exist or is corrupted.");
        } 
   }
  else if(fTiltDependent){
    for (int iik =0; iik <2304;iik++){
      AvgBackround[iik] = (vals[iik%48]*fBaseValue) ; ///Multiplies the supplied avgBackground by the modyfing  term due to tilt correction, populates pixel matrix with all rows having the same average value 
    }
  } ///This AvgBackground array is then read in the same way as if it was populated from a file
  else{
    INFO("Using Constant mean backgroundValue!!"); ///If constant background, pixel matrix does not need to be filled. 
   }
   return eSuccess;
}


VModule::ResultFlag
BackgroundSimulator::Run(Event& event)
{
  INFO("Running BackgroundSimulatorCSM ...");
  fEvent = &event;
  Detector& detector = Detector::GetInstance();
  const fdet::FDetector& detFD = detector.GetFDetector();
  fEyeId = detFD.GetLastEyeId();
  SetUpDataStructure(event);
  SetBackground(event);
  if (numPDMs==0)
    numPDMs=GetPDMNum(event);
  MakeBlobs(event);
  return eSuccess;
}


VModule::ResultFlag
BackgroundSimulator::Finish()
{
  INFO("Finish BackgroundSimulatorCSM ... ");
  return eSuccess;
}



void
BackgroundSimulator::SetUpDataStructure(Event& event)
{
  FEvent& fdEvent = event.GetFEvent();
  ///Creates data structure appropriate for the event 
  for (unsigned iEyeId = 1; iEyeId <= fEyeId; ++iEyeId) {
    Detector& detector = Detector::GetInstance();
    const fdet::FDetector& detFD = detector.GetFDetector();
    fTelId = detFD.GetEye(iEyeId).GetLastTelescopeId();

    for (unsigned iTelId = 1; iTelId <= fTelId ; ++iTelId) {
      Eye& eye = fdEvent.GetEye(iEyeId, ComponentSelector::eExists);

      ///I think the idea is to never have more than 1 CLKB so this hardwired number is okay (for now)
      for (unsigned iCLKId = 1; iCLKId <= 1; ++iCLKId) {
        Telescope& telescope = eye.GetTelescope(iTelId, ComponentSelector::eExists);
        if (!telescope.HasCLK(iCLKId, (ComponentSelector::eExists))){
        telescope.MakeCLK(iCLKId);}
      }
      ///should be read from file, no Get function available
      for (unsigned iCCBId = 1; iCCBId <= 1; ++iCCBId) {
        Telescope& telescope = eye.GetTelescope(iTelId, ComponentSelector::eExists);

	    const fdet::Telescope& detTel = detFD.GetTelescope(telescope);
    	const fdet::Camera& detCamera = detTel.GetCamera();
        binning = detCamera.GetFADCBinSize();  //GTUlength
        nTrace  = 128; //<- readout length has to be set up...

        if (!telescope.HasCCB(iCCBId, (ComponentSelector::eExists))){
        telescope.MakeCCB(iCCBId);
        }

        fPDMId = detFD.GetEye(iEyeId).GetTelescope(iTelId).GetLastPDMId();

        for (unsigned iPDMId = 1; iPDMId <= fPDMId; ++iPDMId) {
          CCB& ccb = telescope.GetCCB(iCCBId, ComponentSelector::eExists);
          if (!ccb.HasPDM(iPDMId, (ComponentSelector::eExists))){
          ccb.MakePDM(iPDMId);}
          ///following numbers are universal to PDM, but maybe still should be read from file? (JE)
          for (unsigned iECId = 1; iECId <= 9; ++iECId) {
            PDM& pdm = ccb.GetPDM(iPDMId, ComponentSelector::eExists);
            if (!pdm.HasEC(iECId, (ComponentSelector::eExists))){
            pdm.MakeEC(iECId);}

            for (unsigned iPMTId = 1; iPMTId <= 4; ++iPMTId) {
              EC& ec = pdm.GetEC(iECId, ComponentSelector::eExists);
              if (!ec.HasPMT(iPMTId, (ComponentSelector::eExists))){
              ec.MakePMT(iPMTId);}

              for (unsigned iPixelId = 1; iPixelId <= 64; ++iPixelId) {
                PMT& pmt = ec.GetPMT(iPMTId, ComponentSelector::eExists);
                if (!pmt.HasPixel(iPixelId, (ComponentSelector::eExists))){
                pmt.MakePixel(iPixelId);
                }

              }
            }
          }
        }
      }
    }
  }
}



void BackgroundSimulator::SetBackground(Event& event){
  FEvent& fdEvent = event.GetFEvent();
  static utl::RandomEngine& RandomEngine =
             fwk::RandomEngineRegistry::GetInstance().Get(fwk::RandomEngineRegistry::eDetector);
  ///Loops to loop over eyes, telescopes, PDMs, ECs, pmts, pixels and GTUs.

  if (fVarriedBackground)
    fVarriedModifier = CLHEP::RandPoisson::shoot(&RandomEngine.GetEngine(), fBaseValue)/fBaseValue;
  for (fevt::FEvent::EyeIterator eye = fdEvent.EyesBegin(ComponentSelector::eExists);
       eye != fdEvent.EyesEnd(ComponentSelector::eHasData); ++eye) {
    for (fevt::Eye::TelescopeIterator tel = eye->TelescopesBegin(ComponentSelector::eExists);
         tel != eye->TelescopesEnd(ComponentSelector::eExists); ++tel) {

      for (fevt::Telescope::CCBIterator ccb = tel->CCBsBegin(ComponentSelector::eExists);
           ccb != tel->CCBsEnd(ComponentSelector::eExists); ++ccb) {
        // add something here to fill sim info at CCB level (trigger data?)

        for (fevt::CCB::PDMIterator pdm = ccb->PDMsBegin(ComponentSelector::eExists);
             pdm != ccb->PDMsEnd(ComponentSelector::eExists); ++pdm) {
          // add something here to fill info at PDM level

          for (fevt::PDM::ECIterator ec = pdm->ECsBegin(ComponentSelector::eExists);
               ec != pdm->ECsEnd(ComponentSelector::eExists); ++ec) {
	           int iec = ec->GetId();

		   for (fevt::EC::PMTIterator pmt = ec->PMTsBegin(ComponentSelector::eExists);
			pmt != ec->PMTsEnd(ComponentSelector::eExists); ++pmt) {
		     int ipmt = pmt->GetId();
		     for (fevt::PMT::PixelIterator pix = pmt->PixelsBegin(ComponentSelector::eExists);
			  pix != pmt->PixelsEnd(ComponentSelector::eExists); ++pix) {
		       int ipix = pix->GetId();
	       
		       int irow = ((iec-1)/3)*16 + ((ipmt-1)/2)*8 + ((ipix-1)/8);
		       int icol = ((iec-1)%3)*16 + ((ipmt-1)%2)*8 + (ipix-1)%8;
		       int ipix2= 48*irow + icol;
		       // create photon trace for background
		       if (!pix->HasSimData()) {
			 pix->MakeSimData();
			 pix->GetSimData().MakeFADCTrace(nTrace, binning, FdConstants::eTotal);
		       }
		       TraceI& trace = pix->GetSimData().GetFADCTrace(FdConstants::eTotal);
		       //add average Background per pixel vary it by random
		       int igstart = trace.GetStart();
		       int igstop = trace.GetStop();
		       double multiplier=0.0;
		       if (irow%8==0 || irow%8==7)
			 multiplier = CLHEP::RandFlat::shoot(&RandomEngine.GetEngine(), 1.2,1.5);
		       else
			 multiplier = CLHEP::RandFlat::shoot(&RandomEngine.GetEngine(), 0.95,1.05);
		       for(int igtu=igstart;igtu<igstop; igtu++){
			 double ToAddBg = 0;
			 if(fReadFromFile || fTiltDependent){
			   if(fIsSPB1){
			     ///set background to 0 if average value is very small, otherwise sets it to Poission dist value of average
			     ///row and column are flipped for SPB1 due to rotation in real data compared to simulated signal by 90°
			     if(AvgBackroundXY[std::make_pair(irow,icol)] < 0.01) ToAddBg = AvgBackroundXY[std::make_pair(icol,irow)];
			     else ToAddBg = CLHEP::RandPoisson::shoot(&RandomEngine.GetEngine(), AvgBackroundXY[std::make_pair(icol,irow)]);
		  
			   }
			   else {
			     if(AvgBackround[ipix2] < 0.01) ToAddBg = AvgBackround[ipix2];
			     else ToAddBg = CLHEP::RandPoisson::shoot(&RandomEngine.GetEngine(), AvgBackround[ipix2]);
			   }
			 }else{
			   ToAddBg = CLHEP::RandPoisson::shoot(&RandomEngine.GetEngine(), fBaseValue);
			 }
			 /// Accounts for the Non-Uniformity of the PMT
			 if  (fNonUniform ){
			   ToAddBg = CLHEP::RandPoisson::shoot(&RandomEngine.GetEngine(), multiplier*fBaseValue);	       double tempTrace=(double)trace[igtu]*multiplier;
			   trace[igtu] = round(tempTrace+ToAddBg*fVarriedModifier);
			 }
			 else{
			   /// makes the background an integer value via rounding
			   if (ToAddBg < 1)trace[igtu] += round(ToAddBg*fVarriedModifier);
			   else trace[igtu] += round(ToAddBg*fVarriedModifier);
			 }
		       }  //GTU loop 
		     }
		   }
          }
        }
      }
    }
  }
}

int BackgroundSimulator::GetPDMNum(Event& event){
  int numPDM=0;
  FEvent& fdEvent = event.GetFEvent();
   for (fevt::FEvent::EyeIterator eye = fdEvent.EyesBegin(ComponentSelector::eExists);
	 eye != fdEvent.EyesEnd(ComponentSelector::eHasData); ++eye) {
      for (fevt::Eye::TelescopeIterator tel = eye->TelescopesBegin(ComponentSelector::eExists);
	   tel != eye->TelescopesEnd(ComponentSelector::eExists); ++tel) {
	
	for (fevt::Telescope::CCBIterator ccb = tel->CCBsBegin(ComponentSelector::eExists);
	     ccb != tel->CCBsEnd(ComponentSelector::eExists); ++ccb) {
	  
	  for (fevt::CCB::PDMIterator pdm = ccb->PDMsBegin(ComponentSelector::eExists);
	       pdm != ccb->PDMsEnd(ComponentSelector::eExists); ++pdm) {
	    numPDM++;
	  }
	}
      }
   }
   return numPDM;

}


void BackgroundSimulator::MakeBlobs(Event& event){
  FEvent& fdEvent = event.GetFEvent();
  static utl::RandomEngine& RandomEngine =
    fwk::RandomEngineRegistry::GetInstance().Get(fwk::RandomEngineRegistry::eDetector);
  double runYN=CLHEP::RandFlat::shoot(&RandomEngine.GetEngine(), 0.,1.);
  if (fBlobFrac>runYN){
    double fN_Blob =CLHEP::RandPoisson::shoot(&RandomEngine.GetEngine(), fAverageBlobSize);
    int fn_Blob = round(fN_Blob);
    if (fn_Blob==0)fn_Blob=1;
    if (fn_Blob >24)fn_Blob=24;
    double fS_Blob = CLHEP::RandFlat::shoot(&RandomEngine.GetEngine(), 0,1);
    double fB_Blob= fAverageBlobBrightness+ pow(fS_Blob,-1.0/3.0)*1.5*fAverageBlobBrightness; 
    int fGTU_Blob=(int)((128-fLengthBloob)*CLHEP::RandFlat::shoot(&RandomEngine.GetEngine(), 0.,1.));
    int xCenter=(int)(48*numPDMs*CLHEP::RandFlat::shoot(&RandomEngine.GetEngine(), 0.05,.95));
    int yCenter=(int)(48*CLHEP::RandFlat::shoot(&RandomEngine.GetEngine(), 0.05,.95));
    int xLoc[fn_Blob];
    int yLoc[fn_Blob];
    int fB[fn_Blob];
    int xMap[25]={0,1,1,0,-1,-1,-1,0,1,2,2,2,2,1,0,-1,-2,-2,-2,-2,-2,-1,0,1,2};
    int yMap[25]={0,0,-1,-1,-1,0,1,1,1,1,0,-1,-2,-2,-2,-2,-2,-1,0,1,2,2,2,2,2};
    for (int i =0;i<fn_Blob;i++){
      xLoc[i]=xMap[i]+xCenter;
      yLoc[i]=yMap[i]+yCenter;
      fB[i]= round(fB_Blob/(float)fn_Blob);
    }
    
  
    ///Loops to loop over eyes, telescopes, PDMs, ECs, pmts, pixels and GTUs. 
    for (fevt::FEvent::EyeIterator eye = fdEvent.EyesBegin(ComponentSelector::eExists);
	 eye != fdEvent.EyesEnd(ComponentSelector::eHasData); ++eye) {
      for (fevt::Eye::TelescopeIterator tel = eye->TelescopesBegin(ComponentSelector::eExists);
	   tel != eye->TelescopesEnd(ComponentSelector::eExists); ++tel) {
	
	for (fevt::Telescope::CCBIterator ccb = tel->CCBsBegin(ComponentSelector::eExists);
	     ccb != tel->CCBsEnd(ComponentSelector::eExists); ++ccb) {
	  // add something here to fill sim info at CCB level (trigger data?)
	  
	  for (fevt::CCB::PDMIterator pdm = ccb->PDMsBegin(ComponentSelector::eExists);
	       pdm != ccb->PDMsEnd(ComponentSelector::eExists); ++pdm) {
	    int ipdm = pdm->GetId();
	    // add something here to fill info at PDM level
	    
	    for (fevt::PDM::ECIterator ec = pdm->ECsBegin(ComponentSelector::eExists);
		 ec != pdm->ECsEnd(ComponentSelector::eExists); ++ec) {
	      int iec = ec->GetId();
	      
	      for (fevt::EC::PMTIterator pmt = ec->PMTsBegin(ComponentSelector::eExists);
		   pmt != ec->PMTsEnd(ComponentSelector::eExists); ++pmt) {
		int ipmt = pmt->GetId();
		for (fevt::PMT::PixelIterator pix = pmt->PixelsBegin(ComponentSelector::eExists);
		     pix != pmt->PixelsEnd(ComponentSelector::eExists); ++pix) {
		  int ipix = pix->GetId();
	       
		  int irow = ((iec-1)/3)*16 + ((ipmt-1)/2)*8 + ((ipix-1)/8);
		  int icol = ((iec-1)%3)*16 + ((ipmt-1)%2)*8 + (ipix-1)%8 + (ipdm-1)*48;
		  for (int i=0;i<fn_Blob;i++){
		    if (irow==yLoc[i]&& icol==xLoc[i]){
		      TraceI& trace = pix->GetSimData().GetFADCTrace(FdConstants::eTotal);
		      for (int j =0;j<fLengthBloob;j++)
		      trace[fGTU_Blob+j] +=fB[i];
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


