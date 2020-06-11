#ifndef _TriggerSPB2cells3x3TG_TriggerSPB2cells3x3_h
#define _TriggerSPB2cells3x3TG_TriggerSPB2cells3x3_h

#include <string>
#include <iostream>
#include <fstream>

#include <fwk/VModule.h>

using namespace std;

namespace TriggerSPB2cells3x3TG {

  class TriggerSPB2cells3x3 : public fwk::VModule {

  public:
    TriggerSPB2cells3x3();
    virtual ~TriggerSPB2cells3x3();

    fwk::VModule::ResultFlag Init();
    fwk::VModule::ResultFlag Run(evt::Event& event);
    fwk::VModule::ResultFlag Finish();


  private :
 
    //output text file with trigger result
    string outFile="./output_triggers.txt";
    ofstream outString;

    const static int nGTU=128;
    const static int nPDM=3;
    const static int nPMT=36;
    const static int nPixX=8;
    const static int nPixY=8;
    const static int nCell=36;
    const static int nGTUpers=10; // the actual value is set in the configuration file with fNGTUpersistence, but here needed to declare boolean_active_cell[nGTUpers][nPDM][nPMT][nCell]
  
    void ProcessBuffer (int buffer[nPDM][nPMT][nPixX][nPixY], int threshold[nPDM][nPMT][nPixX][nPixY], int boolean_buffer[nPDM][nPMT][nPixX][nPixY]); 
    void DefineAndSumCells (int boolean_buffer[nPDM][nPMT][nPixX][nPixY], int pixel_OT_in_cell[nPDM][nPMT][nCell]);
    void WriteBooleanActiveCell (int boolean_active_cell[][nPDM][nPMT][nCell], int pixel_OT_in_cell[nPDM][nPMT][nCell], int n_GTU, int gtu);
    void AnalyzeBooleanActiveCell (int boolean_active_cell[][nPDM][nPMT][nCell], int &trigger_flag, int n_GTU, int event, int gtu, ofstream &outString);
    void Input(evt::Event& event);
  
    //in configuration files
    int fVerbosityLevel;
    bool fSignalOnly;
    bool fWantTriggerOutput;

    int fNGTUpersistence;
    int fNPixelInCell;
    int fNCellInPMT;
    double fNSigma;
    //

    int fPDMid;
    int RunNum;
    int ievent;
    int gtuMax;
    int gtuMin;
    int gtuTrigMin;
    int gtuTrigMax;
    int pdmCounts[3];
    int iphe[nGTU][nPDM][nPMT][nPixX][nPixY];
    int ipheSig[nGTU][nPDM][nPMT][nPixX][nPixY];

    int buffer[nPDM][nPMT][nPixX][nPixY]; // buffer containing 3 PDMs at the time. Shape is 3 PDMs, 36 PMTs, 8x8 pixels
    int boolean_buffer[nPDM][nPMT][nPixX][nPixY]; // 1 if a pixel is above threshold, 0 otherwise. Shape is 3 PDMs, 36 PMTs, 8x8 pixels
   
    int trigger;
    int trigger_flag;
  
    int sum_pixel[nPDM][nPMT][nPixX][nPixY];
    int threshold[nPDM][nPMT][nPixX][nPixY];
 
    int pixel_OT_in_cell[nPDM][nPMT][nCell]; // How many pixels are OverThresholdd in each cell (between 0 and 9). Shape is 3 PDMs, 36 PMTs, 36 cells
    int boolean_active_cell[nGTUpers][nPDM][nPMT][nCell]; // 1 if a cell is active, 0 otherwise. A cell is active if there are more than 'n_pxl_in_cell' pixel above threshold. Holds in memory 'n_GTU' GTUs

    REGISTER_MODULE("TriggerSPB2cells3x3TG", TriggerSPB2cells3x3);
    
  };
}

#endif 
