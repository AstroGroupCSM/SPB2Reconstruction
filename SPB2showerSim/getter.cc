//using namespace std;

// Standard C++ libraries
#include <stdio.h>
#include <iostream> // required for cout etc
#include <fstream>  // for reading in from / writing to an external file
#include <vector>
#include <string>
#include <sstream> // string manipulation
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iomanip>

// ROOT
#include "TH1F.h"   // Histogram
#include "TH2F.h"   // Histogram
#include "TH3F.h"
#include "TF1.h"
#include "TDirectory.h"
#include "TGraph.h" // Basic scatter plot
#include "TGraphErrors.h" 
#include "TGraphAsymmErrors.h"
#include "TMultiGraph.h"
#include "TFile.h"
#include "TList.h"
#include "TRandom3.h"

// Generic ROOT files, always include these
#include "TMath.h"
#include "TApplication.h"
#include "TCanvas.h"
#include "TROOT.h"
#include "TStyle.h"

using namespace std;
void getter()
{
  
TFile f("EusoDataFile.root");
  TTree *TShower=(TTree*)f.Get("tsim");
  bool hT;
  double z;
  TShower ->SetBranchAddress("zenith",&z);
  TShower ->SetBranchAddress("HasTrigger",&hT);
for (int i=0;i<25;i++){
   TShower->GetEntry(i);
   cout <<hT<<'\t'<<z*180.0/3.14159<<endl;
 }
}
