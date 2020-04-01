#ifndef _BackgroundSimulator_h
#define _BackgroundSimulator_h

#include <fwk/VModule.h>
#include <evt/Event.h>
#include <fevt/Telescope.h>
#include <string>


namespace fevt {
  class FEvent;
}


namespace BackgroundSimulatorCSM {

  class BackgroundSimulator : public fwk::VModule {

  public:
    BackgroundSimulator();
    virtual ~BackgroundSimulator();

    fwk::VModule::ResultFlag Init();
    fwk::VModule::ResultFlag Run(evt::Event& event);
    fwk::VModule::ResultFlag Finish();

    void SetUpDataStructure(evt::Event& event);
    void SetBackground(evt::Event& event);
    void MakeBlobs(evt::Event& event);
    int GetPDMNum(evt::Event& event);

  private :
    evt::Event* fEvent;
    int nTrace ;
    double binning;
    std::map<std::pair<int,int>,double> AvgBackroundXY;
    std::map<int,double> AvgBackround;
    unsigned fEyeId;
    unsigned fTelId;
    unsigned fPDMId;
    bool fReadFromFile;
    std::string fPath;
    bool fIsSPB1;
    bool fTiltDependent;
    bool fNonUniform;
    double fBaseValue;
    double fTilt;
    double fAverageBlobSize;
    double fAverageBlobBrightness;
    double fBlobFrac;
    int numPDMs;
    int fLengthBloob;
    bool fVarriedBackground;
    double fVarriedModifier;

    REGISTER_MODULE("BackgroundSimulatorCSM", BackgroundSimulator);
  };
}

#endif //_BackgroundSimulator_h

// End:
