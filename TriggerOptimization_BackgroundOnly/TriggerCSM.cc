#include "TriggerCSM.h"

using namespace fwk;
using namespace utl;
using namespace det;

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
  std::cout<<"TRIGGERS\tTriggerSPB2CSM\t"<<triggerState0<<"\tTriggerSPB2cells3x3TG\t" <<triggerState2<<"\tZenith\t"<<fZenith<<"\tAzimuth\t"<<fAzimuth<<"\tLocation\t"<<fCorePos[0]<<"\t"<<fCorePos[1]<<"\t"<<fCorePos[2]<<std::endl;
  return eSuccess;
}

VModule::ResultFlag 
TriggerCSM::Finish() 
{
  VModule* m1= VModuleFactory::Create("TriggerSPB2CSM");
  m1->Finish();
  return eSuccess;
}

}
