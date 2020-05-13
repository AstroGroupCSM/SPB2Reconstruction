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
  //VModule* m2= VModuleFactory::Create("TriggerSPB2cellsTG");
  //m2->Init();
  VModule* m3= VModuleFactory::Create("TriggerSPB2cells3x3TG");
  m3->Init();
  //VModule* m4= VModuleFactory::Create("TriggerSPB2TG");
  //m4->Init();
  //VModule* m5= VModuleFactory::Create("TriggerTG");
  //m5->Init();
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
  //VModule* m2= VModuleFactory::Create("TriggerSPB2cellsTG");
  //m2->Run(event);
  //int triggerState1=event.HasTrigger();
  //event.SetTriggerState(0);
  VModule* m3= VModuleFactory::Create("TriggerSPB2cells3x3TG");
  m3->Run(event);
  int triggerState2=event.HasTrigger();
  event.SetTriggerState(0);
  //VModule* m4= VModuleFactory::Create("TriggerSPB2TG");
  //m4->Run(event);
  //int triggerState3=event.HasTrigger();
  //event.SetTriggerState(0);
  //VModule* m5= VModuleFactory::Create("TriggerTG");
  //m5->Run(event);
  //int triggerState4=event.HasTrigger();
  //std::cout<<"TRIGGERS\tTriggerSPB2CSM\t"<<triggerState0<<"\tTriggerSPB2CellsTG\t"<<triggerState1<<"\tTriggerSPB2cells3x3TG\t" <<triggerState2<<"\tTriggerSPB2TG\t" <<triggerState3<<"\tTriggerTG\t" <<triggerState4<<"\tZenith\t"<<fZenith<<"\tAzimuth\t"<<fAzimuth<<std::endl;
  std::cout<<"TRIGGERS\tTriggerSPB2CSM\t"<<triggerState0<<"\tTriggerSPB2cells3x3TG\t" <<triggerState2<<"\tZenith\t"<<fZenith<<"\tAzimuth\t"<<fAzimuth<<"\tLocation\t"<<fCorePos[0]<<"\t"<<fCorePos[1]<<"\t"<<fCorePos[2]<<std::endl;
  return eSuccess;
}

VModule::ResultFlag 
TriggerCSM::Finish() 
{
  return eSuccess;
}

}
