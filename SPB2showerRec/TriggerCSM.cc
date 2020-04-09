#include "TriggerCSM.h"

using namespace fwk;


namespace Trigger {

TriggerCSM::TriggerCSM() {}

TriggerCSM::~TriggerCSM(){}

VModule::ResultFlag 
TriggerCSM::Init()
{
  VModule* m1= VModuleFactory::Create("TriggerSPB2CSM");
  m1->Init();
  VModule* m2= VModuleFactory::Create("TriggerSPB2cellsTG");
  m2->Init();
  VModule* m3= VModuleFactory::Create("TriggerSPB2cells3x3TG");
  m3->Init();
  return eSuccess;
}

VModule::ResultFlag 
TriggerCSM::Run(evt::Event& event)
{
  
  event.SetTriggerState(0);
  VModule* m1= VModuleFactory::Create("TriggerSPB2CSM");
  m1->Run(event);
  int triggerState0=event.HasTrigger();
  event.SetTriggerState(0);
  VModule* m2= VModuleFactory::Create("TriggerSPB2cellsTG");
  m2->Run(event);
  int triggerState1=event.HasTrigger();
  event.SetTriggerState(0);
  VModule* m3= VModuleFactory::Create("TriggerSPB2cells3x3TG");
  m3->Run(event);
  int triggerState2=event.HasTrigger();
  std::cout<<"TriggerSPB2CSM\t"<<triggerState0<<"\tTriggerSPB2CellsTG\t"<<triggerState1<<"\tTriggerSPB2cells3x3TG\t" <<triggerState2<<std::endl;
  return eSuccess;
}

VModule::ResultFlag 
TriggerCSM::Finish() 
{
  return eSuccess;
}

}

// For special applications.
void JemEusoOfflineUser()
{
}
