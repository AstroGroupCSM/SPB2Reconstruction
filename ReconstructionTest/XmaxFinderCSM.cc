#include "XmaxFinderCSM.h"

using namespace fwk;
using namespace std;
using namespace utl;
using namespace atm;
namespace XmaxFinderCSM {


XmaxFinderCSM::XmaxFinderCSM()
{
}

XmaxFinderCSM::~XmaxFinderCSM()
{
}

VModule::ResultFlag
XmaxFinderCSM::Init()
{
  return eSuccess;
}

VModule::ResultFlag
XmaxFinderCSM::Run(evt::Event& event)
{
  evt::ShowerSimData& SimShower = event.GetSimShower();
  double fXmax=-1;
  double fSimTheta=-1;
  double fSimPhi=-1;
  if (SimShower.HasGHParameters()) {
      fXmax = SimShower.GetGHParameters().GetXMax() / utl::g * utl::cm2;
      fSimTheta = SimShower.GetZenith();
      fSimPhi=SimShower.GetAzimuth();
  }


  hd =33000.0;
  rho_0=0.00123;
  scale_height=850000.0 ;
  fevt::Telescope& telescope = event.GetFEvent().GetEye(1).GetTelescope(1);

  det::Detector& detector = det::Detector::GetInstance();
    const Atmosphere& atmo = detector.GetAtmosphere();

  bool flatEarth = false;
  ProfileResult const * slantDepthVsDistance = 0;
  ProfileResult const * depthProfile = 0;
  try {
    slantDepthVsDistance = &atmo.EvaluateSlantDepthVsDistance();
  } catch (InclinedAtmosphericProfile::InclinedAtmosphereModelException& e) {
    WARNING("using flat earth!");
    flatEarth = true;
    depthProfile = &atmo.EvaluateDepthVsHeight();
  }


  fevt::CoordinateDataCollection coords = telescope.GetRecData().GetCoordinatesRecData().GetCoordinates();

    for (unsigned i=0; i<coords.size(); ++i) {
      double ni = coords[i].GetPsi() /radian;
      double charge = coords[i].GetCharge();
      fSignal.push_back(charge);
      fPsii.push_back(ni);
    }

    slope[0]=sin(fSimTheta)*cos(fSimPhi);
    slope[1]=sin(fSimTheta)*sin(fSimPhi);
    slope[2]=cos(fSimTheta);

  fR0 = telescope.GetRecData().GetRp() ;
  fPsi0=telescope.GetRecData().GetPsiZero() / radian;
//  fSignal = telescope.GetRecData().GetCoordinatesRecData().GetCoordinatesWeight();
//  fPsii = telescope.GetRecData().GetCoordinatesRecData().GetSpotsPsiAngles();

  //det::Detector& detector = det::Detector::GetInstance();
  const fdet::FDetector& detFD = detector.GetFDetector();
  CoordinateSystemPtr telCS = detFD.GetEye(1).GetTelescope(1).GetTelescopeCoordinateSystem();

  utl::Vector fSDP=telescope.GetRecData().GetSDP();
  utl::Vector fAxis=telescope.GetRecData().GetAxis();
  //utl::Vector fAxis(slope[0],slope[1],slope[2],telCS,Vector::kCartesian);
  utl::Vector R_PAxis= cross(fSDP,fAxis);

  fAxis.Normalize();
  R_PAxis.Normalize();

  //fCorePosReco[0] = telescope.GetRecData().GetAxis().GetX(telCS) * fR0 / sin(kPi - fPsi0 / radian) +R_PAxis.GetX(telCS)*fR0;
  //fCorePosReco[1] = telescope.GetRecData().GetAxis().GetY(telCS) * fR0 / sin(kPi - fPsi0 / radian) +R_PAxis.GetY(telCS)*fR0;
  //fCorePosReco[2] = telescope.GetRecData().GetAxis().GetZ(telCS) * fR0 / sin(kPi - fPsi0 / radian) +R_PAxis.GetZ(telCS)*fR0;
  //fR0=27000;
  fCorePosReco[0] = R_PAxis.GetX(telCS)*fR0;
  fCorePosReco[1] = R_PAxis.GetY(telCS)*fR0;
  fCorePosReco[2] = R_PAxis.GetZ(telCS)*fR0;



  fTheta  = kPi-telescope.GetRecData().GetAxis().GetTheta(telCS) / radian;
  fPhi = telescope.GetRecData().GetAxis().GetPhi(telCS) / radian;
  cout<<fSimTheta<<'\t'<<fTheta<<'\t'<<fPsi0<<'\t'<<  fCorePosReco[2]<<endl;
  //fTheta=fSimTheta;
  //if (abs(fSimTheta-fTheta)>0.17)
  //return eSuccess;
  //fPhi=-1*fPhi;
  //if(fPhi>kPi*2) fPhi-=2*kPi;
  //if(fPhi<0) fPhi+=2*kPi;
  //fTheta = kPi -fTheta;

  //fAxis.Normalize();
  for (int i=0;i<fPsii.size();i++)
  {
    Ri.push_back(fR0 / abs(cos(fPsi0-fPsii[i]))); //Distance from signal to detector
    dl =fR0*tan(fPsi0-fPsii[i]/radian);
  //  if (fPsi0>fPsii[i])dl = -1*fR0*abs(tan(fPsi0-fPsii[i]));
    //if (fPsi0<fPsii[i]) dl = fR0*abs(tan(fPsi0-fPsii[i]));
    dz = dl*cos(fTheta);
    double testt=slantDepthVsDistance->Y( fCorePosReco[2]-1*dl);
    hi.push_back(33000.0-fCorePosReco[2]-dz);
    cout<<Ri[i]<<'\t'<<hi[i]<<'\t'<<testt<<endl;
  }




for (int i=0;i<fSignal.size();i++)
  {
    if (hi[i]<0) hi[i]=0.0;
    if (hi[i]>33000.0) hi[i]=33000.0;
    dE.push_back(4*kPi*Ri[i]*Ri[i]*fSignal[i]);
    //dE.push_back(fSignal[i]);
    X.push_back(rho_0*scale_height /abs(cos(fTheta))* exp(-hi[i]*100.0/scale_height));
    dEdX.push_back(dE[i]/X[i]);
    cout<<X[i]<<'\t'<<dEdX[i]<<endl;
  }
  double maxX=*max_element(X.begin(),X.end());
  double minX=*min_element(X.begin(),X.end());
  double dEdXMax=*max_element(dEdX.begin(),dEdX.end());
  TH1F* h = new TH1F("a","a",10,700.0,1100.0);
  TH1F* h2 = new TH1F("b","b",10,700.0,1100.0);


  for (int i=0;i<dEdX.size();i++)
  {
  h->Fill(X[i],dEdX[i]/dEdXMax);//,dEdX[i]);
  h2->Fill(X[i],fSignal[i]);
}
//for(int i=1;i<=h->GetEntries();i++){
//    h->SetBinError(i,200.0/h2->GetBinContent(i));
//}
  TCanvas *c1 = new TCanvas("c1_fit1","The Fit Canvas",10,10,700,500);

  h->Draw();
  TF1 *f = new TF1("f","[0]*((x-[2])/([1]-[2]))^(([1]-[2])/[3])*exp(([1]-x)/[3])");
 f->SetParameter(1,800);
  f->SetParameter(2,-20);
  f->SetParameter(0,10.0);
  f->SetParameter(3,50);
  f->SetParLimits(0,0.1,10);
  f->SetParLimits(1,600,1000);
  f->SetParLimits(2,-100,100);
  f->SetParLimits(3,-100,100);

  h->Fit("f");
//h->Scale(dEdXMax);

  //f->Scale(dEdXMax);
  string temp="Xmax="+to_string(round(fXmax * 100)/100).substr(0,4)+"Simulated \n Xmax="+to_string(round( (f->GetParameter(1))* 100)/100).substr(0,4)+"Reconstructed;Atmospheric Depth;dNdX";
  h->SetTitle(temp.c_str());

  //h_ShowerAxis->Draw();
  c1->Update();
  int runNum=event.GetHeader().GetEventNumber();
  string s="test"+to_string(runNum)+".png";
//  if(abs(fXmax-(f->GetParameter(1)))<100 && (f->GetParameter(1))!=800)
  c1->SaveAs(s.c_str());
  cout<<fR0<<endl;
  cout<<fXmax<<endl;
  cout <<fTheta<<endl;
  cout<<fSignal.size()<<endl;

  fPsii.clear();
  fSignal.clear();
  hi.clear();
  Ri.clear();
  dE.clear();
  X.clear();
  slope[0]=fAxis.GetX(telCS);
  slope[1]=fAxis.GetY(telCS);
  slope[2]=fAxis.GetZ(telCS);
  dEdX.clear();

    TH3D *h_ShowerAxis = new TH3D("","",300,-50000.1,50000.1,300,-50000.1,50000.1,300,-33000.1,33000.1);
    double x=fCorePosReco[0];
    double y=fCorePosReco[1];
    double z=fCorePosReco[2];

    while (z>0){
      x+=10*slope[0];
      y+=10*slope[1];
      z+=10*slope[2];

      dist =sqrt(x*x+y*y+z*z);
      cout<<dist<<"\t"<<x<<"\t"<<y<<"\t"<<z<<endl;
      h_ShowerAxis->Fill(x,y,33000-z);

    }
    x=fCorePosReco[0];
    y=fCorePosReco[1];
    z=fCorePosReco[2];
    while (z<33000){
      x-=10*slope[0];
      y-=10*slope[1];
      z-=10*slope[2];

      dist =sqrt(x*x+y*y+z*z);
      cout<<dist<<"\t"<<x<<"\t"<<y<<"\t"<<z<<endl;
      h_ShowerAxis->Fill(x,y,33000-z);

    }
    h_ShowerAxis->Draw();
    c1->Update();
    c1->SaveAs("showerAxis.png");
  return eSuccess;
}

VModule::ResultFlag
XmaxFinderCSM::Finish()
{
  return eSuccess;
}

}
