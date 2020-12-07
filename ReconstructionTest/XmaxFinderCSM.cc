#include "XmaxFinderCSM.h"

using namespace fwk;
using namespace std;

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

  fevt::Telescope& telescope = event.GetFEvent().GetEye(1).GetTelescope(1);   //The telescope
  det::Detector& detector = det::Detector::GetInstance();  //The detector
  const atm::Atmosphere& atmo = detector.GetAtmosphere();  //The atmosphere
  const atm::ProfileResult&  densityProfile = atmo.EvaluateDensityVsHeight();  //The density as a function of height of the atmosphere

evt::ShowerSimData& SimShower = event.GetSimShower();
  double fSimXmax=-1;
  double fSimTheta=-1;
  double fSimPhi=-1;
  double fSimEnergy=-1;
  if (SimShower.HasGHParameters()) {
      fSimXmax = SimShower.GetGHParameters().GetXMax() / utl::g * utl::cm2;
      fSimTheta = SimShower.GetZenith();
      fSimPhi=SimShower.GetAzimuth();
      fSimEnergy=SimShower.GetEnergy();
  }




  fevt::CoordinateDataCollection coords = telescope.GetRecData().GetCoordinatesRecData().GetCoordinates();

    for (unsigned i=0; i<coords.size(); ++i) {
      double ni = coords[i].GetPsi() /utl::radian;
      double charge = coords[i].GetCharge();
      fSignal.push_back(charge);
      fPsii.push_back(ni);
    }

  fR0 = telescope.GetRecData().GetRp() ;
  fPsi0=telescope.GetRecData().GetPsiZero() / utl::radian;


  const fdet::FDetector& detFD = detector.GetFDetector();
  utl::CoordinateSystemPtr telCS = detFD.GetEye(1).GetTelescope(1).GetTelescopeCoordinateSystem();

  utl::Vector fSDP=telescope.GetRecData().GetSDP();
  utl::Vector fAxis=telescope.GetRecData().GetAxis();
  utl::Vector R_PAxis= cross(fSDP,fAxis);

  fAxis.Normalize();
  R_PAxis.Normalize();


  fCorePosReco[0] = R_PAxis.GetX(telCS)*fR0;
  fCorePosReco[1] = R_PAxis.GetY(telCS)*fR0;
  fCorePosReco[2] = R_PAxis.GetZ(telCS)*fR0;



  fTheta  = utl::kPi-telescope.GetRecData().GetAxis().GetTheta(telCS) / utl::radian;
  fPhi = telescope.GetRecData().GetAxis().GetPhi(telCS) / utl::radian;
  double appertureSize=utl::kPi*0.5*0.5;
const utl::ReferenceEllipsoid& wgs84 =    utl::ReferenceEllipsoid::Get(utl::ReferenceEllipsoid::eWGS84);
double fHeight;
  for (int i=0;i<fPsii.size();i++)
  {
    Ri.push_back(fR0 / abs(cos(fPsi0-fPsii[i]))); //Distance from signal to detector
    dl =fR0*tan(fPsi0-fPsii[i]/utl::radian);

    dx = dl*sin(fTheta)*cos(fPhi);
    dy = dl*sin(fTheta)*sin(fPhi);
    dz = dl*cos(fTheta);
    x= fCorePosReco[0]+dx;
    y= fCorePosReco[1]+dy;
    z= fCorePosReco[2]+dz;
    utl::Point showerLoc(x,y,z,telCS);
    fHeight= wgs84.PointToLatitudeLongitudeHeight(showerLoc).get<2>();
    double temp=fHeight;
    double fXi=0;
    double stepSize=10.0;
    while (fHeight <densityProfile.MaxX()/utl::m){
      x-=stepSize*sin(fTheta)*cos(fPhi);
      y-=stepSize*sin(fTheta)*sin(fPhi);
      z-=stepSize*cos(fTheta);
      utl::Point showerLocNew(x,y,z,telCS);
      fHeight= wgs84.PointToLatitudeLongitudeHeight(showerLocNew).get<2>();
      double rho= densityProfile.Y(fHeight)/( utl::g / utl::cm3);
      fXi+=rho*stepSize*100.;

    }
    X.push_back(fXi);
    dE.push_back(4*utl::kPi*Ri[i]*Ri[i]*fSignal[i]/appertureSize);
    //cout<<X[i]<<'\t'<<dE[i]<<'\t'<<temp<<endl;
  }




  double maxX=*max_element(X.begin(),X.end());
  double minX=*min_element(X.begin(),X.end());
  double dEdXMax=*max_element(dE.begin(),dE.end());
  TH1F* h_dEDX = new TH1F("","",10,minX-10.0,maxX+10.0);


  for (int i=0;i<dE.size();i++)
  {
    h_dEDX->Fill(X[i],dE[i]/(dEdXMax*h_dEDX->GetBinWidth(i)));//,dEdX[i]);
  }

  TCanvas *c1 = new TCanvas("c1_fit1","The Fit Canvas",10,10,700,500);

  h_dEDX->Draw();
  TF1 *f = new TF1("f","[0]*((x-[2])/([1]-[2]))^(([1]-[2])/[3])*exp(([1]-x)/[3])");   //Gasier Hillas Function
  f->SetParameter(1,800);
  f->SetParameter(2,-20);
  f->SetParameter(0,10.0);
  f->SetParameter(3,50);
  f->SetParLimits(0,0.1,10);
  f->SetParLimits(1,600,1000);
  f->SetParLimits(2,-100,100);
  f->SetParLimits(3,-100,100);

  h_dEDX->Fit("f");
  double fXMax= f->GetParameter(1);
  double fYield337;

  utl::Branch topB = CentralConfig::GetInstance()->GetTopBranch("AirflyFluorescenceModel");
  topB.GetChild("yield337").GetData(fYield337);
  double fEnergy= f->Integral(0,2000)*dEdXMax/fYield337 /utl::eV;
  c1->Update();
  int runNum=event.GetHeader().GetEventNumber();
  string s="test"+to_string(runNum)+".png";
  c1->SaveAs(s.c_str());
  cout<<fXMax  <<endl;
  cout<<fEnergy<<endl;
  cout<<fSimEnergy<<endl;
  cout<<"QWERTY\t"<<fSimTheta/utl::deg<<'\t'<<fTheta/utl::deg<<'\t'<<fSimXmax<<'\t'<<fXMax<<'\t'<<fSimEnergy<<'\t'<<fEnergy<<'\t'<<endl;
  fPsii.clear();
  fSignal.clear();
  Ri.clear();
  dE.clear();
  X.clear();


  return eSuccess;
}

VModule::ResultFlag
XmaxFinderCSM::Finish()
{
  return eSuccess;
}

}
