#include "Halide.h"
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h> 
using namespace Halide;

//Func buildRbfFunc(std::string call, Func rij, Func scalefunc, Expr rin, Expr rmax, int pdegree, int K, )

 void buildSnap(Func &  rbf, Func xij, Func besselparams, Expr rin, Expr rcut,
	       Expr besseldegree, Expr inversedegree, Expr nbseelpars, Expr npairs )
{
  Func abf("snap_abf");//inverse radial
  Func rbfp("snap_rbfr");//bessel

  Var np("nij"), bfp("snap_bfp"), bfd("snap_bfd"), ibfp("snap_abfp");

  Expr one = Expr((double) 1.0);
  Expr rmax = rcut - rin;
  Expr dij = xij(np);
  Expr r = dij - rin;        
  Expr y = r/rmax;      
  Expr y2 = y*y;
  Expr y3 = one - y2*y;
  Expr y4 = y3*y3 + Expr((double)1e-6);
  Expr y5 = sqrt(y4);
  Expr y6 = exp(-one/y5);
  Expr fcut = y6/exp(-one);

  Expr alpha = max(besselparams(bfp), Expr((double) 1e-3));
  Expr x =  (one - exp(-alpha*r/rmax))/(one-exp(-alpha));


  Expr a = (bfd+1)*Expr((double)M_PI);
  Expr b = (sqrt(2 * one/(rmax))/(bfd+1));

  rbfp(bfp, bfd, np) =  b*fcut*sin(a*x)/r;
  rbfp.compute_root();
  abf(ibfp, np) = fcut/pow(dij, ibfp + 1);


  rbf(bfp, bfd, np)= Expr((double) 0.0);
  RDom rpp(0, npairs, 0, besseldegree, 0, nbseelpars); //r.x, r.y, r.z
  rbf(rpp.z, rpp.y, rpp.x) = rbfp(rpp.z, rpp.y, rpp.x); 

  RDom rp(0, npairs, 0, inversedegree);
  rbf(nbseelpars, rp.y, rp.x) = abf(rp.y, rp.x);
  // return rbf;
}

void buildRBF(Func & rbf, Func & drbf, Func & abf, Func & dabf,
	      Func xij, Func besselparams, Expr rin, Expr rmax,
	      Expr bdegree, Expr adegree, Expr nbparams, Expr npairs,
	      Var bfi, Var bfp, Var np, Var dim)
{

  Expr one = Expr((double) 1.0);
  Expr zero = Expr((double) 0.0);
  Expr onefive = Expr((double) 1.5);
  Expr PI = Expr( (double)M_PI);
  Expr xij1 = xij(np, 0);
  Expr xij2 = xij(np, 1);
  Expr xij3 = xij(np, 2);
  // Expr xij1 = print(xij(np, 0), "<- xij1");
  // Expr xij2 = print(xij(np, 1), "<- xij2");
  // Expr xij3 = print(xij(np, 2), "<- xij3");

  Expr s = xij1*xij1 + xij2*xij2 + xij3*xij3;
  Expr dij = sqrt(s);
  Expr dr1 = xij1/dij;    
  Expr dr2 = xij2/dij;    
  Expr dr3 = xij3/dij;    

  Expr r = dij - rin;        
  Expr y = r/rmax;    
  Expr y2 = y*y;
  Expr y3 = one - y2*y;
  Expr y4 = y3*y3 + Expr((double) 1e-6);
  Expr y5 = sqrt(y4); //pow(y4, 0.5);
  Expr y6 = exp(-one/y5);
  Expr y7 = pow(y4, onefive);
  Expr fcut = y6/exp(-one);
  Expr dfcut = (((3 * one)/(rmax*exp(-one)))*(y2)*y6*(y*y2 - one))/y7;

  Expr alpha = max(Expr((double)1e-3), besselparams(bfp));
  Expr x =  (one - exp(-alpha*r/rmax))/(one-exp(-alpha));
  Expr dx = (alpha/rmax)*exp(-(alpha*r/rmax))/(one - exp(-alpha));

  Expr a = (bfi + 1) * PI;
  Expr b = sqrt(2 * one/rmax)/(bfi + 1);
  Expr c = pow(dij, bfi + 1);

  rbf(bfp, bfi, np) = b * fcut * sin(a*x)/r;
  // rbf.trace_stores();
  rbf.bound(bfp, 0, nbparams);
  rbf.bound(bfi, 0, bdegree);
  rbf.bound(np, 0, npairs);
  Expr drbfdr = b*(dfcut*sin(a*x)/r - fcut*sin(a*x)/(r*r) + a*cos(a*x)*fcut*dx/r);
  drbf(bfp, bfi, np, dim) = (xij(np, dim)/dij) * drbfdr;
  // drbf.trace_stores();
  drbf.bound(dim, 0, 3);
  drbf.bound(bfp, 0, nbparams);
  drbf.bound(bfi, 0, bdegree);
  drbf.bound(np, 0, npairs);

  Expr power = pow(dij, bfi+one);
  abf(bfi, np) = fcut/power;;
  // abf.trace_stores();
  abf.bound(bfi, 0, adegree);
  abf.bound(np, 0, npairs);
  Expr drbfdr_a = dfcut/c - (bfi+one)*fcut/(c*dij);
  dabf(bfi, np, dim) = (xij(np, dim)/dij) * drbfdr_a;
  // dabf.trace_stores();
  dabf.bound(dim, 0, 3);
  dabf.bound(bfi, 0, adegree);
  dabf.bound(np, 0, npairs);

  rbf.compute_root();
  drbf.compute_root();

  abf.compute_root();
  
  dabf.compute_root();

  
}

void buildStructureMatMul(Func & energyij, Func & forceij,
			  Func rbf, Func abf, Func drbf, Func dabf, Func Phi1, Func Phi2,
			  Expr bdegree, Expr adegree, Expr tdegree, Expr nbparams, Expr npairs,
			  Var bfi, Var bfa, Var bfp, Var np, Var dim){
  //Multiply atom * basisfunction  by basisfunction * rbf
  Expr zero = Expr((double) 0.0);

  energyij(bfi, np)= zero;
  forceij(bfi, np, dim) = zero;
  
  energyij.bound(bfi, 0, tdegree);
  energyij.bound(np, 0, npairs);
  forceij.bound(bfi, 0, tdegree);
  forceij.bound(np, 0, npairs);
  forceij.bound(dim, 0, 3);

  RDom rbfdom(0, bdegree, 0, nbparams);
  //  RDom drbf(0, bdegree, 0, nbparams, 0, 3);
  // energyij(bfi, np) += rbf(rbfdom.y, rbfdom.x, np) * Phi1(rbfdom.y, rbfdom.x, bfi);//ordering here is questionable
  energyij(bfi, np) += rbf(rbfdom.y, rbfdom.x, np) * Phi1(rbfdom.x, rbfdom.y, bfi);//ordering here is questionable
  forceij(bfi,np, dim) += drbf(rbfdom.y, rbfdom.x, np, dim) * Phi1(rbfdom.x, rbfdom.y, bfi);//ordering here is questionable
  //  energyij.bound(bfi, 0, tdegree);
  //  forceij.bound(bfi, 0, tdegree);

  RDom abfdom(0, adegree);
  energyij(bfi, np) += abf(abfdom.x, np) * Phi2(abfdom.x, bfi);//ordering here is questionable
  forceij(bfi, np, dim) += dabf(abfdom.x, np, dim) * Phi2(abfdom.x, bfi);//ordering here is questionable

  // rbf.trace_loads();
  // drbf.trace_loads();
  // abf.trace_loads();
  // dabf.trace_loads();
  // Phi1.trace_loads();
  // Phi2.trace_loads();
  // energyij.trace_stores();
  // forceij.trace_stores();

  energyij.compute_root();
  forceij.compute_root();
  
  

  
}



void buildPodTally2b(Func & eatom, Func & fatom,
		     Func eij, Func fij, Func o_aiajtitj, Func interaction,
		     Expr npairs, Expr natom, Expr nelements, Expr nelementCombos, Expr nbf,
		     Var np, Var atom, Var bf, Var dim, Var elem, Var inter
		     )
{
  Expr one = Expr((double) 1.0);
  Expr zero = Expr((double) 0.0);
  eatom(atom, inter, bf) = zero;
  fatom(dim, atom, inter, bf) = zero;
  RDom r(0, npairs, 0, nbf);

  Expr i1 = clamp(o_aiajtitj(r.x, 0), 0, natom - 1); //ai(r.x)
  Expr j1 = clamp(o_aiajtitj(r.x, 1), 0, natom - 1);
  Expr typei = clamp(o_aiajtitj(r.x, 2) - 1, 0, nelements - 1);
  Expr typej = clamp(o_aiajtitj(r.x, 3) - 1, 0, nelements - 1);

  Expr inter_ij = clamp(interaction(typei, typej) - 1, 0, nelementCombos - 1);
  eatom(i1, inter_ij,  r.y) += eij(r.y, r.x);
  fatom(dim, i1, inter_ij, r.y) += fij(r.y, r.x, dim);
  fatom(dim, j1, inter_ij, r.y) -= fij(r.y, r.x, dim);
  // eij.trace_loads();
  // fij.trace_loads();


  //  eatom.compute_root();
  //  fatom.compute_root();

  
  eatom.bound(atom, 0, natom);
  eatom.bound(inter, 0, nelementCombos);
  eatom.bound(bf, 0, nbf);
  
  fatom.bound(atom, 0, natom);
  fatom.bound(inter, 0, nelementCombos);
  fatom.bound(bf, 0, nbf);
  fatom.bound(dim, 0, 3);

  eatom.compute_root();
  fatom.compute_root();
  
  


}

void buildPodTally3b(Func & eatom, Func & fatom,
		     Func xij, Func e2ij, Func f2ij, Func interaction,
		     Func pairlist, Func pairnumsum, Func atomtype, Func alist,
		     Expr nrbf, Expr nabf, Expr nelems, Expr nelementCombos, Expr nl, Expr natom, Expr nij, Expr nmax, 
		     Var atom, Var atom_o, Var atom_i, Var atom_j, Var atom_k, Var inter, Var type, Var abf, Var rbf, Var dim){
  Expr one = Expr((double) 1.0);
  Expr zero = Expr((double) 0.0);

  //  eatom(rbf, abf, type, inter, atom) = zero;
  // eatom(atom, inter, type, abf, rbf) = zero;
  // fatom(dim, atom, inter, type, abf, rbf) = zero;
  eatom(atom, inter, type, rbf, abf) = zero;
  fatom(dim, atom, inter, type, rbf, abf) = zero;
  //  fatom(rbf, abf, type, inter, atom,  dim) = zero;

    
  eatom.bound(rbf, 0, nrbf); 
  eatom.bound(abf, 0, nabf);
  eatom.bound(type, 0, nelems);
  eatom.bound(inter, 0, nelementCombos);
  eatom.bound(atom, 0, natom);

  fatom.bound(rbf, 0, nrbf);
  fatom.bound(abf, 0, nabf);
  fatom.bound(type, 0, nelems);
  fatom.bound(inter, 0, nelementCombos);
  fatom.bound(atom, 0, natom);
  fatom.bound(dim, 0, 3);

  /*
  // RDom comp(0, natom, 0, nij, 0, nij, 0, nrbf, 0, nabf);
  RDom comp(0, nabf, 0, nrbf, 0, nmax, 0, nmax, 0, natom);
  
  RVar p = comp[0];
  RVar m = comp[1];
  RVar lk = comp[2];
  RVar lj = comp[3];
  RVar i = comp[4];
  
  comp.where(lj < pairnumsum(i+1) - pairnumsum(i) );
  comp.where(lk < (pairnumsum(i+1) - pairnumsum(i)) - (lj + 1));
  
  //comp.where(lk > lj & lk < pairnumsum(i+1));
  
  //comp.where(lk > lj & lk < pairnumsum(i + 1));
  //comp.where(lj < pairnumsum(i + 1)  & lj >=  pairnumsum(i));

  Expr ljs =clamp(lj + pairnumsum(i), 0, nij - 1);
  Expr lks = clamp(lk + lj + pairnumsum(i) + 1, 0, nij - 1);
  //Expr ljs = lj;
  //Expr lks = lk;
  */

  RDom comp(0, nabf, 0, nrbf, 0, nij, 0, nij, 0, natom);

  RVar p = comp[0];
  RVar m = comp[1];
  RVar lk = comp[2];
  RVar lj = comp[3];
  RVar i = comp[4];
  
  comp.where(lj < pairnumsum(i + 1) && lj >= pairnumsum(i));
  comp.where(lk > lj & lk < pairnumsum(i + 1));

  Expr ljs = lj;
  Expr lks = lk;

  
  // comp.where(lj < pairnumsum(i + 1) - pairnumsum(i));
  // comp.where(lk < pairnumsum(i + 1) - lj);
  // Expr lks = clamp(lk + pairnumsum(i) + lj, 0, nij - 1);
  // Expr ljs = clamp(lj + pairnumsum(i), 0, nij - 1);

  // RVar i = comp[0];
  // RVar lj = comp[1];
  // RVar lk = comp[2];
  // RVar m = comp[3]; //rbf
  // RVar p = comp[4]; //abf

  
  //  comp.where(comp.y >= pairnumsum(comp.x) & comp.y < pairnum(comp.x) + pairnumsum(comp.x) & lk > comp.y & lk < pairnum(comp.x) + pairnumsum(comp.x));
  
  
  //  comp.where(lk < pairnumsum(i + 1));

 
  Func xij_inter("xij_inter");
  xij_inter(atom_o, atom_i, dim) = xij(clamp(pairlist(atom_o), 0, nl * natom - 1), dim) - xij(atom_i, dim);
  // xij_inter(atom_o, atom_i, dim) = print(xij(clamp(pairlist(atom_o), 0, nl * natom - 1), dim) - xij(atom_i, dim), "<- is xij_inter");
  xij_inter.bound(atom_o, 0, nl * natom);
  xij_inter.bound(atom_i, 0, natom);
  // xij_inter.bound(atom_o, 0, nij);
  xij_inter.bound(dim, 0, 3);

  Func rij_inter("rij_inter"); // 1st xdot in original code
  Func rij_sq_inter("rij_sq_inter");
  rij_inter(atom, atom_o) = xij_inter(atom, atom_o, 0) * xij_inter(atom,atom_o, 0) + xij_inter(atom, atom_o, 1) * xij_inter(atom, atom_o, 1) + xij_inter(atom, atom_o, 2) * xij_inter(atom, atom_o, 2);
  rij_inter.bound(atom, 0, nij);
  // rij_inter.bound(atom_o, 0, natom);
  rij_inter.bound(atom_o, 0, nl * natom);
  rij_sq_inter(atom, atom_o) = sqrt(rij_inter(atom, atom_o));
  rij_sq_inter.bound(atom, 0, nij);
  // rij_sq_inter.bound(atom_o, 0, natom);
  rij_sq_inter.bound(atom_o, 0, nl * natom);

  Func costhe("costhe");
  costhe(atom_i, atom_j, atom_k) = clamp((xij_inter(atom_j, atom_i, 0) * xij_inter(atom_k, atom_i, 0) + xij_inter(atom_j, atom_i, 1) * xij_inter(atom_k, atom_i, 1) + xij_inter(atom_j, atom_i, 2) * xij_inter(atom_k, atom_i, 2))/ (rij_sq_inter(atom_j, atom_i) * rij_sq_inter(atom_k, atom_i)), -1, 1);
  // costhe.bound(atom_i, 0, natom);
  costhe.bound(atom_i, 0, nl * natom);
  costhe.bound(atom_j, 0, nij);
  costhe.bound(atom_k, 0, nij);
  
  Func xdot("xdot");
  xdot(atom_i, atom_j, atom_k) = costhe(atom_i, atom_j, atom_k) * (rij_sq_inter(atom_j, atom_i) * rij_sq_inter(atom_k, atom_i));
  Func sinthe("sinthe");
  sinthe(atom_i, atom_j, atom_k) = max(sqrt(one - costhe(atom_i, atom_j, atom_k) * costhe(atom_i, atom_j, atom_k)), Expr( (double) 1e-12));
  
  // xdot.bound(atom_i, 0, natom);
  xdot.bound(atom_i, 0, nl * natom);
  xdot.bound(atom_j, 0, nij);
  xdot.bound(atom_k, 0, nij);

  
  // sinthe.bound(atom_i, 0, natom);
  sinthe.bound(atom_i, 0, nl * natom);
  sinthe.bound(atom_j, 0, nij);
  sinthe.bound(atom_k, 0, nij);

  Func theta("theta");
  theta(atom_i, atom_j, atom_k) = acos(costhe(atom_i, atom_j, atom_k));
  // theta(atom_i, atom_j, atom_k) = print(acos(costhe(atom_i, atom_j, atom_k)), "<- theta", atom_i, atom_j, atom_k);;
  
  // theta.bound(atom_i, 0, natom);
  theta.bound(atom_i, 0, nl * natom);
  theta.bound(atom_j, 0, nij);
  theta.bound(atom_k, 0, nij);
  Func dtheta("dtheta");
  dtheta(atom_i, atom_j, atom_k) = -one/sinthe(atom_i, atom_j, atom_k);
  // dtheta(atom_i, atom_j, atom_k) = print(-one/sinthe(atom_i, atom_j, atom_k), "<- dtheta", atom_i, atom_j, atom_k);
  // dtheta.trace_stores();
  // dtheta.trace_loads();
  
  dtheta.bound(atom_i, 0, natom);
  // dtheta.bound(atom_i, 0, nl * natom);
  dtheta.bound(atom_j, 0, nij);
  dtheta.bound(atom_k, 0, nij);
  // dtheta.bound(atom_j, 0, nl * natom);
  // dtheta.bound(atom_k, 0, nl * natom);

  Func dct("dct");
  dct(atom_i, atom_j, atom_k, dim) = (xij_inter(atom_k, atom_i, dim) * rij_inter(atom_j, atom_i) - xij_inter(atom_j, atom_i, dim) * xdot(atom_i, atom_j, atom_k)) * (one/(pow(rij_inter(atom_j, atom_i), Expr((double) 1.5)) * rij_sq_inter(atom_k, atom_i)));
  // dct(atom_i, atom_j, atom_k, dim) = print((xij_inter(atom_k, atom_i, dim) * rij_sq_inter(atom_j, atom_i) - xij_inter(atom_j, atom_i, dim) * xdot(atom_i, atom_j, atom_k)) * (one/(pow(rij_sq_inter(atom_j, atom_i), Expr((double) 1.5)) * rij_inter(atom_k, atom_i))), "<- dct", atom_i, atom_j, atom_k, dim);
  // dct.trace_stores();
  // dct.trace_loads();
  
  // dct.bound(atom_i, 0, natom);
  dct.bound(atom_i, 0, nl * natom);
  dct.bound(atom_j, 0, nij);
  dct.bound(atom_k, 0, nij);
  dct.bound(dim, 0, 3);
  

  /*
  Func dct123("dct123");
  Func dct456("dct456");
  dct123(atom_i, atom_j, atom_k, dim) = (xij_inter(atom_k, atom_i, dim) * rij_inter(atom_j, atom_i) - xij_inter(atom_j, atom_i, dim) * xdot(atom_i, atom_j, atom_k)) * (one/(pow(rij_inter(atom_j, atom_i), Expr((double) 1.5)) * rij_sq_inter(atom_k, atom_i)));
  dct456(atom_i, atom_j, atom_k, dim) = (xij_inter(atom_j, atom_i, dim) * rij_inter(atom_k, atom_i) - xij_inter(atom_k, atom_i, dim) * xdot(atom_i, atom_j, atom_k)) * (one/pow(rij_inter(atom_k, atom_i), Expr((double) 1.5)) * rij_sq_inter(atom_j, atom_i));
  dct123.bound(atom_i, 0, nl * natom);
  dct123.bound(atom_j, 0, nij);
  dct123.bound(atom_k, 0, nij);
  dct456.bound(atom_i, 0, nl * natom);
  dct456.bound(atom_j, 0, nij);
  dct456.bound(atom_k, 0, nij);
  */
  

  Func pre_abf("pre_abf");
  pre_abf(atom_i, atom_j, atom_k, abf) = cos(abf * theta(atom_i, atom_j, atom_k));
  
  pre_abf.bound(abf, 0, nabf);
  // pre_abf.bound(atom_i, 0, natom);
  pre_abf.bound(atom_i, 0, nl * natom);
  pre_abf.bound(atom_j, 0, nij);
  pre_abf.bound(atom_k, 0, nij);
  
  Func pre_dabf("pre_dabf");
  pre_dabf(atom_i, atom_j, atom_k, abf, dim) = -abf * sin(abf * theta(atom_i, atom_j, atom_k)) * dtheta(atom_i, atom_j, atom_k) * dct(atom_i, atom_j, atom_k, dim); // switched k and j in dct -- changed back
  // pre_dabf(atom_i, atom_j, atom_k, abf, dim) = print(-abf * sin(abf * theta(atom_i, atom_j, atom_k)) * dtheta(atom_i, atom_j, atom_k) * dct(atom_i, atom_j, atom_k, dim), "<- dabf");
  pre_dabf.bound(abf, 0, nabf);
  // pre_dabf.bound(atom_i, 0, natom);
  pre_dabf.bound(atom_i, 0, nl * natom);
  pre_dabf.bound(atom_j, 0, nij);
  pre_dabf.bound(atom_k, 0, nij);
  pre_dabf.bound(dim, 0, 3);

  /*
  Func pre_dabf123("pre_dabf123");
  Func pre_dabf456("pre_dabf456");
  pre_dabf123(atom_i, atom_j, atom_k, abf, dim) = -abf * sin(abf * theta(atom_i, atom_j, atom_k)) * dtheta(atom_i, atom_j, atom_k) * dct123(atom_i, atom_j, atom_k, dim);
  pre_dabf456(atom_i, atom_j, atom_k, abf, dim) = -abf * sin(abf * theta(atom_i, atom_j, atom_k)) * dtheta(atom_i, atom_j, atom_k) * dct456(atom_i, atom_j, atom_k, dim);
  pre_dabf123.bound(atom_i, 0, nl * natom);
  pre_dabf123.bound(atom_j, 0, nij);
  pre__dabf123.bound(atom_k, 0, nij);
  pre_dabf456.bound(atom_i, 0, nl * natom);
  pre_dabf456.bound(atom_j, 0, nij);
  pre_dabf456.bound(atom_k, 0, nij);
  */

  Func pre_rbf("pre_rbf");
  pre_rbf(rbf, atom_j, atom_k) = e2ij(rbf, atom_j) * e2ij(rbf, atom_k);
  pre_rbf.bound(atom_j, 0, nij);
  pre_rbf.bound(atom_k, 0, nij);
  pre_rbf.bound(rbf, 0, nrbf);
  
  Func pre_drbf("pre_drbf");
  pre_drbf(rbf, atom_j, atom_k, dim) = f2ij(rbf, atom_j, dim) * e2ij(rbf, atom_k);
  // pre_drbf(rbf, atom_j, atom_k, dim) = print(f2ij(rbf, atom_j, dim) * e2ij(rbf, atom_k), "<- pre_drbf");
  // pre_dabf.trace_stores();
  // pre_drbf.trace_loads();
  pre_drbf.bound(atom_j, 0, nij);
  pre_drbf.bound(atom_k, 0, nij);
  pre_drbf.bound(rbf, 0, nrbf);
  pre_drbf.bound(dim, 0, 3);
  /*
  Func pre_drbf123("pre_drbf123");
  Func pre_drbf456("pre_drbf456");
  pre_drbf123(rbf, atom_j, atom_k, dim) = f2ij(rbf, atom_j, dim) * e2ij(rbf, atom_k);
  pre_drbf456(rbf, atom_j, atom_k, dim) = f2ij(rbf, atom_k, dim) * e2ij(rbf, atom_j);
  pre_drbf123.bound(atom_j, 0, nij);
  pre_drbf123.bound(atom_k, 0, nij);
  pre_drbf123.bound(rbf, 0, nrbf);
  pre_drbf123.bound(dim, 0, 3);
  pre_drbf456.bound(atom_j, 0, nij);
  pre_drbf456.bound(atom_k, 0, nij);
  pre_drbf456.bound(rbf, 0, nrbf);
  pre_drbf456.bound(dim, 0, 3);
  */
  
 
  Func pre_f("pre_f");
  pre_f(atom_j, atom_k, atom_i, rbf, abf, dim) =
    // pre_drbf(rbf, atom_k, atom_j, dim) * pre_abf(atom_i, atom_k, atom_j, abf) + pre_rbf(rbf, atom_k, atom_j) * pre_dabf(atom_i, atom_j, atom_k, abf, dim); // kj jk
    pre_drbf(rbf, atom_k, atom_j, dim) * pre_abf(atom_i, atom_j, atom_k, abf) + pre_rbf(rbf, atom_j, atom_k) * pre_dabf(atom_i, atom_k, atom_j, abf, dim); // kj kj
    // pre_drbf(rbf, atom_j, atom_k, dim) * pre_abf(atom_i, atom_k, atom_j, abf) + pre_rbf(rbf, atom_k, atom_j) * pre_dabf(atom_i, atom_j, atom_k, abf, dim); // jk jk
    // pre_drbf(rbf, atom_j, atom_k, dim) * pre_abf(atom_i, atom_k, atom_j, abf) + pre_rbf(rbf, atom_k, atom_j) * pre_dabf(atom_i, atom_k, atom_j, abf, dim); // jk kj

  pre_f.bound(atom_j, 0, nij);
  pre_f.bound(atom_k, 0, nij);
  // pre_f.bound(atom_i, 0, natom);
  pre_f.bound(atom_i, 0, nl * natom);
  pre_f.bound(rbf, 0, nrbf);
  pre_f.bound(abf, 0, nabf);
  pre_f.bound(dim, 0, 3);
  /*
  Func pre_fj("pre_fj");
  Func pre_fk("pre_fk");
  pre_fj(atom_k, atom_j, atom_i, rbf, abf, dim) = 
	  pre_drbf123(rbf, atom_j, atom_k, dim) * pre_abf(atom_i, atom_j, atom_k, abf) 
	  + pre_rbf(rbf, atom_j, atom_k) * pre_dabf123(atom_i, atom_j, atom_k, abf, dim);
  pre_fk(atom_k, atom_j, atom_i, rbf, abf, dim) =
	  pre_drbf456(rbf, atom_j, atom_k, dim) * pre_abf(atom_i, atom_j, atom_k, abf)
	  + pre_rbf(rbf, atom_j, atom_k) + pre_dabf456(atom_i, atom_j, atom_k, abf, dim);
  pre_fj.bound(atom_i, 0, nl * natom);
  pre_fj.bound(atom_j, 0, nij);
  pre_fj.bound(atom_k, 0, nij);
  pre_fj.bound(rbf, 0, nrbf);
  pre_fj.bound(abf, 0, nabf);
  pre_fj.bound(dim, 0, 3);
  pre_fk.bound(atom_i, 0, nl * natom);
  pre_fk.bound(atom_j, 0, nij);
  pre_fk.bound(atom_k, 0, nij);
  pre_fk.bound(rbf, 0, nrbf);
  pre_fk.bound(abf, 0, nabf);
  pre_fk.bound(dim, 0, 3);
  */
  



  Expr typei = clamp(atomtype(i) - 1, 0, nelems - 1);
  // Expr gj = clamp(pairlist(ljs), 0, natom - 1);
  Expr gj = clamp(pairlist(ljs), 0, nl * natom - 1);
  Expr j = clamp(alist(gj), 0, natom - 1);
  Expr typej = clamp(atomtype(j) - 1, 0, nelems - 1);
  // Expr gk = clamp(pairlist(lks), 0, natom - 1);
  Expr gk = clamp(pairlist(lks), 0, nl * natom - 1);
  Expr k = clamp(alist(gk), 0, natom - 1);
  Expr typek = clamp(atomtype(k) - 1, 0, nelems - 1);
  Expr interact = clamp(interaction(typek, typej) - 1, 0, nelementCombos - 1);
  eatom(i, interact, typei, m, p) += pre_rbf(m, ljs, lks) * pre_abf(i, ljs, lks, p);
  // eatom(i, interact, typei, m, p) += print(pre_rbf(m, ljs, lks) * pre_abf(i, ljs, lks, p), "<- eatom params: i: ", i, ", j: ", j, ", k: ", k, ", m: ", m, ", p: ", p, ", gk: ", gk, ", lks: ", lks);

  // fatom(dim, i, interact, typei, m, p) += print((pre_f(ljs, lks, i, m, p, dim) + pre_f(lks, ljs, i, m, p, dim)), "<- sum of ", ljs, lks);  
  fatom(dim, i, interact, typei, m, p) += pre_f(ljs, lks, i, m, p, dim) + pre_f(lks, ljs, i, m, p, dim);  
  // fatom(dim, i, interact, typei, m, p) = print(fatom(dim, i, interact, typei, m, p), "<-fatom +");
  // fatom(dim, k, interact, typei, m, p) -= print(pre_f(ljs, lks, i, m, p, dim), "<- pre_f ljs");
  fatom(dim, k, interact, typei, m, p) -= pre_f(ljs, lks, i, m, p, dim);
  // fatom(dim, j, interact, typei, m, p) = print(fatom(dim, j, interact, typei, m, p), "<- fatom -1");
  fatom(dim, j, interact, typei, m, p) -= pre_f(lks, ljs, i, m, p, dim);
  // fatom(dim, k, interact, typei, m, p) = print(fatom(dim, k, interact, typei, m, p), "<- fatom -2");
  /*
  fatom(dim, i, interact, typei, m, p) += pre_fj(ljs, lks, i, m, p, dim) + pre_fk(ljs, lks, i, m, p, dim);
  fatom(dim, j, interact, typei, m, p) -= pre_fj(ljs, lks, i, m, p, dim);
  fatom(dim, k, interact, typei, m, p) -= pre_fk(ljs, lks, i, m, p, dim);
 */
  // fatom.trace_stores();
  eatom.compute_root();

  fatom.compute_root();
  
  //  fatom(p, m, typei, interact, i, dim) += pre_f(ljs, lks, i, m, p, dim) + pre_f(lks, ljs, i, m, p, dim);

  //  fatom(p, m, typei, interact, j, dim) -= pre_f(ljs, lks, i, m, p, dim);

  //  fatom(p, m, typei, interact, k, dim) -= pre_f(lks, ljs, i, m, p, dim);


  //  fatom.update(0).reorder(dim, p, m, lk, lj, i);
  //  fatom.update(1).reorder(dim, p, m, lk, lj, i);
  //  fatom.update(2).reorder(dim, p, m, lk, lj, i);

}

void buildNeighPairs(Func & outputs, Func & vectors,
		     Func pairlist, Func pairnumsum, Func atomtype, Func alist, Func atompos,
		     Expr nl, Expr natom, Expr dim, Expr nmax, Expr npairs,
		     Var atom, Var d, Var nm, Var np, Var numOuts){
  
  outputs(np, numOuts) = mux(numOuts,{-1, -1, -1, -1});
  outputs.bound(np, 0, npairs);
  outputs.bound(numOuts, 0, 4);
  
  vectors(np, d) = Expr((double) 0.0);
  vectors.bound(d, 0, dim).reorder_storage(d, np);
  // vectors.bound(d, 0, dim);
  vectors.bound(np, 0, npairs);


  RDom r(0, natom, 0, npairs);
  r.where(r.y < pairnumsum(r.x + 1) && r.y >= pairnumsum(r.x));

  // Expr jacc = clamp(print(pairlist(r.y), "<- pairlist"), 0, print(npairs- 1, "<- npair - 1"));
  Expr jacc = clamp(pairlist(r.y), 0, nl * natom - 1);
  Expr att = clamp(alist(jacc), 0, natom - 1);  //
  Expr att_tt = atomtype(att);
  outputs(r.y, numOuts) = mux(numOuts, {r.x, att, atomtype(r.x), att_tt}); //ai[k], aj[k], ti, tj
  // outputs(r.y, numOuts) = mux(numOuts, {print(r.x, "<- ai:"), print(att, "<- aj:"), print(atomtype(r.x), "<- ti"), print(att_tt, "<- tj")}); //ai[k], aj[k], ti, tj
  // vectors(r.y, d) = atompos(print(jacc, "<- jacc", r.y, "<- r.y"), d) - atompos(print(r.x, "<- r.x"), d);
  vectors(r.y, d) = atompos(jacc, d) - atompos(r.x, d);
  // vectors(r.x, d) = atompos(jacc, d) - atompos(r.y, d);
  // vectors(r.y, d) = atompos(r.x, d) - atompos(jacc, d);
  // outputs.trace_stores();
  // atomtype.trace_loads();
  // atompos.trace_loads();
  // vectors.trace_stores();

  outputs.compute_root();
  vectors.compute_root();

  outputs.update(0).reorder(numOuts, r.y, r.x);
  vectors.update(0).reorder(d, r.y, r.x);
  
  //  ou// tputs.update(0).reorder(r.x, r.y, numOuts);
  //  vectors.update(0).reorder(r.x, r.y, d);
  
}


void buildPod1Body(Func & eatom, Func atomtype,
		   Expr nelements, Expr natom,
		   Var i, Var m){
  eatom(i, m) = select(atomtype(i) == m, (Expr((double)1.0)), (Expr((double)0.0)));
  eatom.bound(i, 0, natom);
  eatom.bound(m, 0, nelements);
}

void buildPod1Body_p(Func & eatom, Func & fatom,
		     Func atomtype,
		     Expr nelements, Expr natom,
		     Var i, Var m, Var dim){
    eatom(i, m) = select(atomtype(i) == m + 1, (Expr((double)1.0)), (Expr((double)0.0)));
    eatom.bound(i, 0, natom);
  eatom.bound(m, 0, nelements);
  fatom(i, m, dim) = (Expr((double)0.0));
  //  fatom.bound(i, 0, dim * natom * nelements);
  fatom.bound(i, 0, natom);
  fatom.bound(m, 0, nelements);
  fatom.bound(dim, 0, 3);
}



class pod1 : public Halide::Generator<pod1> {
public:
  //Func pairnumsum, Func pairlist,
  //				       Expr NPairs, Expr NAtoms, Expr NMax, Expr dim,
  //				       Func atomtype, Func alist, Func atompos
  
  Input<Buffer<int>> pairlist{"pairlist", 1};
  Input<Buffer<int>> pairnumsum{"pairnumsum", 1};
  Input<Buffer<int>> atomtype{"atomptype", 1};
  Input<Buffer<int>> alist{"alist", 1};
  Input<Buffer<double>> atompos{"atompos", 2};

  Output<Buffer<double>> rij{"rij", 2};
  Output<Buffer<int>> meta{"meta", 2};

  Pipeline pipeline;

  GeneratorParam<int> NMax{"NMax", 100};
  GeneratorParam<int> NTypes{"NTypes", 3};
  //  GeneratorParam<int> M{"M", 50};
  //  GeneratorParam<int> I{"I", 5000};
  


  void generate (){

    Expr NPairs = pairlist.dim(0).max();
    Expr NAtoms = atomtype.dim(0).max();
    alist.dim(0).set_bounds(0, NAtoms);
    atompos.dim(0).set_bounds(0, NAtoms);
    atompos.dim(1).set_bounds(0, 3);
    Func np1_data, np1_vecs;

    Var i("i"), j("j");

    //    rij(i, j)= np1_vecs(i, j);
    //    meta(i, j) = np1_data(i, j);

  }

};


class poddesc1 : public Halide::Generator<poddesc1> {
public:

  Input<Buffer<int>> pairlist{"pairlist", 1};
  Input<Buffer<int>> pairnumsum{"pairnumsum", 1};
  Input<Buffer<int>> atomtype{"atomtype", 1};
  Input<Buffer<int>> alist{"alist", 1};
  Input<Buffer<int>> interactions{"interactions", 2};
  Input<Buffer<double>> besselparams{"besselparams", 1};
  Input<Buffer<double>> Phi1{"Phi1", 3};
  Input<Buffer<double>> Phi2{"Phi2", 2};
  

  Input<Buffer<double>> y{"y", 2};

  Input<int> nl{"nl", 1};
  Input<int> npairs{"npairs", 1};
  Input<int> natom{"natom", 1};
  Input<int> bdegree{"bdegree", 1};
  Input<int> adegree{"adegree", 1};
  Input<int> adegreep{"adegreep", 1};
  Input<int> tdegree1{"tdegree1", 1};
  Input<int> tdegree2{"tdegree2", 1};
  Input<int> nbesselparams{"nbesselparams", 1};
  Input<int> nelems{"nelems", 1};
  Input<int> nelemscombos{"nelemsCombos", 1};
  
  
  Input<double> rin{"rin", 1};
  Input<double> rcut{"rcut", 1};



  GeneratorParam<int> nmaxp{"nmax", 100};

  // Output<Buffer<int>> ijs{"ijs", 2};
  // Output<Buffer<double>> rijs{"rijs", 2};

  // Output<Buffer<double>> rbf{"rbf", 3};
  // Output<Buffer<double>> drbf{"drbf", 4};
  // Output<Buffer<double>> abf{"abf", 2};
  // Output<Buffer<double>> dabf{"dabf", 3};

  // Output<Buffer<double>> energyij{"energyij", 2};
  // Output<Buffer<double>> forceij{"forceij", 3};

  Output<Buffer<double>> eatom1{"eatom1", 2};
  Output<Buffer<double>> fatom1{"fatom1", 3};
  
  Output<Buffer<double>> eatom2{"eatom2", 3};
  Output<Buffer<double>> fatom2{"fatom2", 4};
  
  Output<Buffer<double>> eatom3{"eatom3", 5};
  Output<Buffer<double>> fatom3{"fatom3", 6};

  void generate (){

    Var atom("atom");
    Var dim("dim");
    Var elem("elem");
    Var inter("inter");
    Var atom_o("atom_o");
    Var atom_i("atom_i");
    Var atom_j("atom_j");
    Var atom_k("atom_k");
    Var type("type");

    


    Var nm("nm");
    Var np("pairindex");
    Var numOuts("numOuts");

    Var bfi("basis function index");
    Var bfp("basis function param");
    Var bfa("inverse basis function index");
    Var rbf_v("rbf_v");

    Expr tdegree = max(tdegree1, tdegree2);

    
    Expr nmax = min(nmaxp, natom);
    pairlist.dim(0).set_bounds(0, npairs); // inum + nghost
    pairnumsum.dim(0).set_bounds(0, natom + 1); // inum + 1
    atomtype.dim(0).set_bounds(0, natom); // inum
    alist.dim(0).set_bounds(0, nl * natom);//are we sure? // inum + nghost
    // y.dim(0).set_bounds(0, npairs).set_stride(1);
    // y.dim(1).set_bounds(0, 3).set_stride(npairs);
    y.dim(0).set_bounds(0, nl * natom); // inum + nghost
    y.dim(1).set_bounds(0, 3);
    besselparams.dim(0).set_bounds(0, nbesselparams);
    Phi1.dim(0).set_bounds(0, nbesselparams);
    Phi1.dim(1).set_bounds(0, bdegree);
    Phi1.dim(2).set_bounds(0, tdegree); //tdegree1
    Phi2.dim(0).set_bounds(0, adegree);
    Phi2.dim(1).set_bounds(0, tdegree); //tdegree2
    interactions.dim(0).set_bounds(0, nelems);
    interactions.dim(1).set_bounds(0, nelems);
    
    

    Func ijs_f("ijs_f");
    Func rijs_f("rijs_f");
    
    buildNeighPairs(ijs_f, rijs_f,
		    pairlist, pairnumsum, atomtype, alist, y,
		    nl, natom, 3, nmax, npairs,
		    atom, dim, nm, np, numOuts);

    Func rbf_f("rbf_f"), drbf_f("drbf_f"), abf_f("abf_f"), dabf_f("dabf_f");
    buildRBF(rbf_f, drbf_f, abf_f, dabf_f,
	     rijs_f, besselparams, rin, rcut-rin,
	     bdegree, adegree, nbesselparams, npairs,
	     bfi, bfp, np, dim);

    Func energyij_f("energyij_f"), forceij_f("forceij_f");
    buildStructureMatMul(energyij_f, forceij_f,
			 rbf_f, abf_f, drbf_f, dabf_f, Phi1, Phi2,
			 bdegree, adegree, tdegree, nbesselparams, npairs,
			 bfi, bfa, bfp, np, dim);


    //copy intermediates

    Func eatom1_f("eatom1_f"), fatom1_f("fatom1_f");
    buildPod1Body_p(eatom1_f, fatom1_f,
		    atomtype,
		    nelems, natom,
		    atom, elem, dim);


    Func eatom2_f("eatom2_f"), fatom2_f("fatom2_f");
    buildPodTally2b(eatom2_f, fatom2_f,
		    energyij_f, forceij_f,
		    ijs_f, interactions,
		    npairs, natom,  nelems, nelemscombos, tdegree1,
		    np, atom, bfi, dim, elem, inter);


    Func eatom3_f("eatom3_f"), fatom3_f("fatom3_f");
    buildPodTally3b(eatom3_f, fatom3_f,
		    y, energyij_f, forceij_f, interactions,
		    pairlist, pairnumsum, atomtype, alist,
		    tdegree2, adegreep + 1, nelems, nelemscombos, nl, natom, npairs, nmax,
		    atom, atom_o, atom_i, atom_j, atom_k, inter, type,  bfa, rbf_v, dim);

    Var ox("ox"), oy("oy"), oz("oz"), ozz("ozz"), ozzz("ozzz"), ozzzz("ozzzz");
    
    // ijs(ox, oy) = ijs_f(ox, oy);
    // rijs(ox, oy) = rijs_f(ox, oy);
    // rbf(ox, oy, oz) = rbf_f(ox, oy, oz);
    // drbf(ox, oy, oz, ozz)= drbf_f(ox, oy, oz, ozz);
    // abf(ox, oy) = abf_f(ox, oy);
    // dabf(ox, oy, oz) = dabf_f(ox, oy, oz);
    // energyij(ox, oy) = energyij_f(ox, oy);
    // forceij(ox, oy, oz) = forceij_f(ox, oy, oz);
    
    eatom1(ox, oy) = eatom1_f(ox, oy);
    fatom1(ox, oy, oz) = fatom1_f(ox, oy, oz);

    eatom2(ox, oy, oz) = eatom2_f(ox, oy, oz);
    fatom2(ox, oy, oz, ozz) = fatom2_f(ox, oy, oz, ozz);
    // eatom2(ox, oy, oz) = Expr((double) 0);
    // fatom2(ox, oy, oz, ozz) = Expr((double) 0);

    eatom3(ox, oy, oz, ozz, ozzz) = eatom3_f(ox, oy, oz, ozzz, ozz);
    fatom3(ox, oy, oz, ozz, ozzz, ozzzz) = fatom3_f(ox, oy, oz, ozz, ozzzz, ozzz);
    // eatom3(ox, oy, oz, ozz, ozzz) = Expr((double) 0);
    // fatom3(ox, oy, oz, ozz, ozzz, ozzzz) = Expr((double) 0);
    
    // ijs.dim(0).set_bounds(0, natom);
    // ijs.dim(1).set_bounds(0, 4);
    // rijs.dim(0).set_bounds(0, natom);
    // rijs.dim(1).set_bounds(0, 3);
    // energyij.dim(0).set_bounds(0, tdegree); //tdegree
    // forceij.dim(0).set_bounds(0, tdegree);
    // energyij.dim(1).set_bounds(0, npairs);
    // forceij.dim(1).set_bounds(0, npairs);
    // forceij.dim(2).set_bounds(0, 3);
    
    eatom1.dim(0).set_bounds(0, natom);
    eatom1.dim(1).set_bounds(0, nelems);
    fatom1.dim(0).set_bounds(0, natom);
    fatom1.dim(1).set_bounds(0, nelems);
    fatom1.dim(2).set_bounds(0, 3);

    eatom2.dim(0).set_bounds(0, natom);
    eatom2.dim(1).set_bounds(0, nelemscombos);
    eatom2.dim(2).set_bounds(0, tdegree1);

    fatom2.dim(0).set_bounds(0, 3);
    fatom2.dim(1).set_bounds(0, natom);
    fatom2.dim(2).set_bounds(0, nelemscombos);
    fatom2.dim(3).set_bounds(0, tdegree1);



    eatom3.dim(0).set_bounds(0, natom);
    eatom3.dim(1).set_bounds(0, nelemscombos);
    eatom3.dim(2).set_bounds(0, nelems);
    eatom3.dim(3).set_bounds(0, adegreep + 1);
    eatom3.dim(4).set_bounds(0, tdegree2);

    fatom3.dim(0).set_bounds(0, 3);
    fatom3.dim(1).set_bounds(0, natom);
    fatom3.dim(2).set_bounds(0, nelemscombos);
    fatom3.dim(3).set_bounds(0, nelems);
    fatom3.dim(4).set_bounds(0, adegreep + 1);
    fatom3.dim(5).set_bounds(0, tdegree2);

      
    
    
  }


};


class snapshot : public Halide::Generator<snapshot> {
public:

  
  Input<Buffer<double>> xij{"xij", 1};
  Input<Buffer<double>> besselparams{"besselparams_buf", 1};
  Input<double> rin{"rin", 1};
  Input<double> rcut{"rcut", 1};
  Input<int> inversedegree_pre{"inversedegree", 1};
  Input<int> bessel_degree_pre{"radialdegree", 1};
  Input<int> nbssselpars{"besselparams", 1};
  Input<int> npairs{"npairs", 1};
  Output<Buffer<double>> rbf{"rbf", 3};


  void generate (){
    xij.dim(0).set_bounds(0, npairs);
    besselparams.dim(0).set_bounds(0, nbssselpars);
    //    Expr npairs = xij.dim(0).max();
    //    Expr nbseelparams = besselparams.dim(0).max();
    //    Expr totbesseldegree = rbf.dim(1).max();
    //    Expr nbesselpars = rbf.dim(2).max();
    //    Expr inversedegree = max(min(totbesseldegree, inversedegree_pre), 0);
    //    Expr radialdegree = max(min(totbesseldegree - inversedegree, totbesseldegree), 0);


    Func temp;
    buildSnap(temp, xij, besselparams, rin, rcut, bessel_degree_pre, inversedegree_pre, nbssselpars, npairs);
    Var a, b,c;
    rbf(a,b,c) = temp(a,b,c);
    rbf.dim(2).set_bounds(0, npairs);
    rbf.dim(1).set_bounds(0, max(inversedegree_pre, bessel_degree_pre));
    rbf.dim(0).set_bounds(0, nbssselpars + 1);
  }

};
  

HALIDE_REGISTER_GENERATOR(pod1, pod1);
HALIDE_REGISTER_GENERATOR(snapshot, snapshot);
HALIDE_REGISTER_GENERATOR(poddesc1, poddesc1);
