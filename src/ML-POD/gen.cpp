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

Expr get_abf_index(Expr original_index, Expr rbf_info_length) {
  return rbf_info_length + original_index;
}

void buildRBF(Func & rbft,
	      Func xij, Func besselparams, Expr rin, Expr rmax,
	      Expr bdegree, Expr adegree, Expr nbparams, Expr npairs,
	      Var bfi, Var bfp, Var np, Var dim, Expr ns)
{

  Expr one = Expr((double) 1.0);
  Expr zero = Expr((double) 0.0);
  Expr onefive = Expr((double) 1.5);
  Expr PI = Expr( (double)M_PI);
  Expr xij1 = xij(np, 0);
  Expr xij2 = xij(np, 1);
  Expr xij3 = xij(np, 2);

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

  Func rbf("rbf"), drbf("drbf_f"), abf("abf_f"), dabf("dabf_f");

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
  // rbf.size() = nbparams * bdegree * npairs
  // drbf[x].size() = nbparams * bdegree * npairs
  // abf.size() = adegree * npairs 
  // dabf[x].size() = adegree * npairs
  // output.size() = nbparams * bdgree * npairs + adegree * npairs
  Var rbf_abf_info("rbf_abf_info"), drbf_dabf_info("drbf_dabf_info");
  RDom r1(0, nbparams, 0, bdegree, 0, npairs);
  RDom r2(0, adegree, 0, npairs);

  Func rbf_f("rbf_f"), rbfx_f("rbfx_f"), rbfy_f("rbfy_f"), rbfz_f("rbfz_f");
  // Set up rbf_info
  rbf_f(rbf_abf_info) = zero;
  // rbf_f(r1.z * (nbparams * bdegree) + r1.y * (nbparams) + r1.x) = rbf(r1.x, r1.y, r1.z);
  rbf_f(r1.z + r1.y * (npairs) + r1.x * (npairs * bdegree)) = rbf(r1.x, r1.y, r1.z);

  // Set up abf_info
  Var abf_index("abf_index");
  Expr rbf_info_length = nbparams * bdegree * npairs;

  // rbf_f(get_abf_index(r2.y * adegree + r2.x)) = abf(r2.x, r2.y);
  rbf_f(get_abf_index(r2.y + r2.x * (npairs), rbf_info_length)) = abf(r2.x, r2.y);

  rbf_f.bound(rbf_abf_info, 0, nbparams * bdegree * npairs + adegree * npairs);
  

  // Do the same for rbfx_f
  // Set up drbf_dabf_info
  rbfx_f(drbf_dabf_info) = zero;
  // rbfx_f(r1.z * (nbparams * bdegree) + r1.y * (nbparams) + r1.x) = drbf(r1.x, r1.y, r1.z, 0);
  rbfx_f(r1.z + r1.y * (npairs) + r1.x * (npairs * bdegree)) = drbf(r1.x, r1.y, r1.z, 0);

  // Set up dabf_info
  // rbfx_f(get_abf_index(r2.y * adegree + r2.x)) = dabf(r2.x, r2.y, 0);
  rbfx_f(get_abf_index(r2.y + r2.x * (npairs), rbf_info_length)) = dabf(r2.x, r2.y, 0);

  rbfx_f.bound(drbf_dabf_info, 0, nbparams * bdegree * npairs + adegree * npairs);

  // Do the same for rbfy_f
  // Set up drbf_dabf_info
  rbfy_f(drbf_dabf_info) = zero;
  // rbfy_f(r1.z * (nbparams * bdegree) + r1.y * (nbparams) + r1.x) = drbf(r1.x, r1.y, r1.z, 1);
  rbfy_f(r1.z + r1.y * (npairs) + r1.x * (npairs * bdegree)) = drbf(r1.x, r1.y, r1.z, 1);

  // Set up dabf_info
  // rbfy_f(get_abf_index(r2.y * adegree + r2.x)) = dabf(r2.x, r2.y, 1);
  rbfy_f(get_abf_index(r2.y + r2.x * (npairs), rbf_info_length)) = dabf(r2.x, r2.y, 1);

  rbfy_f.bound(drbf_dabf_info, 0, nbparams * bdegree * npairs + adegree * npairs);

  // Do the same for rbfz_f
  // Set up drbf_dabf_info
  rbfz_f(drbf_dabf_info) = zero;
  // rbfz_f(r1.z * (nbparams * bdegree) + r1.y * (nbparams) + r1.x) = drbf(r1.x, r1.y, r1.z, 2);
  rbfz_f(r1.z + r1.y * (npairs) + r1.x * (npairs * bdegree)) = drbf(r1.x, r1.y, r1.z, 2);

  // Set up dabf_info
  // rbfz_f(get_abf_index(r2.y * adegree + r2.x)) = dabf(r2.x, r2.y, 2);
  rbfz_f(get_abf_index(r2.y + r2.x * (npairs), rbf_info_length)) = dabf(r2.x, r2.y, 2);
  
  rbfz_f.bound(drbf_dabf_info, 0, nbparams * bdegree * npairs + adegree * npairs);


  Var nps("np"), n("n"), c1("c1");
  rbft(nps, n, c1) = zero;
  RDom t(0, npairs, 0, ns);
  rbft(t.x, t.y, 0) = rbf_f(t.y + t.x * npairs);
  rbft(t.x, t.y, 1) = rbfx_f(t.y + t.x * npairs);
  rbft(t.x, t.y, 2) = rbfy_f(t.y + t.x * npairs);
  rbft(t.x, t.y, 3) = rbfz_f(t.y + t.x * npairs);

  rbft.bound(nps, 0, npairs);
  rbft.bound(n, 0, ns);
  rbft.bound(c1, 0, 4);

  rbft.compute_root();
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


void radialAngularBasis(Func & sumU, Func & U,
        Func rbf, Func abf,
        Func atomtype, Expr N, Expr K, Expr M, Expr Nen)
{
    Expr zero = Expr((double) 0.0);

    Var k("k"), m("m"), n("n"), c("c");
    Var ne("ne");
    sumU(m, k, ne) = zero;

    //Expr c1 = rbf(m, n);
    //Expr c2 = abf(k, n);
    
    // U(m, k, n) = c1 * c2;
    // Ux(m, k, n) = abfx(k, n) * c1 + c2 * rbfx(m, n);
    // Uy(m, k, n) = abfy(k, n) * c1 + c2 * rbfy(m, n);
    // Uz(m, k, n) = abfz(k, n) * c1 + c2 * rbfz(m, n);
    U(m, k, n, c) = zero;

    RDom r(0, M, 0, K, 0, N, 1, 4);
    Expr in = atomtype(r.z) - 1;

    Expr c1 = rbf(r.z, r.x, 0);
    Expr c2 = abf(r.z, r.y, 0);

    U(r.x, r.y, r.z, 0) = c1 * c2;
    U(r.x, r.y, r.z, r[3]) = abf(r.z, r.y, r[3]) * c1 + c2 * rbf(r.z, r.x, r[3]);
    // sumU(r.x, r.y, clamp(in, 0, Ne - 1)) += rbf(r.x, r.z) * abf(r.y, r.z);
    sumU(r.x, r.y, clamp(in, 0, Nen - 1)) += rbf(r.z, r.x, 0) * abf(r.z, r.y, 0);

    sumU.bound(m, 0, M);
    U.bound(m, 0, M);

    sumU.bound(k, 0, K);
    U.bound(k, 0, K);

    sumU.bound(ne, 0, Nen);
    U.bound(n, 0, N);

    U.bound(c, 0, 4);
}

void twoBodyDescDeriv(Func & d2, Func & dd2, Func rbf, Func tj, Expr N, Expr Ne, Expr nrbf2)
{
    Expr zero = Expr((double) 0.0);

    Var ne("ne"), m("m"), n("n"), dim("dim");
    d2(ne, m) = zero;
    dd2(ne, m, n, dim) = zero;

    RDom r(0, N, 0, nrbf2, 0, 3);

    d2(clamp(tj(r.x)-1, 0, Ne - 1), r.y) += rbf(r.x, r.y, 0);
    dd2(clamp(tj(r.x)-1, 0, Ne - 1), r.y, r.x, r.z) += rbf(r.x, r.y, r.z + 1);

    d2.bound(ne, 0, Ne);
    d2.bound(m, 0, nrbf2);

    dd2.bound(ne, 0, Ne);
    dd2.bound(m, 0, nrbf2);
    dd2.bound(n, 0, N);
    dd2.bound(dim, 0, 3);
    
}

//void tallyTwoBodyLocalForce(Func & fij, Func & e, Func coeff2, Func rbf, Func rbfx, Func rbfy, Func rbfz, Func tj, Expr nbf, Expr N)
void tallyTwoBodyLocalForce(Func & fij, Func & e, Func coeff2, Func rbf, Func tj, Expr nbf, Expr N)
{
    Expr zero = Expr((double) 0.0);

    Var n("n"), m("m"), dim("dim"), empty("empty");
    e() = zero;
    fij(n, dim) = zero;

    RDom r(0, N, 0, nbf, 0, 3);

    Expr c = coeff2(clamp(tj(r.x), 1, N - 1) - 1, r.y);
    e() += c * rbf(r.x, r.y, 0);
    fij(r.x, r.z) += c * rbf(r.x, r.y, r.z + 1);

    fij.bound(n, 0, N);
    fij.bound(dim, 0, 3);
}

/*
class poddescTallyTwoBodyLocalForce : public Halide::Generator<poddescTallyTwoBodyLocalForce> {
public:
    Output<Buffer<double>> fij_o{"fij_o", 2};
    Output<double> e_o{"e_o"};

    Input<Buffer<double>> coeff2{"coeff2", 2};
    Input<Buffer<double>> rbf{"rbf", 2};
    Input<Buffer<double>> rbfx{"rbfx", 2};
    Input<Buffer<double>> rbfy{"rbfy", 2};
    Input<Buffer<double>> rbfz{"rbfz", 2};

    Input<Buffer<int>> tj{"tj", 1};

    Input<int> nbf{"nbf", 1};
    Input<int> N{"N", 1};
    Input<int> ns{"ns", 1};

    void generate() {
        rbf.dim(0).set_bounds(0, N).set_stride(1);
        rbf.dim(1).set_bounds(0, ns).set_stride(N);
        rbfx.dim(0).set_bounds(0, N).set_stride(1);
        rbfx.dim(1).set_bounds(0, ns).set_stride(N);
        rbfy.dim(0).set_bounds(0, N).set_stride(1);
        rbfy.dim(1).set_bounds(0, ns).set_stride(N);
        rbfz.dim(0).set_bounds(0, N).set_stride(1);
        rbfz.dim(1).set_bounds(0, ns).set_stride(N);

        coeff2.dim(0).set_bounds(0, N).set_stride(nbf);
        coeff2.dim(1).set_bounds(0, nbf).set_stride(1);

        Func fij("fij"), e("e");
        tallyTwoBodyLocalForce(fij, e, coeff2, rbf, rbfx, rbfy, rbfz, tj, nbf, N);

        Var n("n"), dim("dim");
        fij_o(n, dim) = fij(n, dim);
        e_o() = e();

        fij_o.dim(0).set_bounds(0, N).set_stride(3);
        fij_o.dim(1).set_bounds(0, 3).set_stride(1);
    }
};



class poddescTwoBodyDescDeriv : public Halide::Generator<poddescTwoBodyDescDeriv> {
public:
    Output<Buffer<double>> d2_o{"d2_o", 2};
    Output<Buffer<double>> dd2_o{"dd2_o", 4};

    Input<Buffer<double>> rbf{"rbf", 2};
    Input<Buffer<double>> rbfx{"rbfx", 2};
    Input<Buffer<double>> rbfy{"rbfy", 2};
    Input<Buffer<double>> rbfz{"rbfz", 2};

    Input<Buffer<int>> tj{"tj", 1};
    
    Input<int> N{"N", 1};
    Input<int> Ne{"Ne", 1};
    Input<int> nrbf2{"nrbf2", 1};
    Input<int> ns{"ns", 1};

    void generate() {
        rbf.dim(0).set_bounds(0, N).set_stride(1);
        rbf.dim(1).set_bounds(0, ns).set_stride(N);
        rbfx.dim(0).set_bounds(0, N).set_stride(1);
        rbfx.dim(1).set_bounds(0, ns).set_stride(N);
        rbfy.dim(0).set_bounds(0, N).set_stride(1);
        rbfy.dim(1).set_bounds(0, ns).set_stride(N);
        rbfz.dim(0).set_bounds(0, N).set_stride(1);
        rbfz.dim(1).set_bounds(0, ns).set_stride(N);

        Func d2("d2"), dd2("dd2");
        twoBodyDescDeriv(d2, dd2, rbf, rbfx, rbfy, rbfz, tj, N, Ne, nrbf2);

        Var ne("ne"), m("m"), n("n"), dim("dim");
        d2_o(ne, m) = d2(ne, m);
        dd2_o(ne, m, n, dim) = dd2(ne, m, n, dim);

        d2_o.dim(0).set_bounds(0, Ne).set_stride(nrbf2);
        d2_o.dim(1).set_bounds(0, nrbf2).set_stride(1);

        dd2_o.dim(0).set_bounds(0, Ne).set_stride(3 * N * nrbf2);
        dd2_o.dim(1).set_bounds(0, nrbf2).set_stride(3 * N);
        dd2_o.dim(2).set_bounds(0, N).set_stride(3);
        dd2_o.dim(3).set_bounds(0, 3).set_stride(1);
    }
};


class poddescRadialAngularBasis : public Halide::Generator<poddescRadialAngularBasis> {
public:

    Output<Buffer<double>> sumU_o{"sumU", 3};
    Output<Buffer<double>> U_o{"U", 3};
    Output<Buffer<double>> Ux_o{"Ux", 3};
    Output<Buffer<double>> Uy_o{"Uy", 3};
    Output<Buffer<double>> Uz_o{"Uz", 3};

    Input<Buffer<double>> rbf{"rbf", 2};
    Input<Buffer<double>> rbfx{"rbfx", 2};
    Input<Buffer<double>> rbfy{"rbfy", 2};
    Input<Buffer<double>> rbfz{"rbfz", 2};
    Input<Buffer<double>> abf{"abf", 2};
    Input<Buffer<double>> abfx{"abfx", 2};
    Input<Buffer<double>> abfy{"abfy", 2};
    Input<Buffer<double>> abfz{"abfz", 2};
    Input<Buffer<int>> tj{"tj", 1};

    Input<int> Nj{"Nj", 1};
    Input<int> K3{"K3", 1};
    Input<int> nrbf3{"nrbf3", 1};
    Input<int> nelements{"nelements", 1};
    Input<int> ns{"ns", 1};

    void generate() {
        rbf.dim(0).set_bounds(0, Nj).set_stride(1);
        rbf.dim(1).set_bounds(0, ns).set_stride(Nj);
        rbfx.dim(0).set_bounds(0, Nj).set_stride(1);
        rbfx.dim(1).set_bounds(0, ns).set_stride(Nj);
        rbfy.dim(0).set_bounds(0, Nj).set_stride(1);
        rbfy.dim(1).set_bounds(0, ns).set_stride(Nj);
        rbfz.dim(0).set_bounds(0, Nj).set_stride(1);
        rbfz.dim(1).set_bounds(0, ns).set_stride(Nj);

        abf.dim(0).set_bounds(0, Nj).set_stride(1);
        abf.dim(1).set_bounds(0, K3).set_stride(Nj);
        abfx.dim(0).set_bounds(0, Nj).set_stride(1);
        abfx.dim(1).set_bounds(0, K3).set_stride(Nj);
        abfy.dim(0).set_bounds(0, Nj).set_stride(1);
        abfy.dim(1).set_bounds(0, K3).set_stride(Nj);
        abfz.dim(0).set_bounds(0, Nj).set_stride(1);
        abfz.dim(1).set_bounds(0, K3).set_stride(Nj);


        Func sumU("sumU"), U("U"), Ux("Ux"), Uy("Uy"), Uz("Uz");
        radialAngularBasis(sumU, U, Ux, Uy, Uz,
                rbf, rbfx, rbfy, rbfz,
                abf, abfx, abfy, abfz,
                tj, Nj, K3, nrbf3, nelements);

        Var m("m"), k("k"), n("n"), ne("ne");

        //sumU_o(m, k, ne) = sumU(m, k, ne);
        //U_o(m, k, n) = U(m, k, n);
        //Ux_o(m, k, n) = Ux(m, k, n);
        //Uy_o(m, k, n) = Uy(m, k, n);
        //Uz_o(m, k, n) = Uz(m, k, n);
        sumU_o(ne, k, m) = sumU(m, k, ne);
        U_o(n, k, m) = U(m, k, n);
        Ux_o(n, k, m) = Ux(m, k, n);
        Uy_o(n, k, m) = Uy(m, k, n);
        Uz_o(n, k, m) = Uz(m, k, n);

        sumU_o.dim(0).set_bounds(0, nelements).set_stride(1);
        sumU_o.dim(1).set_bounds(0, K3).set_stride(nelements);
        sumU_o.dim(2).set_bounds(0, nrbf3).set_stride(nelements * K3);
        U_o.dim(0).set_bounds(0, Nj).set_stride(1);
        U_o.dim(1).set_bounds(0, K3).set_stride(Nj);
        U_o.dim(2).set_bounds(0, nrbf3).set_stride(Nj * K3);
        Ux_o.dim(0).set_bounds(0, Nj).set_stride(1);
        Ux_o.dim(1).set_bounds(0, K3).set_stride(Nj);
        Ux_o.dim(2).set_bounds(0, nrbf3).set_stride(Nj * K3);
        Uy_o.dim(0).set_bounds(0, Nj).set_stride(1);
        Uy_o.dim(1).set_bounds(0, K3).set_stride(Nj);
        Uy_o.dim(2).set_bounds(0, nrbf3).set_stride(Nj * K3);
        Uz_o.dim(0).set_bounds(0, Nj).set_stride(1);
        Uz_o.dim(1).set_bounds(0, K3).set_stride(Nj);
        Uz_o.dim(2).set_bounds(0, nrbf3).set_stride(Nj * K3);
    }
};


class poddescRBF : public Halide::Generator<poddescRBF> {
public:

    Output<Buffer<double>> rbf_o{"rbf_f", 1};
    Output<Buffer<double>> rbfxf_o{"rbfx_f", 1};
    Output<Buffer<double>> rbfyf_o{"rbfy_f", 1};
    Output<Buffer<double>> rbfzf_o{"rbfz_f", 1};
    
    Input<Buffer<double>> rijs{"rijs", 2};
    Input<Buffer<double>> besselparams{"besselparams", 1};
    Input<int> nbesselparams{"nbesselpars", 1};
    Input<int> bdegree{"bdegree", 1};
    Input<int> adegree{"adegree", 1};
    Input<int> npairs{"npairs", 1};

    Input<double> rin{"rin", 1};
    Input<double> rcut{"rcut", 1};

    void generate() {
        rijs.dim(0).set_bounds(0, npairs).set_stride(3);
        rijs.dim(1).set_bounds(0, 3).set_stride(1);


        besselparams.dim(0).set_bounds(0, nbesselparams);
        Var bfi("basis function index");
        Var bfp("basis function param");
        Var np("pairindex");
        Var numOuts("numOuts");
        Var dim("dim");

        Func rbf_f("rbf_f"), rbfx_f("rbfx_f"), rbfy_f("rbfy_f"), rbfz_f("rbfz_f");
        buildRBF(rbf_f, rbfx_f, rbfy_f, rbfz_f,
             rijs, besselparams, rin, rcut-rin,
             bdegree, adegree, nbesselparams, npairs,
             bfi, bfp, np, dim);

        Var rbf_output("rbf_output");

        rbf_o(rbf_output) = rbf_f(rbf_output);
        rbfxf_o(rbf_output) = rbfx_f(rbf_output);
        rbfyf_o(rbf_output) = rbfy_f(rbf_output);
        rbfzf_o(rbf_output) = rbfz_f(rbf_output);
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

class poddescFourMult : public Halide::Generator<poddescFourMult> {
public:


  Input<int> npairs{"npairs", 1};
  Input<int> nrbfmax{"nrbfmax", 1};
  Input<int> ns{"ns", 1};
  Input<Buffer<double>> rbf4_in{"rbf_in", 3};
  Input<Buffer<double>> Phi{"Phi", 2};

  Output<Buffer<double>> rbf4_o{"rbf_o", 3};
  
    void generate() {
      Phi.dim(0).set_bounds(0, ns).set_stride(1);
      Phi.dim(1).set_bounds(0, ns).set_stride(ns);

      rbf4_in.dim(2).set_bounds(0, 4).set_stride(npairs * ns);
      rbf4_in.dim(1).set_bounds(0, ns).set_stride(npairs);
      rbf4_in.dim(0).set_bounds(0, npairs).set_stride(1);
      Var i("i");
      Var j("j");
      Var k("k");
      Var c("c");
      Func prod("prod");
      prod(c, k, i, j) = Phi(k, i) * rbf4_in(j, k, c);
      prod.bound(c, 0, 4);
      prod.bound(k, 0, nrbfmax);
      prod.bound(j, 0, npairs);
      prod.bound(i, 0, npairs);
      rbf4_o(j, i, c) = Expr((double) 0.0);
      RDom r(0, ns);
      rbf4_o(j, i, c) += prod(c, r, i, j);

      rbf4_o.dim(2).set_bounds(0, 4).set_stride(nrbfmax * npairs);
      rbf4_o.dim(1).set_bounds(0, nrbfmax).set_stride(npairs);
      rbf4_o.dim(0).set_bounds(0, npairs).set_stride(1);

    }
};


class poddescAngularBasis : public Halide::Generator<poddescAngularBasis> {
public:


  Input<int> npairs{"npairs", 1};
  Input<int> k3{"k3", 1};
  Input<Buffer<int>> pq{"pq", 1};
  Input<Buffer<double>> rij{"rij", 2};
  Output<Buffer<double>> abf4{"abf", 3};
  
    
    void generate() {
      pq.dim(0).set_bounds(0, 3* k3).set_stride(1);
      rij.dim(0).set_bounds(0, 3).set_stride(1);
      rij.dim(1).set_bounds(0, npairs).set_stride(3);

      Var c("c");
      Var pair("pair");
      Var abfi("abfi");
      
      Expr x = rij(0, pair);
      Expr y = rij(1, pair);
      Expr z = rij(2, pair);
      
      Expr xx = x*x;
      Expr yy = y*y;
      Expr zz = z*z;
      Expr xy = x*y;
      Expr xz = x*z;
      Expr yz = y*z;

      Expr dij = sqrt(xx + yy + zz);
      Expr u = x/dij;
      Expr v = y/dij;
      Expr w = z/dij;
    
      Expr dij3 = dij*dij*dij;
      Expr dudx = (yy+zz)/dij3;
      Expr dudy = -xy/dij3;
      Expr dudz = -xz/dij3;

      Expr dvdx = -xy/dij3;
      Expr dvdy = (xx+zz)/dij3;
      Expr dvdz = -yz/dij3;

      Expr dwdx = -xz/dij3;
      Expr dwdy = -yz/dij3;
      Expr dwdz = (xx+yy)/dij3;

      Func tm("tm");
      Var abfip("abfip");
      Expr zero = Expr((double) 0.0);
      tm(pair, abfi, abfip, c) = zero;
      tm(pair, abfi, 0, 0) = Expr((double) 1.0);
      RDom rn(1, k3 + 1, 0, 4);
      Expr m = clamp(pq(rn.x) - 1, 0, 3 * k3 - 1);
      Expr prev0 = tm(pair, abfi, m, 0);
      Expr d = pq(rn.x + k3);
      Expr uvw = select(d == 1, u, select(d==2, v, select(d==3, w, Expr((double) 0.0))));
      tm(pair, abfi, rn.x, rn.y) = tm(pair, abfi, m, rn.y) * uvw + select(d == rn.y, tm(pair, abfi, m, 0), zero);
      abf4(pair, abfi, c) = zero;
      abf4(pair, abfi, 0) = tm(pair, abfi, abfi, 0);
      abf4(pair, abfi, 1) = tm(pair, abfi, abfi, 1) * dudx + tm(pair, abfi, abfi, 2) * dvdx + tm(pair, abfi, abfi, 3) * dwdx;
      abf4(pair, abfi, 2) = tm(pair, abfi, abfi, 1) * dudy + tm(pair, abfi, abfi, 2) * dvdy + tm(pair, abfi, abfi, 3) * dwdy;
      abf4(pair, abfi, 3) = tm(pair, abfi, abfi, 1) * dudz + tm(pair, abfi, abfi, 2) * dvdz + tm(pair, abfi, abfi, 3) * dwdz;
      
      
      abf4.dim(2).set_bounds(0, 4).set_stride(k3* npairs);
      abf4.dim(1).set_bounds(0, k3).set_stride(npairs);
      abf4.dim(0).set_bounds(0, npairs).set_stride(1);
      
    }
};

*/
class poddescTwoBody: public Halide::Generator<poddescTwoBody> {
public:

    Input<Buffer<double>> rijs{"rijs", 2};
    Input<Buffer<double>> besselparams{"besselparams", 1};
    Input<int> nbesselparams{"nbesselpars", 1};
    Input<int> bdegree{"bdegree", 1};
    Input<int> adegree{"adegree", 1};
    Input<int> npairs{"npairs", 1};
    Input<int> nrbfmax{"nrbfmax", 1};
    Input<double> rin{"rin", 1};
    Input<double> rcut{"rcut", 1};

    Input<Buffer<double>> Phi{"Phi", 2};
    Input<int> ns{"ns", 1};

    Input<Buffer<double>> coeff2{"coeff2", 2};
    Input<Buffer<int>> tj{"tj", 1};
    Input<int> nrbf2{"nrbf2", 1};
    Output<Buffer<double>> fij_o{"fij_o", 2};
    Output<double> e_o{"e_o"};

    Input<int> k3{"k3", 1};
    Input<Buffer<int>> pq{"pq", 1};

    Input<int> nrbf3{"nrbf3", 1};
    Input<int> nelements{"nelements", 1};
    Output<Buffer<double>> sumU_o{"sumU_o", 3};
    Output<Buffer<double>> U_o{"U_o", 4};

    Output<Buffer<double>> d2_o{"d2_o", 2};
    Output<Buffer<double>> dd2_o{"dd2_o", 4};

    void generate() {
        rijs.dim(0).set_bounds(0, npairs).set_stride(3);
        rijs.dim(1).set_bounds(0, 3).set_stride(1);

        besselparams.dim(0).set_bounds(0, nbesselparams);
        Var bfi("basis function index");
        Var bfp("basis function param");
        Var np("pairindex");
        Var numOuts("numOuts");
        Var dim("dim");

        Func rbft("rbft");
        buildRBF(rbft, rijs, besselparams, rin, rcut-rin,
                bdegree, adegree, nbesselparams, npairs,
                bfi, bfp, np, dim, ns);

        // MatMul
        Phi.dim(0).set_bounds(0, ns).set_stride(1);
        Phi.dim(1).set_bounds(0, ns).set_stride(ns);

        //rbft.dim(2).set_bounds(0, 4).set_stride(npairs * ns);
        //rbft.dim(1).set_bounds(0, ns).set_stride(npairs);
        //rbft.dim(0).set_bounds(0, npairs).set_stride(1);
        Var i("i");
        Var j("j");
        Var k("k");
        Var c("c");
        Func prod("prod");
        prod(c, k, i, j) = Phi(k, i) * rbft(j, k, c);
        prod.bound(c, 0, 4);
        prod.bound(k, 0, nrbfmax);
        prod.bound(j, 0, npairs);
        prod.bound(i, 0, npairs);
        Func rbf("rbf");
        rbf(j, i, c) = Expr((double) 0.0);
        RDom r(0, ns);
        rbf(j, i, c) += prod(c, r, i, j);

        //rbf.dim(2).set_bounds(0, 4).set_stride(nrbfmax * npairs);
        //rbf.dim(1).set_bounds(0, nrbfmax).set_stride(npairs);
        //rbf.dim(0).set_bounds(0, npairs).set_stride(1);
        // end MatMul

        coeff2.dim(0).set_bounds(0, npairs).set_stride(nrbf2);
        coeff2.dim(1).set_bounds(0, nrbf2).set_stride(1);

        Func fij("fij"), e("e");
        tallyTwoBodyLocalForce(fij, e, coeff2, rbf, tj, nrbf2, npairs);

        Var n("n");
        fij_o(n, dim) = fij(n, dim);
        e_o() = e();

        fij_o.dim(0).set_bounds(0, npairs).set_stride(3);
        fij_o.dim(1).set_bounds(0, 3).set_stride(1);
        // if nd3 > 0
        pq.dim(0).set_bounds(0, 3* k3).set_stride(1);

        Var pair("pair");
        Var abfi("abfi");
          
        Expr x = rijs(pair, 0);
        Expr y = rijs(pair, 0);
        Expr z = rijs(pair, 0);
        
        Expr xx = x*x;
        Expr yy = y*y;
        Expr zz = z*z;
        Expr xy = x*y;
        Expr xz = x*z;
        Expr yz = y*z;

        Expr dij = sqrt(xx + yy + zz);
        Expr u = x/dij;
        Expr v = y/dij;
        Expr w = z/dij;
        
        Expr dij3 = dij*dij*dij;
        Expr dudx = (yy+zz)/dij3;
        Expr dudy = -xy/dij3;
        Expr dudz = -xz/dij3;

        Expr dvdx = -xy/dij3;
        Expr dvdy = (xx+zz)/dij3;
        Expr dvdz = -yz/dij3;

        Expr dwdx = -xz/dij3;
        Expr dwdy = -yz/dij3;
        Expr dwdz = (xx+yy)/dij3;

        Func tm("tm");
        Var abfip("abfip");
        Expr zero = Expr((double) 0.0);
        tm(pair, abfi, abfip, c) = zero;
        tm(pair, abfi, 0, 0) = Expr((double) 1.0);
        RDom rn(1, k3 + 1, 0, 4);
        Expr m1 = clamp(pq(rn.x) - 1, 0, 3 * k3 - 1);
        Expr prev0 = tm(pair, abfi, m1, 0);
        Expr d = pq(rn.x + k3);
        Expr uvw = select(d == 1, u, select(d==2, v, select(d==3, w, Expr((double) 0.0))));
        tm(pair, abfi, rn.x, rn.y) = tm(pair, abfi, m1, rn.y) * uvw + select(d == rn.y, tm(pair, abfi, m1, 0), zero);
        Func abf4("abf4");
        abf4(pair, abfi, c) = zero;
        abf4(pair, abfi, 0) = tm(pair, abfi, abfi, 0);
          abf4(pair, abfi, 1) = tm(pair, abfi, abfi, 1) * dudx + tm(pair, abfi, abfi, 2) * dvdx + tm(pair, abfi, abfi, 3) * dwdx;
          abf4(pair, abfi, 2) = tm(pair, abfi, abfi, 1) * dudy + tm(pair, abfi, abfi, 2) * dvdy + tm(pair, abfi, abfi, 3) * dwdy;
          abf4(pair, abfi, 3) = tm(pair, abfi, abfi, 1) * dudz + tm(pair, abfi, abfi, 2) * dvdz + tm(pair, abfi, abfi, 3) * dwdz;
          
          
        //abf4.dim(2).set_bounds(0, 4).set_stride(k3* npairs);
        //abf4.dim(1).set_bounds(0, k3).set_stride(npairs);
        //abf4.dim(0).set_bounds(0, npairs).set_stride(1);
        // end angular basis

        Func sumU("sumU"), U("U");
        radialAngularBasis(sumU, U, rbf, abf4,
                tj, npairs, k3, nrbf3, nelements);

        Var ne("ne"), m("m");
        sumU_o(ne, k, m) = sumU(m, k, ne);
        U_o(n, k, m, c) = U(m, k, n, c);

        sumU_o.dim(0).set_bounds(0, nelements).set_stride(1);
        sumU_o.dim(1).set_bounds(0, k3).set_stride(nelements);
        sumU_o.dim(2).set_bounds(0, nrbf3).set_stride(nelements * k3);
        U_o.dim(0).set_bounds(0, npairs).set_stride(1);
        U_o.dim(1).set_bounds(0, k3).set_stride(npairs);
        U_o.dim(2).set_bounds(0, nrbf3).set_stride(npairs * k3);
        U_o.dim(3).set_bounds(0, 4).set_stride(npairs * k3 * nrbf3);
        // if nd23 > 0
        Func d2("d2"), dd2("dd2");
        twoBodyDescDeriv(d2, dd2, rbf, tj, npairs, nelements, nrbf2);

        d2_o(ne, m) = d2(ne, m);
        dd2_o(ne, m, n, dim) = dd2(ne, m, n, dim);

        d2_o.dim(0).set_bounds(0, nelements).set_stride(nrbf2);
        d2_o.dim(1).set_bounds(0, nrbf2).set_stride(1);

        dd2_o.dim(0).set_bounds(0, nelements).set_stride(3 * npairs * nrbf2);
        dd2_o.dim(1).set_bounds(0, nrbf2).set_stride(3 * npairs);
        dd2_o.dim(2).set_bounds(0, npairs).set_stride(3);
        dd2_o.dim(3).set_bounds(0, 3).set_stride(1);
    }
};

/*
HALIDE_REGISTER_GENERATOR(pod1, pod1);
HALIDE_REGISTER_GENERATOR(snapshot, snapshot);
HALIDE_REGISTER_GENERATOR(poddescRBF, poddescRBF);
HALIDE_REGISTER_GENERATOR(poddescRadialAngularBasis, poddescRadialAngularBasis);
HALIDE_REGISTER_GENERATOR(poddescTwoBodyDescDeriv, poddescTwoBodyDescDeriv);
HALIDE_REGISTER_GENERATOR(poddescTallyTwoBodyLocalForce, poddescTallyTwoBodyLocalForce);
HALIDE_REGISTER_GENERATOR(poddescFourMult, poddescFourMult);
HALIDE_REGISTER_GENERATOR(poddescAngularBasis, poddescAngularBasis);
*/
HALIDE_REGISTER_GENERATOR(poddescTwoBody, poddescTwoBody);
