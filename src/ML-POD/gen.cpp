#include "Halide.h"
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h> 
using namespace Halide;

Expr get_abf_index(Expr original_index, Expr rbf_info_length) {
  return rbf_info_length + original_index;
}

void buildRBF( Func & rbfall,
	       Func xij, Func dij, Func idij, Func fcut, Func dfcut, Func besselparams, Expr rin, Expr rmax,
	       Expr bdegree, Expr adegree, Expr nbparams, Expr npairs, Expr ns,
	       Var bfi, Var bfp, Var np, Var dim)
{

  Expr one = Expr((double) 1.0);
  Expr zero = Expr((double) 0.0);
  Expr onefive = Expr((double) 1.5);
  Expr PI = Expr( (double)M_PI);
  Expr xij1 = xij(0, np);
  Expr xij2 = xij(1, np);
  Expr xij3 = xij(2, np);

  Expr s = xij1*xij1 + xij2*xij2 + xij3*xij3;
  dij(np) = sqrt(s);
  idij(np) = one / dij(np); 
  Expr dr1 = xij1 * idij(np);
  Expr dr2 = xij2 * idij(np);
  Expr dr3 = xij3 * idij(np);
  Expr irmax = one / rmax;

  Func y7("y7");
  Expr r = dij(np) - rin;
  Expr y = r * irmax;    
  Expr y2 = y*y;
  Expr y3 = one - y2*y;
  Expr y4 = y3*y3 + Expr((double) 1e-6);
  Expr y5 = sqrt(y4); //pow(y4, 0.5);
  Expr y6 = exp(-one/y5);
  y7(np) = one / pow(y4, onefive);
  fcut(np) = y6/exp(-one);
  dfcut(np) = (((3 * one)/(rmax*exp(-one)))*(y2)*y6*(y*y2 - one)) * y7(np);
  
  Expr alpha = max(Expr((double)1e-3), besselparams(bfp));
  Expr x =  (one - exp(-alpha*r * irmax))/(one-exp(-alpha));
  Expr dx = (alpha * irmax)*exp(-(alpha*r * irmax))/(one - exp(-alpha));

  Func c("c");
  Expr a = (bfi + 1) * PI;
  Expr b = sqrt(2 * one * irmax)/(bfi + 1);
  c(np, bfi) = one / pow(dij(np), bfi + 1); 

  Func rbf("rbf"), drbf("drbf_f"), abf("abf_f"), dabf("dabf_f");

  rbf(bfp, bfi, np) = b * fcut(np) * sin(a*x)/r;

  Expr drbfdr = b*(dfcut(np)*sin(a*x)/r - fcut(np)*sin(a*x)/(r*r) + a*cos(a*x)*fcut(np)*dx/r);
  drbf(bfp, bfi, np, dim) = (xij(dim, np) * idij(np)) * drbfdr;

  abf(bfi, np) = fcut(np) * c(np, bfi);

  Expr drbfdr_a = (dfcut(np) - (bfi+one)*fcut(np)*idij(np)) * c(np, bfi);
  dabf(bfi, np, dim) = (xij(dim, np) * idij(np)) * drbfdr_a;

  rbf.reorder(bfi, bfp, np);
  rbf.compute_root();
  //rbf.compute_root().specialize(nbparams == 1).specialize(nbparams == 2).specialize(nbparams == 3);
  //Func rbf is scheduled to be computed with drbf_f, so it must not have any specializations.

  drbf.reorder(dim, bfi, bfp, np).unroll(dim, 3);
  drbf.compute_root();
  //drbf.compute_root().specialize(nbparams == 1).specialize(nbparams == 2).specialize(nbparams == 3);

  abf.reorder(bfi, np);
  abf.compute_root();
  
  dabf.reorder(dim, bfi, np).unroll(dim, 3);
  dabf.compute_root();

  rbf.compute_with(drbf, bfi);
  //rbf.compute_with(drbf, bfi).unroll(bfp, 1); // Unhandled exception: Error: Invalid compute_with: # of fused dims of drbf_f.s0 and rbf.s0 do not match.
  dabf.compute_with(abf, bfi);

  // Scattering
  Var rbf_abf_info("rbf_abf_info"), drbf_dabf_info("drbf_dabf_info");
  Var rbfty("rbfty");
  RDom r1(0, nbparams, 0, bdegree, 0, npairs, 0, 3);
  RDom r2(0, npairs, 0, adegree, 0, 3);
  Var abf_index("abf_index");
  Expr rbf_info_length = nbparams * bdegree;
  Expr nsp = rbf_info_length + adegree;
  // Set up rbf_info
  rbfall(np, rbf_abf_info, rbfty) = zero;

  rbfall(r1.z, r1.y + r1.x * bdegree, 0) = rbf(r1.x, r1.y, r1.z);
  rbfall(r2.x, get_abf_index(r2.y, rbf_info_length), 0) = abf(r2.y, r2.x);

  rbfall(r1.z, r1.y + r1.x * bdegree, r1[3] + 1) = drbf(r1.x, r1.y, r1.z, r1[3]);
  rbfall(r2.x, get_abf_index(r2.y, rbf_info_length), r2.z + 1) = dabf(r2.y, r2.x, r2.z);

  rbfall.reorder(rbfty, rbf_abf_info, np);
  rbfall.update(0).reorder(r1.w, r1.z);
  rbfall.update(1).reorder(r2.z, r2.y, r2.x);
  rbfall.update(2).reorder(r1.x, r1.y, r1.z);
  rbfall.update(3).reorder(r2.z, r2.y, r2.x);
  rbfall.store_root().compute_root(); 
  //rbfall.reorder_storage(
}


void radialAngularBasis(Func & sumU, Func & U,

			Func rbf, Func abf,  Func atomtype,
			Expr N, Expr K, Expr M, Expr Ne)
            
{
  Expr zero = Expr((double) 0.0);

  Var n("n"), k("k"), m("m"), ne("ne"), c("crab");
  sumU(n, k, m) = zero;

  U(n, k, m, c) = zero;

  RDom r(0, M, 0, K, 0, N);
  Expr c1 = rbf(r.z, r.x, 0);
  Expr c2 = abf(r.z, r.y, 0);
  Expr in = atomtype(r.z) - 1;

  U(r.z, r.y, r.x, c) = select(c == 0, c1 * c2,
			       select(c == 1, abf(r.z, r.y, 1) * c1 + c2 * rbf(r.z, r.x, 1),
				      select(c== 2, abf(r.z, r.y, 2) * c1 + c2 * rbf(r.z, r.x, 2) ,
					     select(c==3, abf(r.z, r.y, 3) * c1+ c2 * rbf(r.z, r.x, 3), Expr((double) 0.0)))));
  sumU(clamp(in, 0, Ne - 1), r.y, r.x) += rbf(r.z, r.x, 0) * abf(r.z, r.y, 0);

  U.reorder(c, n, k, m);
  U.store_root().compute_root();
  U.reorder_storage(c, n, k, m);
  sumU.reorder(n, k, m);
  sumU.store_root().compute_root();
  sumU.reorder_storage(n, k, m);
  sumU.compute_with(U, n);

  //U.update(0).compute_with(sumU.update(0), r.y); // Unhandled exception: Error: Found cyclic dependencies between compute_with of U and sumU
  U.update(0).reorder(c, r.z, r.y, r.x).unroll(c, 4).specialize(Ne == 1); 
  sumU.update(0).reorder(r.z, r.y, r.x).specialize(Ne == 1);
}

void twoBodyDescDeriv(Func & d2, Func & dd2, Func rbf,  Func tj, Expr N, Expr Ne, Expr nrbf2)
{
  Expr zero = Expr((double) 0.0);

  Var ne("ne"), m("m"), n("n"), dim("dim");
  d2(ne, m) = zero;
  dd2(ne, m, n, dim) = zero;

  RDom r(0, N, 0, nrbf2);

  d2(clamp(tj(r.x)-1, 0, Ne - 1), r.y) += rbf(r.x, r.y, 0);
  dd2(clamp(tj(r.x)-1, 0, Ne - 1), r.y, r.x, dim) += rbf(r.x, r.y, dim + 1);
  /*
  dd2(clamp(tj(r.x)-1, 0, Ne - 1), r.y, r.x, 0) += rbf(r.x, r.y, 1);
  dd2(clamp(tj(r.x)-1, 0, Ne - 1), r.y, r.x, 1) += rbf(r.x, r.y, 2);
  dd2(clamp(tj(r.x)-1, 0, Ne - 1), r.y, r.x, 2) += rbf(r.x, r.y, 3);
  */
    
  // These provide wrong answers ...
  d2.compute_root();
  //d2.reorder_storage(m, ne);
  dd2.compute_root();
  //dd2.reorder_storage(dim, n, m, ne);
  d2.update(0).reorder(r.x, r.y);
  dd2.update(0).reorder(dim, r.x, r.y).unroll(dim);
  d2.update(0).compute_with(dd2.update(0), r.y);
  //d2.update(0).compute_with(dd2.update(0), r.x); // Invalid compute_with: types of dim 0 of dd2.s1(r197$x is PureRVar) and d2$0.s1(r197$x is ImpureRVar) do not match.
}


void tallyTwoBodyLocalForce(Func & fij, Func & e, Func coeff2, Func rbf, Func tj, Func ti, Expr nbf, Expr N, Expr nelements)
{
  Expr zero = Expr((double) 0.0);

  Var n("n"), m("m"), dim("dim"), empty("empty");
  e() = zero;
  fij(n, dim) = zero;

  RDom r(0, N, 0, nbf);

  Expr c = coeff2(clamp(tj(r.x), 1, N - 1) - 1, r.y, clamp(ti(0) - 1, 0, nelements - 1));
  e() += c * rbf(r.x, r.y, 0);
  fij(r.x, dim) += c * rbf(r.x, r.y, dim + 1);

  fij.bound(n, 0, N);
  fij.bound(dim, 0, 3);

  fij.reorder(dim, n);
  fij.update(0).reorder(dim, r.x, r.y).unroll(dim);
  fij.reorder_storage(dim, n);
  fij.update(0).reorder(dim, r.x, r.y).unroll(dim);
  e.compute_root();
  fij.compute_root();
  //e.compute_with(fij.update(0), r.x);
}


void buildAngularBasis(Expr k3, Expr npairs, Func pq, Func rij, Func idij,
		       Func & abf4, Func & tm,
		       Var c, Var pair, Var abfi,Var abfip){
      
  Expr x = rij(0, pair);
  Expr y = rij(1, pair);
  Expr z = rij(2, pair);
      
  Expr xx = x*x;
  Expr yy = y*y;
  Expr zz = z*z;
  Expr xy = x*y;
  Expr xz = x*z;
  Expr yz = y*z;

  Expr u = x * idij(pair);
  Expr v = y * idij(pair);
  Expr w = z * idij(pair);
    
  Expr idij3 = idij(pair)*idij(pair)*idij(pair);
  Expr dudx = (yy+zz)*idij3;
  Expr dudy = -xy*idij3;
  Expr dudz = -xz*idij3;

  Expr dvdx = -xy*idij3;
  Expr dvdy = (xx+zz)*idij3;
  Expr dvdz = -yz*idij3;

  Expr dwdx = -xz*idij3;
  Expr dwdy = -yz*idij3;
  Expr dwdz = (xx+yy)*idij3;

  Expr zero = Expr((double) 0.0);

  tm(pair, abfip, c) = zero;
  tm(pair, 0, 0) = Expr((double) 1.0);
  RDom rn(1, k3 + 1, 0, 4);
  Expr m = clamp(pq(rn.x) - 1, 0, 3 * k3 - 1);
  Expr prev0 = tm(pair, m, 0);
  Expr d = clamp(pq(rn.x + k3), 1, 3);
  Var selected("selected");
  Func uvw("uvw");
  uvw(pair, selected) = select(selected == 1, u, select(selected==2, v, select(selected==3, w, Expr((double) 0.0))));
  tm(pair, rn.x, rn.y) = tm(pair, m, rn.y) * uvw(pair, d) + select(d == rn.y, tm(pair, m, 0), zero);

  Func jacobian("jacobian");
  Var dim("dim"), dim_p("dim_p");
  jacobian(pair, dim, dim_p) =
    select(dim == 0,
	   select(dim_p == 0,
		  dudx,
		  select(dim_p == 1,
			 dvdx,
			 select(dim_p == 2,
				dwdx, zero))), 
	   select(dim == 1,
		  select(dim_p == 0,
			 dudy,
			 select(dim_p == 1,
				dvdy,
				select(dim_p == 2,
				       dwdy, zero))),
		  select(dim == 2,
			 select(dim_p == 0,
				dudz,
				select(dim_p == 1,
				       dvdz,
				       select(dim_p == 2,
					      dwdz, zero))), zero)));
  abf4(pair, abfi, c) = select(c == 0, tm(pair, abfi, 0),
			       tm(pair, abfi, 1) * jacobian(pair, c-1, 0) + tm(pair, abfi, 2) * jacobian(pair, c-1, 1) + tm(pair, abfi, 3) * jacobian(pair, c-1, 2));   

  tm.reorder(c, abfip, pair);
  tm.reorder_storage(c, abfip, pair);
  tm.store_root().compute_at(abf4, abfi);
  tm.update(0).unscheduled();
  tm.update(1).reorder(pair, rn.x, rn.y).unroll(rn.y, 4);
  abf4.reorder(c, abfi, pair).unroll(c, 4);
  abf4.reorder_storage(c, pair, abfi);
  abf4.store_root().compute_root();
}


void tallyLocalForce(Func & fij, Func atomtype, Func cU, Func U, Expr nrbf3, Expr K3, Expr npairs, Expr nelements, Var dim)
{
  RDom r(0, nrbf3, 0, K3, 0, npairs);
  Expr i2 = atomtype(r.z) - 1;
  Expr c = cU(clamp(i2, 0, nelements - 1), r.y, r.x);
  fij(r.z, dim) += c * U(r.z, r.y, r.x, dim + 1);
  fij.update(1).reorder(dim, r.z, r.y, r.x);
}

void tallyLocalForceRev(Func & fij, Func atomtype, Func cU, Func U, Expr nrbf3, Expr K3, Expr npairs, Expr nelements, Var dim, int up)
{
  RDom r(0, nrbf3, 0, K3, 0, npairs);
  Expr i2 = atomtype(r.z) - 1;
  Expr c = cU(clamp(i2, 0, nelements - 1), r.y, r.x);
  fij(r.z, dim) += c * U(r.z, r.y, r.x, dim + 1);
  if (up != -1){
    fij.update(up).reorder(dim, r.z, r.y, r.x).unroll(dim).specialize(nelements == 1);
  }
}

void threeBodyCoeff(Func & cU, Func & e, Func coeff3, Func sumU, Func pn3, Func pc3, Func ti, Expr npairs, Var ne, Var k3, Var rbf3,
		    Expr nelements, Expr K3, Expr nrbf3, Expr nabf3, Expr me)
{
  Expr zero = Expr((double) 0.0);

  cU(ne, k3, rbf3) = zero;

  RDom r(0, K3, 0, nabf3, 0, nelements, 0, nelements, 0, nrbf3);
  Expr n1 = pn3(r.y);
  Expr n2 = pn3(r.y + 1);
  r.where(n1 <= r.x);
  r.where(r.x < n2);
  r.where(r[3] >=  r.z);
    
  Expr k = (2 * nelements - 3 - r.z) * (r.z/ 2) + r[3] - 1; //mem  - ki + kij;
  Expr t1 = pc3(r.x) * sumU(r.z, r.x, r[4]);
  Expr c2 = sumU(r[3], r.x, r[4]);
  Expr c3 = coeff3(r.y, r[4], clamp(k, 0, me - 1), clamp(ti(0) - 1, 0, nelements - 1));
  Expr t2 = c3 * t1;
  e() += t2 * c2;
  cU(r[3], r.x, r[4]) += t2;
  cU(r.z, r.x, r[4]) += pc3(r.x) * c2 * c3;

  cU.reorder_storage(ne, k3, rbf3);
  cU.update().reorder(r.w, r.z, r.y, r[4]);
  cU.update(1).reorder(r.w, r.z, r.y, r[4]);
  // cU.update().compute_with(cU.update(1), r.w); // Unhandled exception: Error: Cannot schedule cU.update(0) to be computed with cU.s2.r239$w
  e.update().reorder(r.w, r.z, r.y, r[4]);
  e.compute_root();
}

void fourbodycoeff(Func & e4, Func  & cU4,
		   Func coeff4, Func  sumU4, Func ti, Func pa4, Func pb4, Func pc4,
		   Expr nrbf4, Expr nabf4, Expr nelements, Expr k4, Expr Q4){

  Var ne("ne");
  Var kv("kv");
  Var rbf("rbf");
  Expr acc = clamp(ti(0) - 1, 0, nelements - 1);
  Expr zero = Expr((double) 0.0);
  cU4(ne, kv, rbf) = zero;
  e4() = zero;
  Expr q = pa4(nabf4);
  RDom r(0, nelements, 0, nelements, 0, nelements, 0, Q4, 0, nabf4, 0, nrbf4);
  Expr rbfr = r[5];
  Expr p = r[4];
  Expr n1 = pa4(p);
  Expr n2 = pa4(p+1);
  Expr nn = n2 - n1;
  Expr n1pq = r[3];
  r.where(n1 <= n1pq);
  r.where(n1pq < n2);
  Expr c = pc4(n1pq);
  Expr j1 = unsafe_promise_clamped(pb4(n1pq, 0), 0, k4);
  Expr j2 = unsafe_promise_clamped(pb4(n1pq, 1), 0, k4);
  Expr j3 = unsafe_promise_clamped(pb4(n1pq, 2), 0, k4);
  Expr i1 = r[0];
  Expr i2 = r[1];
  Expr i3 = r[2];
  r.where(i1 >= i2 && i2 >= i3);
  Expr sym3NE = (nelements) * (nelements+1) * (nelements+2)/6;
  Expr k = unsafe_promise_clamped((i1*(i1+1)*(i1+2)/6) + (i2*(i2+1)/2) + i3, 0, sym3NE);

  Expr c1 = c * sumU4(i1, j1, rbfr);
  Expr c0 = sumU4(i2, j2, rbfr);
  Expr c2 = c * c0;
  Expr c4 = c1 * c0;
  Expr c5 = coeff4(p, rbfr, k, acc);
  Expr c6 = c5 * sumU4(i3, j3, rbfr);

  e4() += c4 * c6;
  Expr scat = scatter(0, 1, 2);
  cU4(mux(scat, {i3, i2, i1}), mux(scat, {j3, j2, j1}), rbf) = gather(c5*c4, c6*c1, c6*c2); // think this is adding if statements

  cU4.reorder_storage(ne, kv, rbf);
  cU4.update().specialize(nelements == 1); // not sure if unrolling correct thing, which nelements RDOM
  e4.update().reorder(r[2], r[1], r[0], r[3], r[4], r[5]).specialize(nelements == 1);
  e4.compute_at(cU4, r[2]);
  cU4.compute_root();
  e4.compute_root();
}

void threeBodyDescDeriv(Func & dd3, Func sumU, Func U, Func atomtype, Func pn3, Func pc3,
			Func elemindex, Expr npairs, Expr q, Expr nelements, Var dim, Var nj, Var abf3, Expr nabf3, 
			Var rbf3, Expr nrbf3, Var kme, Expr me, RDom r)
{
  Expr zero = Expr((double) 0.0);
    
  dd3(dim, nj, abf3, rbf3, kme) = zero;
    
  Expr n1 = pn3(r.y);
  Expr n2 = pn3(r.y + 1);
  r.where(n1 <= r.x); 
  r.where(r.x < n2);
  RVar ry = r.y; // p / nabf3
  RVar rx = r.x; // q
  RVar rz = r.z; // i1
  RVar rzz = r[3]; // j / N
    
  Expr t1 = pc3(rx) * sumU(rz, rx, rbf3);
  Expr i2 = atomtype(rzz) - 1;
  Expr k = elemindex(clamp(i2, 0, nelements - 1), rz);
  Expr f = select(rz == i2, 2 * t1, t1);
    
  dd3(dim, rzz, ry, rbf3, clamp(k, 0, me - 1)) += f * U(rzz, rx, rbf3, dim + 1);

  dd3.reorder_storage(dim, nj, abf3, rbf3, kme);
  dd3.update().reorder(dim, rzz, rz, rx, ry, rbf3).unroll(dim, 3).specialize(nelements == 1);
}

void threeBodyDesc(Func & d3,
		   Func sumU, Func pn3, Func pc3,
		   Expr npairs, Expr nelements, Expr nrbf3, Expr nabf3, Expr k3,
		   Var abf3, Var rbf3, Var kme)
{
  Expr zero = Expr((double) 0.0);
  Expr me = nelements * (nelements + 1)/2;
  Expr mem = nelements * (nelements - 1)/2;

  d3(abf3, rbf3, kme) = zero;

  RDom r(0, k3, 0, nabf3, 0, nelements, 0, nelements);
  Expr n1 = pn3(r.y);
  Expr n2 = pn3(r.y + 1);
  r.where(n1 <= r.x);
  r.where(r.x < n2);
  RVar ry = r.y;
  RVar rx = r.x;
  RVar rz = r.z;
  RVar rzz = r[3];
  //r.where(rzz <= rz); // This feels like rz is i2, but you use it below as i1
  r.where(rz <= rzz);

  Expr t1 = pc3(rx) * sumU(rz, rx, rbf3);
  Expr k = (2 * nelements - 3 - rz) * (rz/ 2) + rzz - 1; //mem  - ki + kij;
  Expr t2 = sumU(rzz, rx, rbf3);
  d3(ry, rbf3, clamp(k, 0, me -1)) += t1 * t2;

  d3.reorder_storage(abf3, rbf3, kme);
  d3.update().reorder(rz, rzz, ry, rbf3).specialize(nelements == 1);
  
  //k is a trian
  //(n*(n-1)/2) - (n-i)*((n-i)-1)/2 + j - i - 1
  //Formula for indexing
  //https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix

}

void indexMap3(Func & indexmap, Expr n1, Expr n2, Expr n3, Expr N1, Expr N2){
  Var v1("v1");
  Var v2("v2");
  Var v3("v3");
  Var k("k");
  Var c("c");

  Expr i3 =  k / (N1 * N2);
  Expr kp = (k - i3 * N1 * N2);
  Expr i2 = kp/ N1;
  Expr i1 = kp % N1;

  indexmap(k, c) = mux(c, {i1, i2, i3});
  indexmap.compute_root();
}


void fourbodystuff(Func & fij, Func & e23,
		   Func ind23, Func ind32, Func coeff23, Func d2, Func d3, Func dd3, Func dd2, Func ti,
		   Expr npairs, Expr n23, Expr n32, Expr nelements, Expr nrbf2, Expr nrbf3, Expr nabf3, Expr me,
		   Var pairindex)
{
  Func d23("d23");
  Var d23i("d23i");
  Var d23j("d23j");
  e23() = Expr((double) 0.0);
  Expr acc = unsafe_promise_clamped(ti(0) - 1, 0, nelements - 1);
  // j is n32
  //i is n23
  d23(d23i, d23j) = d2(unsafe_promise_clamped(ind23(d23i, 1), 0, nelements -1),
		       unsafe_promise_clamped(ind23(d23i, 2), 0, nrbf2 -1)) * d3(unsafe_promise_clamped(ind32(d23j, 0), 0, nabf3- 1),
										 unsafe_promise_clamped(ind32(d23j, 1),0, nrbf3 - 1),
										 unsafe_promise_clamped(ind32(d23j, 2), 0, me -1));
  d23.compute_root();
  d23.reorder(d23i, d23j);
  d23.reorder_storage(d23i, d23j);

  RDom e23rdom(0, n23, 0, n32);
  e23() += d23(e23rdom.x, e23rdom.y) * coeff23(e23rdom.x, e23rdom.y, acc);
  e23.compute_root();

  Expr zero = Expr((double) 0.0);
  Func cf1("cf1");
  Var j("j");
  Var dim("dim");
  RDom r1(0, n23);
  cf1(j) = zero;
  cf1(j) += d2(unsafe_promise_clamped(ind23(r1.x, 1), 0, nelements -1), unsafe_promise_clamped(ind23(r1.x, 2), 0, nrbf2 -1)) * coeff23(r1.x, j, acc);

  RDom r2(0, n32);
  fij(pairindex, dim) += cf1(r2.x) * dd3(dim, pairindex,   unsafe_promise_clamped(ind32(r2.x, 0), 0, nabf3- 1), unsafe_promise_clamped(ind32(r2.x, 1),0, nrbf3 - 1), unsafe_promise_clamped(ind32(r2.x, 2), 0, me -1));

  Func cf2("cf2");
  Var i("i");
  cf2(i) = zero;
  cf2(i) += d3( unsafe_promise_clamped(ind32(r2.x, 0), 0, nabf3- 1), unsafe_promise_clamped(ind32(r2.x, 1),0, nrbf3 - 1), unsafe_promise_clamped(ind32(r2.x, 2), 0, me -1)) * coeff23(i, r2.x, acc);

  fij(pairindex, dim) += cf2(r1.x) * dd2(unsafe_promise_clamped(ind23(r1.x, 1), 0, nelements -1), unsafe_promise_clamped(ind23(r1.x, 2), 0, nrbf2 -1), pairindex, dim);

  cf1.update().reorder(r1.x, j);
  cf1.compute_root();

  fij.update(2).reorder(dim, pairindex, r2.x);
  fij.compute_root();

  cf2.update().reorder(r2.x, i);
  cf2.compute_root();

  fij.update(3).reorder(dim, pairindex, r1.x);
  fij.compute_root();
}


void fivebodystuff(Func & fij, Func & e33,
		   Func ind33, Func coeff33, Func d3, Func ti, Func dd3,
		   Expr npairs, Expr n33, Expr nelements, Expr nabf3, Expr nrbf3, Expr me,
		   Var pairindex)
{
  Expr symMe = nelements*(nelements+1)/2;
  Expr symN33 = n33 * (n33+1)/2;
  Func d33("d33");
  Var kv("k");
  Var dim("dim");
  d33(kv) = Expr((double) 0.0);
  RDom r(0, n33, 0, n33);
  r.where(r.x >= r.y);
  //col + row*(M-1)-row*(row-1)/2
  //https://stackoverflow.com/questions/242711/algorithm-for-index-numbers-of-triangular-matrix-coefficients
  Expr temp = clamp(r.x + r.y * (n33 - 1) - r.y * (r.y-1)/2, 0, symN33 - 1);
  Expr k =  print_when(temp < 0 || temp > symN33- 1, temp, "error!"); //(print(temp, symN33, r.x, r.y));
  
  //abf, rbf, me
  Expr abfacc1 = unsafe_promise_clamped(ind33(r.x, 0), 0, nabf3 - 1);
  Expr abfacc2 = unsafe_promise_clamped(ind33(r.y, 0), 0, nabf3 - 1);
  Expr rbfacc1 = unsafe_promise_clamped(ind33(r.x, 1), 0, nrbf3 - 1);
  Expr rbfacc2 = unsafe_promise_clamped(ind33(r.y, 1), 0, nrbf3 - 1);
  Expr meacc1 = unsafe_promise_clamped(ind33(r.x, 2), 0, symMe - 1);
  Expr meacc2 = unsafe_promise_clamped(ind33(r.y, 2), 0, symMe - 1);
  d33(k) += d3(abfacc2, rbfacc2, meacc2) * d3(abfacc1, rbfacc1, meacc1);

  d33.compute_root();
  d33.update().reorder(r.x, r.y);

  RDom rdot(0, symN33);
  Expr acc = clamp(ti(0) - 1, 0, nelements - 1);
  e33() = Expr((double) 0.0);
  e33() += d33(rdot) * coeff33(rdot, acc);
  e33.compute_root();


  Func cf133("cf133");
  Var j("j");
  cf133(j) = Expr((double) 0.0);
  cf133(r.y) += d3(abfacc1, rbfacc1, meacc1) * coeff33(k, acc);
  cf133.compute_root();

  RDom rn33(0, n33);
  Expr abfacc3 = unsafe_promise_clamped(ind33(rn33, 0), 0, nabf3 - 1);
  Expr rbfacc3 = unsafe_promise_clamped(ind33(rn33, 1), 0, nrbf3 - 1);
  Expr meacc3 = unsafe_promise_clamped(ind33(rn33, 2), 0, symMe - 1);

  fij(pairindex, dim) += cf133(rn33) * dd3(dim, pairindex, abfacc3, rbfacc3, meacc3);

  Func cf233("cf233");
  Var i("j");
  cf233(i) = Expr((double) 0.0);
  cf233(r.x) += d3(abfacc2, rbfacc2, meacc2) * coeff33(k, acc);
  cf233.compute_root();

  fij(pairindex, dim) += cf233(rn33) * dd3(dim, pairindex, abfacc3, rbfacc3, meacc3);

  cf133.update().reorder(r.x, r.y);

  fij.update(4).reorder(dim, pairindex, rn33.x);

  cf233.update().reorder(r.x, r.y);

  fij.update(5).reorder(dim, pairindex, rn33.x);
}

class  poddescTwoBody : public Halide::Generator<poddescTwoBody> {
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

  Input<Buffer<double>> coeff2{"coeff2", 3};
  Input<Buffer<int>> ti{"ti", 1};
  Input<Buffer<int>> tj{"tj", 1};
  
  
  Input<int> nrbf2{"nrbf2", 1};
  Output<Buffer<double>> fij_o{"fij_o", 2};
  Output<double> e_o{"e_o"};

  Input<int> k3{"k3", 1};
  Input<int> k4{"k4", 1};
  Input<int> q4{"q4", 1};
  Input<Buffer<int>> pq{"pq", 1};
  Input<Buffer<int>> pn3{"pn3", 1};
  Input<Buffer<int>> pc3{"pc3", 1};
  Input<Buffer<int>> pa4{"pa4", 1};
  Input<Buffer<int>> pb4{"pb4", 2};
  Input<Buffer<int>> pc4{"pc4", 1};
  Input<Buffer<int>> elemindex{"elemindex", 2};


  Input<int> nrbf3{"nrbf3", 1};
  Input<int> nrbf4{"nrbf4", 1};
  Input<int> nelements{"nelements", 1};

  Input<int> nd23{"nd23", 1};
  Input<int> nd33{"nd33", 1};
  Input<int> nd34{"nd34", 1};
  Input<int> n32{"n32", 1};
  Input<int> n23{"n23", 1};
  Input<int> n33{"n33", 1};
  Input<int> n43{"n43", 1};
  Input<int> n34{"n34", 1};
  Input<int> n44{"n44", 1};
  Input<int> nabf3{"nabf3", 1};
  Input<int> nabf4{"nabf4", 1};
  Input<int> nrbf23{"nrbf23", 1};
  Input<int> nrbf33{"nrbf33", 1};
  Input<int> nrbf34{"nrbf34", 1};
  Input<int> nrbf44{"nrbf44", 1};
  Input<int> nabf23{"nabf23", 1};
  Input<int> nabf33{"nabf33", 1};
  Input<int> nabf34{"nabf34", 1};
  Input<int> nabf44{"nabf44", 1};

  
  
  Input<Buffer<double>> coeff3{"coeff3", 4};
  Input<Buffer<double>> coeff23{"coeff23", 3};
  Input<Buffer<double>> coeff33{"coeff33", 2};
  Input<Buffer<double>> coeff4{"coeff4", 4};
  Input<Buffer<double>> coeff34{"coeff34", 1};
  Input<Buffer<double>> coeff44{"coeff44", 1};


  Output<double> e3_o{"e3_o"};

  void generate() {

    Func ind23("ind23");
    Func ind32("ind32");
    Func ind33("ind33");
    Expr symMe = nelements*(nelements+1)/2;
    indexMap3(ind23, Expr(1), nrbf23, nelements, Expr(1), nrbf2);
    indexMap3(ind32, nabf23, nrbf23, symMe, nabf3, nrbf3);
    indexMap3(ind33, nabf33, nrbf33, symMe, nabf3, nrbf3);

    
    rijs.dim(0).set_bounds(0, 3).set_stride(1);
    rijs.dim(1).set_bounds(0, npairs).set_stride(3);
    Phi.dim(0).set_bounds(0, ns).set_stride(1);
    Phi.dim(1).set_bounds(0, ns).set_stride(ns);
    coeff2.dim(0).set_bounds(0, npairs).set_stride(nrbf2);
    coeff2.dim(1).set_bounds(0, nrbf2).set_stride(1);
    coeff2.dim(2).set_bounds(0, nelements).set_stride(nrbf2 * npairs);
    coeff23.dim(2).set_bounds(0, nelements).set_stride(n23 * n32);
    coeff23.dim(1).set_bounds(0, n32).set_stride(n23);
    coeff23.dim(0).set_bounds(0, n23).set_stride(1);
    coeff33.dim(0).set_bounds(0, n33 * (n33+1)/2).set_stride(1);
    coeff33.dim(1).set_bounds(0, nelements).set_stride(n33 * (n33+1)/2);
    besselparams.dim(0).set_bounds(0, nbesselparams);

    Expr me = nelements * (nelements + 1)/2;
    
    Var bfi("basis function index");
    Var bfp("basis function param");
    Var np("pairindex");
    Var numOuts("numOuts");
    Var dim("dim");

    Func rbft("rbft");
    Func dij("dij"), idij("idij"), fcut("fcut"), dfcut("dfcut");
    /*
    buildRBF(rbft, rijs, besselparams, rin, rcut-rin,
	     bdegree, adegree, nbesselparams, npairs, ns,
	     bfi, bfp, np, dim);
         */
    buildRBF(rbft, rijs, dij, idij, fcut, dfcut, besselparams, rin, rcut-rin,
	     bdegree, adegree, nbesselparams, npairs, ns,
	     bfi, bfp, np, dim);

    // MatMul
    Var i("i");
    Var j("j");
    Var k("k");
    Var c("c");
    Func prod("prod");
    prod(c, k, i, j) = Phi(k, i) * rbft(j, k, c);
    Func rbf("rbf");
    rbf(j, i, c) = Expr((double) 0.0);
    RDom r(0, ns);
    rbf(j, i, c) += prod(c, r, i, j);
    //j is num pairs so we sum over basis functions here...
    //rbf.update(0).reorder(c, r, i, j); // TODO
    rbf.store_root().compute_root();

    Func fij("fij"), e("e");
    tallyTwoBodyLocalForce(fij, e, coeff2, rbf, tj, ti, nrbf2, npairs, nelements);
    e_o() = e();
    Var n("n");
    Func abf4("abf4");
    Func tm("tm");
    Var abfi("abfi");
    Var abfip("abfip");
    /*
    buildAngularBasis(k3, npairs, pq, rijs,
		      abf4, tm,
		      c, np,  abfi, abfip
		      );
              */
    buildAngularBasis(k3, npairs, pq, rijs, idij,
		      abf4, tm,
		      c, np,  abfi, abfip
		      );


    Func sumU("sumU"), U("U");
    radialAngularBasis(sumU, U, rbf, abf4,
		       tj, npairs, k3, nrbf3, nelements);

    Func d2("d2"), dd2("dd2");
    twoBodyDescDeriv(d2, dd2, rbf, tj, npairs, nelements, nrbf2);

    Func d3("d3");
    Var abfThree("abfThree");
    Var rbfThree("rbfThree");
    Var kme("kme");
    threeBodyDesc(d3, sumU, pn3, pc3,
		  npairs, nelements, nrbf3, nabf3, k3,
		  abfThree, rbfThree, kme);

    d3.compute_root();
    
    Func dd3("dd3");
    Var nj("nj");    
    RDom r3body(0, k3, 0, nabf3, 0, nelements, 0, npairs);
    threeBodyDescDeriv(dd3, sumU, U, tj, pn3, pc3,
		       elemindex, npairs, k3, nelements, dim, nj, abfThree, nabf3, 
		       rbfThree, nrbf3, kme, me, r3body);

    dd3.update(0).reorder(dim, r3body[3], r3body.x, r3body.y, rbfThree);
    dd3.compute_root();
 
    Func cU("cU");
    Func e3("e3");
    Var ne("ne"), k3var("k3var");
   
    threeBodyCoeff(cU, e3, coeff3, sumU, pn3, pc3, ti, nj, ne, k3var, rbfThree,
		   nelements, k3, nrbf3, nabf3, me);
    cU.compute_root();
    e3.compute_root();
    e3_o() = e3();

    tallyLocalForceRev(fij, tj, cU, U, nrbf3, k3, npairs, nelements, dim, 1);
    Func e23("e23");
    fourbodystuff(fij, e23, ind23, ind32, coeff23, d2, d3, dd3, dd2, ti, npairs, n23, n32, nelements, nrbf2, nrbf3, nabf3,me, n);
    e3_o() += e23();

    Func e33("e33");
    fivebodystuff(fij, e33,
		  ind33, coeff33, d3, ti, dd3,
		  npairs, n33, nelements, nabf3, nrbf3, me,
		  n);
    e3_o() += e33();

    Func sumU4("sumU4");
    Func U4("u4");
    Var m4v("m4");
    Var k4v("k4");
    Var e4v("e4");
    Var dim4v("dim4");


    sumU4(m4v, k4v, e4v) = sumU(m4v, k4v, e4v);
    U4(m4v, k4v, e4v, dim4v) = U(m4v, k4v, e4v, dim4v);

    Func cu4("cu4");
    Func e4("e4");
    fourbodycoeff(e4, cu4,
		  coeff4, sumU4, ti, pa4, pb4, pc4,
		  nrbf4, nabf4, nelements, k4, q4);
    e3_o()+= e4();

    tallyLocalForceRev(fij, tj, cu4, U4, nrbf4, k4, npairs, nelements, dim, 6);


    fij.store_root().compute_root();

    fij_o(n, dim) = fij(n, dim);
    fij_o.reorder(dim, n);
    fij_o.dim(0).set_bounds(0, npairs).set_stride(3);
    fij_o.dim(1).set_bounds(0, 3).set_stride(1);
  }
};

HALIDE_REGISTER_GENERATOR(poddescTwoBody, poddescTwoBody);
