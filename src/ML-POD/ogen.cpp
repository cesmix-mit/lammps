#include "Halide.h"
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h> 
using namespace Halide;

Expr get_abf_index(Expr original_index, Expr rbf_info_length) {
  return rbf_info_length + original_index;
}
//Func & rbf_f, Func & rbfx_f, Func & rbfy_f, Func & rbfz_f,
void buildRBF( Func & rbfall,
	       Func xij, Func besselparams, Func offsets,
	       Expr rin, Expr rmax,
	       Expr bdegree, Expr adegree, Expr nbparams, Expr npairs, Expr ns, Expr nijmax,
	       Var bfi, Var bfp, Var np, Var dim, Var oatom)
{

  Expr one = Expr((double) 1.0);
  Expr zero = Expr((double) 0.0);
  Expr onefive = Expr((double) 1.5);
  Expr PI = Expr( (double)M_PI);
  Expr oatomnext = min(offsets(oatom) + nijmax, offsets(oatom+1))-1;
  Expr pairclamp = unsafe_promise_clamped(unsafe_promise_clamped(np + offsets(oatom), offsets(oatom), oatomnext), 0, npairs-1);

  Expr xij1 = xij(0, pairclamp);
  Expr xij2 = xij(1, pairclamp);
  Expr xij3 = xij(2, pairclamp);

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

  rbf(bfp, bfi, np, oatom) = b * fcut * sin(a*x)/r;
  // rbf.trace_stores();
  //  rbf.bound(bfp, 0, nbparams);
  //  rbf.bound(bfi, 0, bdegree);
  //  rbf.bound(np, 0, npairs);
  Expr drbfdr = b*(dfcut*sin(a*x)/r - fcut*sin(a*x)/(r*r) + a*cos(a*x)*fcut*dx/r);
  drbf(bfp, bfi, np, dim, oatom) = (xij(dim, pairclamp)/dij) * drbfdr;
  // drbf.trace_stores();
  drbf.bound(dim, 0, 3);
  //  drbf.bound(bfp, 0, nbparams);
  //  drbf.bound(bfi, 0, bdegree);
  //  drbf.bound(np, 0, npairs);

  Expr power = pow(dij, bfi+one);
  abf(bfi, np, oatom) = fcut/power;;
  // abf.trace_stores();
  //  abf.bound(bfi, 0, adegree);
  //  abf.bound(np, 0, npairs);
  Expr drbfdr_a = dfcut/c - (bfi+one)*fcut/(c*dij);
  dabf(bfi, np, dim, oatom) = (xij(dim, pairclamp)/dij) * drbfdr_a;
  // dabf.trace_stores();
  dabf.bound(dim, 0, 3);
  dabf.bound(bfi, 0, adegree);
  //  dabf.bound(np, 0, npairs);

  rbf.reorder(bfi, bfp, np, oatom);
  //  rbf.compute_root();

  drbf.reorder(dim, bfi, bfp, np, oatom);
  //  drbf.compute_root();

  abf.reorder(bfi, np, oatom);
  //  abf.compute_root();
  
  dabf.reorder(dim, bfi, np, oatom);
  //  dabf.compute_root();

  //  rbf.compute_with(drbf, bfi);
  //  dabf.compute_with(abf, bfi);

  // Loop order bfi first was 7ish seconds
  //rbf.compute_with(abf, bfi).compute_with(drbf,bfi).compute_with(dabf, bfi);
  
  
  

  // rbf.size() = nbparams * bdegree * npairs
  // drbf[x].size() = nbparams * bdegree * npairs
  // abf.size() = adegree * npairs 
  // dabf[x].size() = adegree * npairs
  // output.size() = nbparams * bdgree * npairs + adegree * npairs
  Var rbf_abf_info("rbf_abf_info"), drbf_dabf_info("drbf_dabf_info");
  Var rbfty("rbfty");
  RDom r1(0, nbparams, 0, bdegree, 0, 3);
  RDom r2(0, adegree, 0, 3);
  Var abf_index("abf_index");
  Expr rbf_info_length = nbparams * bdegree;
  Expr nsp = rbf_info_length + adegree;
  // Set up rbf_info
  rbfall(np, rbf_abf_info, rbfty, oatom) = zero;
  rbfall.bound(rbfty, 0, 4);
  //  rbfall.bound(np, 0, npairs);
  rbfall.bound(rbf_abf_info, 0, ns);

  rbfall(np, r1.y + r1.x * bdegree, 0, oatom) = rbf(r1.x, r1.y, np, oatom);
  rbfall(np, get_abf_index(r2.x, rbf_info_length), 0, oatom) = abf(r2.x, np, oatom);

  rbfall(np, r1.y + r1.x * bdegree, r1.z + 1, oatom) = drbf(r1.x, r1.y, np, r1.z, oatom);
  rbfall(np, get_abf_index(r2.x, rbf_info_length), r2.y + 1, oatom) = dabf(r2.x, np, r2.y, oatom);

  /*
    rbfall(r1.z, r1.y + r1.x * bdegree, 1) = drbf(r1.x, r1.y, r1.z, 0);
    rbfall(r2.x, get_abf_index(r2.y, rbf_info_length), 1) = dabf(r2.y, r2.x, 0);
    rbfall(r1.z, r1.y + r1.x * bdegree, 2) = drbf(r1.x, r1.y, r1.z, 1);
    rbfall(r2.x, get_abf_index(r2.y, rbf_info_length), 2) = dabf(r2.y, r2.x, 1);
    rbfall(r1.z, r1.y + r1.x * bdegree, 3) = drbf(r1.x, r1.y, r1.z, 2);
    rbfall(r2.x, get_abf_index(r2.y, rbf_info_length), 3) = dabf(r2.y, r2.x, 2);
  */

  // This seems like it was the most important thing that needed to be changed 
  // rbfall.compute_root();  // 19353.844 ms .001 ms 0%
  rbfall.reorder(rbfty, rbf_abf_info, np, oatom);
  rbfall.update(0).reorder(r1.y, r1.z, oatom);
  rbfall.update(1).reorder(r2.y, r2.x, oatom);
  rbfall.update(2).reorder(r1.x, r1.y, r1.z, oatom);
  rbfall.update(3).reorder(r2.y, r2.x, oatom);
  //rbfall.store_root().compute_root();  // 19247.139 ms .001 ms 0%
  // nothing? TERRIBLE TERRIBLE TERRIBLE, possibly bottleneck from earlier? 295 seconds total! with rbft taking 2.378 ms (77%)
}


void radialAngularBasis(Func & sumU, Func & U,

			Func rbf, Func abf,  Func atomtype, Func aj, Func offsets,
			Var n, Var k, Var m, Var ne, Var c,
			Expr N, Expr K, Expr M, Expr Ne,
			Expr natoms, Expr nijmax,
			Var oatom)
{
  Expr zero = Expr((double) 0.0);


  sumU(ne, k, m, oatom) = zero;


  //  Func prodU("prodU");
  //  prodU(n, k, m, oatom)= rbf(n, m, 0)  * abf(n, k, 0, oatom);
  Expr npoff = unsafe_promise_clamped(n + offsets(oatom), 0 , N-1);
  Expr c1 = rbf(n, m, 0, oatom);
  Expr c2 = abf(n, k, 0, oatom);


  U(n, k, m, c, oatom) = select(c == 0, rbf(n, m, 0, oatom)  * abf(n, k, 0, oatom),
				select(c == 1, abf(n, k, 1, oatom) * c1 + c2 * rbf(n, m, 1, oatom),
				       select(c== 2, abf(n, k, 2, oatom) * c1 + c2 * rbf(n, m, 2, oatom) ,
					      select(c==3, abf(n, k, 3, oatom) * c1+ c2 * rbf(n, m, 3, oatom), Expr((double) 0.0)))));
  U.reorder(c, m, k, n, oatom);
  // sumU(r.x, r.y, clamp(in, 0, Ne - 1)) += rbf(r.x, r.z) * abf(r.y, r.z);

  RDom r(0, M, 0, K, 0, nijmax);
  Expr rzz = oatom;
  Expr rz = r.z;
  r.where(rz < offsets(rzz + 1) - offsets(rzz));
  Expr oatommax = min(nijmax + offsets(oatom), offsets(oatom+1))-1;
  Expr rzboundeds = unsafe_promise_clamped(unsafe_promise_clamped(rz + offsets(oatom), offsets(oatom), offsets(oatom)+nijmax - 1), 0, N-1);
  Expr rzbounded = clamp(rz + offsets(oatom), offsets(oatom), offsets(oatom)+nijmax - 1);
  Expr ry = r.y;
  Expr rx = r.x;
  Expr in = atomtype(rzboundeds) - 1;
  sumU(clamp(in, 0, Ne - 1), r.y, r.x, rzz) += rbf(r.z, r.x, 0, oatom)  * abf(r.z, r.y, 0, oatom); //prodU(rzbounded, ry, rx); //rbf(r.z, r.x, 0) * abf(r.z, r.y, 0);

  //  sumU.update(0).reorder(r.x);
  //  abf.in(sumU).compute_at(sumU, oatom);
  //sumU.update(0).reorder(oatom, r.z, r.x, r.y);

  //  rbf.in(prodU).compute_at(prodU, oatom);

  //  sumU.bound(oatom, 0, natoms);
  sumU.bound(m, 0, M);
  U.bound(m, 0, M);

  sumU.bound(k, 0, K);
  U.bound(k, 0, K);

  sumU.bound(ne, 0, Ne);
  //  U.bound(n, 0, N);

  U.bound(c, 0, 4);

  /*
    U.reorder(c, m, k, n);
    U.store_root().compute_root();
    sumU.reorder(m, k, ne);
    sumU.update(0).reorder(r.x, r.y, r.z);
    sumU.store_root().compute_root();
    //sumU.compute_with(U, m);
    //sumU.update(0).compute_with();
    */

  U.reorder(c, k, m, n);//b.unroll(c);
  //  U.store_root().compute_root();
  sumU.reorder(ne, k, m, oatom);
  //  sumU.store_root().compute_root();
  sumU.update(0).reorder(r.z, r.y, r.x);
  //sumU.compute_with(U, k);

  //  U.update(0).reorder(c, n, k,m).unroll(c);

  //sumU.update(0).compute_with(U.update(0), r.y);
}

void twoBodyDescDeriv(Func & d2, Func & dd2,
		      Func rbf,  Func tj, Func aj, Func offsets,
		      Expr N, Expr Ne, Expr nrbf2, Expr natoms, Expr nijmax, Var oatom)
{
  Expr zero = Expr((double) 0.0);

  Var ne("ne"), m("m"), n("n"), dim("dim");
  Func combinedD("combinedD");
  d2(ne, m, oatom) = zero;
  //  dd2(ne, m, n, dim, oatom) = zero;
  dd2(dim, n, ne, m, oatom) = zero;

  RDom r(0, nrbf2, 0, nijmax);
  //  r.where(offsets(oatom) <= r.y);
  //  r.where(r.y < offsets(oatom+1));
  r.where(r.y < offsets(oatom + 1) - offsets(oatom));

  

  //  Expr acc = unsafe_promise_clamped(aj(r.x), 0, natoms - 1);
  //  Expr oatommax = min(offsets(oatom + 1), nijmax + offsets(oatom)) - 1;
  Expr bound = unsafe_promise_clamped(r.y + offsets(oatom), 0, N-1);
  //  Expr boundlhs = unsafe_promise_clamped(r.y- offsets(oatom), 0, nijmax);
  Expr ty = clamp(tj(bound)-1, 0, Ne - 1);
  d2(ty, r.x, oatom) += rbf(r.y, r.x, 0, oatom);
  dd2(dim, r.y, ty, r.x, oatom) += rbf(r.y, r.x, dim + 1, oatom);

  //dd2.reorder_storage(dim, n, m, ne, oatom);
  //  dd2.reorder_storage(oatom, ne, m, n, dim);
  //  dd2(clamp(tj(r.x)-1, 0, Ne - 1), r.y, r.x, 1, oatom) += rbf(r.x, r.y, 2);
  //  dd2(clamp(tj(r.x)-1, 0, Ne - 1), r.y, r.x, 2, oatom) += rbf(r.x, r.y, 3);

  d2.bound(ne, 0, Ne);
  d2.bound(m, 0, nrbf2);
  //  d2.bound(oatom, 0, natoms);

  dd2.bound(ne, 0, Ne);
  dd2.bound(m, 0, nrbf2);
  dd2.bound(dim, 0, 3);
  dd2.reorder(dim, ne, m, n, oatom);
  dd2.update(0).reorder(r.x, r.y, oatom);
  //  dd2.bound_storage(n, offsets(oatom+1) - offsets(oatom))
  //  dd2.bound(oatom, 0, natoms);
    
  //  d2.compute_root();
  //  dd2.compute_root();
}


void tallyTwoBodyLocalForce(Func & fij, Func & e,
			    Func coeff2, Func rbf, Func tj, Func tA, Func aj, Func offsets,
			    Expr nbf, Expr N, Expr nelements, Expr natoms, Expr nijmax,
			    Var oatom)
{
  Expr zero = Expr((double) 0.0);

  Var n("n"), m("m"), dim("dim"), empty("empty");

  RDom r(0, nijmax, 0, nbf, "fij2atomrdom");
  r.where(r.x < offsets(oatom + 1) - offsets(oatom));
  Expr oatommax = min(offsets(oatom + 1), nijmax + offsets(oatom)) - 1;
  Expr bound = unsafe_promise_clamped(unsafe_promise_clamped(r.x + offsets(oatom), 0, N - 1), offsets(oatom), oatommax);
  
  Expr c = coeff2(clamp(tj(bound), 1, N - 1) - 1, r.y, clamp(clamp(tA(oatom), 0, natoms -1) - 1, 0, nelements - 1));
  e(oatom) += c * rbf(r.x, r.y, 0, oatom);
  fij(dim, r.x, oatom) += c * rbf(r.x, r.y, dim + 1, oatom);


  fij.bound(dim, 0, 3);

  //  fij.reorder(dim, n);
  fij.update(0).reorder(dim, r.y, r.x);
}



void buildAngularBasis(Expr k3, Expr npairs, Expr nijmax,
		       Func pq, Func rij, Func offsets,
		       Func & abf4, Func & tm,
		       Var c, Var pair, Var abfi,Var abfip, Var oatom){

  Expr oatomnext = min(offsets(oatom) + nijmax, offsets(oatom+1))-1;
  Expr pairclamp = unsafe_promise_clamped(unsafe_promise_clamped(pair + offsets(oatom), offsets(oatom), oatomnext), 0, npairs-1);
      
  Expr x = rij(0, pairclamp);
  Expr y = rij(1, pairclamp);
  Expr z = rij(2, pairclamp);
      
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

  Expr zero = Expr((double) 0.0);


  tm(pair, abfi, c, oatom) = zero;
  tm(pair, 0, 0, oatom) = Expr((double) 1.0);
  RDom rn(1, k3 - 1, 0, 4);
  Expr m = clamp(pq(rn.x) - 1, 0, 3 * k3 - 1);
  Expr prev0 = tm(pairclamp, m, 0, oatom);
  Expr d = clamp(pq(rn.x + k3), 1, 3);
  Var selected("selected");
  Func uvw("uvw");
  uvw(pair, selected, oatom) = select(selected == 1, u, select(selected==2, v, select(selected==3, w, Expr((double) 0.0))));
  tm(pair, rn.x, rn.y, oatom) = tm(pair, m, rn.y, oatom) * uvw(pair, d, oatom) + select(d == rn.y, tm(pair, m, 0, oatom), zero);
  // TODO: maybe something we can do with select statement -- ordering seems to be correct


  Func jacobian("jacobian");
  Var dim("dim"), dim_p("dim_p");
  jacobian(pair, dim, dim_p, oatom) =
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

  abf4(pair, abfi, c, oatom) = select(c == 0, tm(pair, abfi, 0, oatom),
				      tm(pair, abfi, 1, oatom) * jacobian(pair, c-1, 0, oatom) + tm(pair, abfi, 2, oatom) * jacobian(pair, c-1, 1, oatom) + tm(pair, abfi, 3, oatom) * jacobian(pair, c-1, 2, oatom));
  // tm.compute_root(); // 21697.324 ms total -- .119 ms tm -- 52%
  // tm.store_root().compute_root(); abf4.compute_root(); // 21753.123 ms total -- .118 ms tm -- 52%
  // tm.store_root().compute_at(abf4, pair); // 66554 ms total
  // abf4.compute_root();

  //abf4(pair, abfi, c) = abf4tm(pair, abfi, 0, c, 0);
  //abf4.store_root().compute_root(); 	//70+ seconds
  //abf4tm.reorder(c, abfi, pair);
  //abf4tm.store_root().compute_root();
  // abf4tm.store_root().compute_at(abf4, abfi);
  // abf4.store_root().compute_root();

  /*
    uvw.compute_at(tm, pair);
    //tm.update(0).reorder(abfip, pair);
    tm.compute_at(abf4, pair);
  */
  tm.reorder(pair, abfi, c, oatom);
  //  tm.store_root().compute_root();
  //  abf4.compute_at(fife_o, rout.z);
  abf4.reorder(pair, abfi, c, oatom);//.unroll(c, 4);
  // tm.compute_at(abf4, pair);
  //  abf4.store_root().compute_root();
  //abf4.store_root().compute_root();
}


// void tallyLocalForce(Func & fij, Func atomtype, Func aj, Func cU, Func U, Expr nrbf3, Expr K3, Expr npairs, Expr nelements, Var dim)
// {
//   RDom r(0, nrbf3, 0, K3, 0, npairs);
//   Expr i2 = atomtype(r.z) - 1;
//   Expr c = cU(clamp(i2, 0, nelements - 1), r.y, r.x, unsafe_promise_clamped(aj(r.z), 0, natoms - 1));
//   fij(r.z, dim) += c * U(r.z, r.y, r.x, dim + 1);
//   fij.update(1).reorder(dim, r.z, r.y, r.x);
// }

void tallyLocalForceRev(Func & fij,
			Func atomtype, Func aj, Func offsets,
			Func cU, Func U, Func UW,
			Expr nrbf3, Expr K3, Expr npairs, Expr nelements, Expr natoms, Expr nijmax,
			Var oatom, Var dim, int up)
{
  RDom r(0, nrbf3, 0, K3, 0, npairs, "fijRevRdom");
  r.where(offsets(oatom) <= r.z && r.z < offsets(oatom+1));
  Expr i2 = atomtype(r.z) - 1;
  //  Expr c = cU(clamp(i2, 0, nelements - 1), r.y, r.x);
  Expr c = cU(clamp(i2, 0, nelements - 1), r.y, r.x, oatom);
  Expr oatomDiff = offsets(oatom + 1) - offsets(oatom);
  Expr oatommax = min(nijmax + offsets(oatom), offsets(oatom+1)) - 1;
  Expr rzbounded = unsafe_promise_clamped(r.z, offsets(oatom), oatommax);
  Expr erg = unsafe_promise_clamped(clamp(r.z, 0, npairs - 1), offsets(oatom), oatommax);
  Expr lhs =  unsafe_promise_clamped(r.z - offsets(oatom), 0, nijmax - 1);
  //    fij(dim, lhs, oatom) += c * U(rzbounded, r.y, r.x, dim + 1);
  fij(dim, lhs, oatom) += c * UW(lhs, r.y, r.x, dim + 1, oatom);
  if (up != -1){
    fij.update(up).reorder(dim, r.z, r.y, r.x, oatom);
    //    UW.in(fij).compute_at(fij, oatom);
  }
}

void threeBodyCoeff(Func & cU, Func & e,
		    Func coeff3, Func sumU, Func pn3, Func pc3, Func ti, Func tA,
		    Expr npairs,
		    Var ne, Var k3, Var rbf3, Var oatom,
		    Expr nelements, Expr K3, Expr nrbf3, Expr nabf3, Expr me, Expr natoms)
{
  Expr zero = Expr((double) 0.0);

  cU(ne, k3, rbf3, oatom) = zero;

  cU.bound(ne, 0, nelements);
  cU.bound(k3, 0, K3);
  cU.bound(rbf3, 0, nrbf3);
  //  cU.bound(oatom, 0, natoms);

  //RDom r(0, nabf3, 0, K3, 0, nelements, 0, nelements, 0, nrbf3);
  RDom r(0, K3, 0, nabf3, 0, nelements, 0, nelements, 0, nrbf3);
  Expr n1 = pn3(r.y);
  Expr n2 = pn3(r.y + 1);
  r.where(n1 <= r.x);
  r.where(r.x < n2);
  r.where(r[3] >=  r.z);
    
  Expr k = (2 * nelements - 3 - r.z) * (r.z/ 2) + r[3] - 1; //mem  - ki + kij;
  Expr t1 = pc3(r.x) * sumU(r.z, r.x, r[4], oatom);
  Expr c2 = sumU(r[3], r.x, r[4], oatom);
  ///atom type map is needed.
  Expr c3 = coeff3(r.y, r[4], clamp(k, 0, me - 1), clamp(tA(oatom) - 1, 0, nelements - 1));
  Expr t2 = c3 * t1;
  e(oatom) += t2 * c2;
  cU(r[3], r.x, r[4], oatom) += t2;
  cU(r.z, r.x, r[4], oatom) += pc3(r.x) * c2 * c3;
}

void fourbodycoeff(Func & e4, Func  & cU4,
		   Func coeff4, Func  sumU4, Func ti, Func pa4, Func pb4, Func pc4, Func tA,
		   Expr nrbf4, Expr nabf4, Expr nelements, Expr k4, Expr Q4, Expr natoms,
		   Var oatom){

  Var ne("ne");
  Var kv("kv");
  Var rbf("rbf");
  Expr acc = clamp(tA(oatom) - 1, 0, nelements - 1);
  Expr zero = Expr((double) 0.0);
  cU4(ne, kv, rbf, oatom) = zero;
  //  cU4.bound(ne,0, nelements).bound(kv, 0, k4 + 1).bound(rbf, 0, nrbf4);//.bound(oatom, 0, natoms);
  e4(oatom) = zero;
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
  Expr j1 = unsafe_promise_clamped(pb4(n1pq, 0), 0, k4 - 1);
  Expr j2 = unsafe_promise_clamped(pb4(n1pq, 1), 0, k4 - 1);
  Expr j3 = unsafe_promise_clamped(pb4(n1pq, 2), 0, k4 - 1);
  Expr i1 = r[0];
  Expr i2 = r[1];
  Expr i3 = r[2];
  r.where(i1 >= i2 && i2 >= i3);
  Expr sym3NE = (nelements) * (nelements+1) * (nelements+2)/6;
  Expr k = unsafe_promise_clamped((i1*(i1+1)*(i1+2)/6) + (i2*(i2+1)/2) + i3, 0, sym3NE);

  Expr c1 = c * sumU4(i1, j1, rbfr, oatom);
  Expr c0 = sumU4(i2, j2, rbfr, oatom);
  Expr c2 = c * c0;
  Expr c4 = c1 * c0;
  Expr c5 = coeff4(p, rbfr, k, acc);
  Expr c6 = c5 * sumU4(i3, j3, rbfr, oatom);

  e4(oatom) += c4 * c6;
  Expr scat = scatter(0, 1, 2);
  Expr extra = select(i3 == i2 && j3 == j2, c5 * c4, Expr((double) 0.0));
  Expr extra1 = select(i1 == i2 && j1 == j2, c6*c1 + extra, extra);
  Expr init3 = cU4(i3, j3, rbfr, oatom);
  Expr init2 = cU4(i2, j2, rbfr, oatom);
  Expr init1 = cU4(i1, j1, rbfr, oatom);
  cU4(i3, j3, rbfr, oatom) += c5*c4;
  cU4(i2, j2, rbfr, oatom) += c6*c1;
  cU4(i1, j1, rbfr, oatom) += c6*c2;
  // cU4(mux(scat, {i3, i2, i1}), mux(scat, {j3, j2, j1}), rbfr, oatom) += gather(c5*c4,
  // 									       c6*c1 + extra,
  // 									       c6*c2 + extra1);
    
  
}

void threeBodyDescDeriv(Func & dd3,
			Func sumU, Func U, Func UW, Func atomtype, Func pn3, Func pc3,
			Func elemindex, Func offsets,
			Expr npairs, Expr q, Expr nelements,
			Var dim, Var nj, Var abf3,
			Expr nabf3,  Expr natoms, Expr nijmax,
			Var rbf3, Expr nrbf3, Var kme, Expr me,
			RDom r, Var oatom)
{
  Expr zero = Expr((double) 0.0);
    
  //Var rbfTres("rbftres");
  //dd3(dim, nj, abf3, rbfTres, kme) = zero;
  dd3(dim, nj, abf3, rbf3, kme, oatom) = zero;
    
  dd3.bound(dim, 0, 3);
  //  dd3.bound(nj, 0, npairs);
  dd3.bound(abf3, 0, nabf3);
  //dd3.bound(rbfTres, 0, nrbf3);
  dd3.bound(rbf3, 0, nrbf3);
  dd3.bound(kme, 0, me);
  //  dd3.bound(oatom, 0, natoms);
    
  //RDom r(0, nabf3, 0, q, 0, nelements, 0, npairs);
  //RDom r(0, q, 0, nabf3, 0, nelements, 0, npairs);
  Expr npmaxs = max(npairs - 1, 0);
  Expr off1 = offsets(oatom);
  Expr off2 = offsets(oatom+1);

  r.where(off1 <= r[3]);
  r.where(r[3] < off2);
  Expr n1 = pn3(r.y);
  Expr n2 = pn3(r.y + 1);
  
  r.where(n1 <= r.x); 
  r.where(r.x < n2);	       
  RVar ry = r.y;
  RVar rx = r.x;
  RVar rz = r.z;
  RVar rzz = r[3];

  Expr oatomDiff = min(offsets(oatom + 1), nijmax+ offsets(oatom));
  Expr rzbounded = unsafe_promise_clamped(rzz, offsets(oatom), min(offsets(oatom+1), nijmax+offsets(oatom)) - 1);

  //Expr rzlhs = unsafe_promise_clamped(rzz - offsets(oatom), nijmax);

  //  Expr rzz = clamp(clamp(rzzPre, offsets(oatom), offsets(oatom) + nijmax), 0, npairs);
  //Expr t1 = pc3(rx) * sumU(rz, rx, rbfTres);
  Expr t1 = pc3(rx) * sumU(rz, rx, rbf3, oatom);
  Expr i2 = atomtype(rzz) - 1;
  Expr k = elemindex(clamp(i2, 0, nelements - 1), rz);
  Expr f = select(rz == i2, 2 * t1, t1);

  Expr rhsBound = min(nijmax, offsets(oatom+1)-offsets(oatom));
  Expr rzzlhs = unsafe_promise_clamped(rzz - offsets(oatom), 0, rhsBound - 1);
  //dd3(dim, rzz, ry, rbfTres, clamp(k, 0, me - 1)) += f * U(rzz, rx, rbfTres, dim + 1);
  //  Expr erg = unsafe_promise_clamped(rzzlhs + unsafe_promise_clamped(offsets(oatom),0, npairs-1),min(offsets(oatom), 0), min(oatomDiff, npairs));
  //  Expr oatommax = min(nijmax + offsets(oatom), offsets(oatom+1)) - 1;
  //  Expr erg = unsafe_promise_clamped(clamp(rzz, 0, npairs -1), offsets(oatom), oatommax);
  //  Expr oatommax = min(nijmax + offsets(oatom), offsets(oatom+1)) - 1;
  //  Expr erg = unsafe_promise_clamped(clamp(rzz, 0, npairs -1), off1, off2 - 1);
  //  Func ergbound(nj, oatom) =
  //  Expr erg = unsafe_promise_clamped(unsafe_promise_clamped(rzz, off1,  ), off1, off1 + nijmax - 1);
  
  dd3(dim, rzzlhs, ry, rbf3, clamp(k, 0, me - 1), oatom) += f * UW(rzzlhs, rx, rbf3, dim + 1, oatom);
  //  UW.in(dd3).compute_at(dd3, oatom);
}

void threeBodyDesc(Func & d3,
		   Func sumU, Func pn3, Func pc3,
		   Expr npairs, Expr nelements, Expr nrbf3, Expr nabf3, Expr k3, Expr natoms,
		   Var abf3, Var rbf3, Var kme, Var oatom)
{
  Expr zero = Expr((double) 0.0);
  Expr me = nelements * (nelements + 1)/2;
  Expr mem = nelements * (nelements - 1)/2;
  // Var abf3("abf3");
  // Var rbf3("rbf3");
  // var kme("kme");

  d3(abf3, rbf3, kme, oatom) = zero;

  d3.bound(abf3, 0, nabf3);
  d3.bound(rbf3, 0, nrbf3);
  d3.bound(kme, 0, me);
  //  d3.bound(oatom, 0, natoms);

  //RDom r(0, nabf3, 0, k3, 0, nelements, 0, nelements);
  RDom r(0, k3, 0, nabf3, 0, nelements, 0, nelements);
  Expr n1 = pn3(r.y);
  Expr n2 = pn3(r.y + 1);
  r.where(n1 <= r.x);
  r.where(r.x < n2);
  RVar ry = r.y;
  RVar rx = r.x;
  RVar rz = r.z;
  RVar rzz = r[3];
  r.where(rzz <= rz);

  Expr t1 = pc3(rx) * sumU(rz, rx, rbf3, oatom);
  //  Expr ki = (nelements - rz) * ((nelements - rz) - 1)/2;
  //  Expr kij = rzz - rz - 1;
  Expr k = (2 * nelements - 3 - rz) * (rz/ 2) + rzz - 1; //mem  - ki + kij;
  Expr t2 = sumU(rzz, rx, rbf3, oatom);
  d3(ry, rbf3, clamp(k, 0, me -1), oatom) += t1 * t2;
  
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
  indexmap.bound(k, 0, n1 * n2 * n3);
  indexmap.bound(c, 0, 3);
  indexmap.compute_root();
  //  indexmap.
}


void fourbodystuff(Func & fij, Func & e23, Func & d23,
		   Func ind23, Func ind32, Func coeff23, Func d2, Func d3, Func dd3, Func dd2,
		   Func ti, Func tA, Func offsets,
		   Expr npairs, Expr n23, Expr n32, Expr nelements, Expr nrbf2, Expr nrbf3,
		   Expr nabf3, Expr me, Expr natoms, Expr nijmax, 
		   Var pairindex, Var oatom)
{

  Var d23i("d23i");
  Var d23j("d23j");
  e23(oatom) = Expr((double) 0.0);
  ///Needs type of oatom.
  Expr acc = unsafe_promise_clamped(tA(oatom) - 1, 0, nelements - 1);
  //Use unsafe_promise_clamped
  // j is n32
  //i is n23
  d23(d23i, d23j, oatom) = d2(unsafe_promise_clamped(ind23(d23i, 1), 0, nelements -1),
			      unsafe_promise_clamped(ind23(d23i, 2), 0, nrbf2 -1), oatom) * d3(unsafe_promise_clamped(ind32(d23j, 0), 0, nabf3- 1),
											       unsafe_promise_clamped(ind32(d23j, 1),0, nrbf3 - 1),
											       unsafe_promise_clamped(ind32(d23j, 2), 0, me -1),
											       oatom);
  d23.bound(d23i, 0, n23);
  d23.bound(d23j, 0, n32);

  //  d23.compute_root();


  RDom e23rdom(0, n23, 0, n32);
  e23(oatom) += d23(e23rdom.x, e23rdom.y,oatom) * coeff23(e23rdom.x, e23rdom.y, acc);

  Expr zero = Expr((double) 0.0);
  Func cf1("cf1");
  Var j("j");
  Var dim("dim");
  RDom r1(0, n23);
  RDom r1f(0, n23, 0, nijmax);
  r1f.where(r1f.y < offsets(oatom + 1) - offsets(oatom));
  cf1(j, oatom) = zero;
  cf1(j, oatom) += d2(unsafe_promise_clamped(ind23(r1.x, 1), 0, nelements -1), unsafe_promise_clamped(ind23(r1.x, 2), 0, nrbf2 -1), oatom) * coeff23(r1.x, j, acc);

  // RDom r2(0, n32);
  // fij(pairindex, dim) += cf1(r2.x) * dd3(dim, pairindex, ind32(r2.x,0), ind32(r2.x,1), ind32(r2.x,2));
  RDom r2(0, n32);
  RDom r2f(0, n32, 0, nijmax);
  r2f.where(r2f.y < offsets(oatom + 1) - offsets(oatom));
  //Rdom pairindex -> oatom
  Expr oatommax = min(offsets(oatom + 1), nijmax + offsets(oatom)) - 1;
  Expr r2fy =  unsafe_promise_clamped(unsafe_promise_clamped(r2f.y + offsets(oatom), 0, min(npairs - 1, offsets(oatom + 1))), offsets(oatom), oatommax);
  //Expr r2fyd = unsafe_promise_clamped()
  fij(dim, r2f.y, oatom) += cf1(r2f.x, oatom) * dd3(dim, r2f.y,
						    unsafe_promise_clamped(ind32(r2f.x, 0), 0, nabf3- 1),
						    unsafe_promise_clamped(ind32(r2f.x, 1),0, nrbf3 - 1),
						    unsafe_promise_clamped(ind32(r2f.x, 2), 0, me -1),
						    oatom);

  Func cf2("cf2");
  Var i("i");
  cf2(i, oatom) = zero;
  cf2(i, oatom) += d3( unsafe_promise_clamped(ind32(r2.x, 0), 0, nabf3- 1),
		       unsafe_promise_clamped(ind32(r2.x, 1),0, nrbf3 - 1),
		       unsafe_promise_clamped(ind32(r2.x, 2), 0, me -1),
		       oatom) * coeff23(i, r2.x, acc);

  //needs to inverse pairindex to atom...
  Expr offBound = min(offsets(oatom + 1), nijmax + offsets(oatom)) - 1; //min(offsets(oatom + 1) - offsets(oatom), nijmax);
  Expr offsetedAcc = unsafe_promise_clamped(unsafe_promise_clamped(r1f.y + offsets(oatom), 0, npairs),  offsets(oatom), offBound - 1);
  fij(dim, r1f.y, oatom) += cf2(r1f.x, oatom) * dd2(dim, r1f.y,
						    unsafe_promise_clamped(ind23(r1f.x, 1), 0, nelements -1),
						    unsafe_promise_clamped(ind23(r1f.x, 2), 0, nrbf2 -1),
						    oatom);

  cf2.compute_at(fij, oatom);
  cf1.compute_at(fij, oatom);
  fij.update(2).reorder(dim, r2f.x, r2f.y, oatom);
  fij.update(3).reorder(dim, r1f.x, r1f.y, oatom);
  //  dd2.compute_at(fij, oatom);
  
}


void fivebodystuff(Func & fij, Func & e33, Func & d33,
		   Func ind33, Func coeff33, Func d3, Func ti, Func dd3, Func offsets, Func tA,
		   Expr npairs, Expr nijmax, Expr n33, Expr nelements, Expr nabf3, Expr nrbf3, Expr me, Expr natoms, 
		   Var pairindex, Var oatom)
{
  Expr symMe = nelements*(nelements+1)/2;
  Expr symN33 = n33 * (n33+1)/2;
  Var kv("k");
  Var dim("dim");
  d33(kv, oatom) = Expr((double) 0.0);
  d33.bound(kv, 0, symN33);
  //  d33.bound(oatom, 0, natoms);
  RDom r(0, n33, 0, n33, "fivebodysym");
  r.where(r.x >= r.y);
  //col + row*(M-1)-row*(row-1)/2
  //https://stackoverflow.com/questions/242711/algorithm-for-index-numbers-of-triangular-matrix-coefficients
  Expr temp = clamp(r.x + r.y * (n33 - 1) - r.y * (r.y-1)/2, 0, symN33 - 1);
  Expr k =  temp; //print_when(temp < 0 || temp > symN33- 1, temp, "error!"); //(print(temp, symN33, r.x, r.y));

  Expr kp = k;//clamp((2 * n33 - 3 - r.y) * (r.y/ 2) + r.x - 1, 0, symN33 - 1); //mem  - ki + kij;
  //abf, rbf, me
  Expr abfacc1 = unsafe_promise_clamped(ind33(r.x, 0), 0, nabf3 - 1);
  Expr abfacc2 = unsafe_promise_clamped(ind33(r.y, 0), 0, nabf3 - 1);
  Expr rbfacc1 = unsafe_promise_clamped(ind33(r.x, 1), 0, nrbf3 - 1);
  Expr rbfacc2 = unsafe_promise_clamped(ind33(r.y, 1), 0, nrbf3 - 1);
  Expr meacc1 = unsafe_promise_clamped(ind33(r.x, 2), 0, symMe - 1);
  Expr meacc2 = unsafe_promise_clamped(ind33(r.y, 2), 0, symMe - 1);
  d33(k, oatom) = d3(abfacc2, rbfacc2, meacc2, oatom) * d3(abfacc1, rbfacc1, meacc1, oatom);

  //  d33.compute_root();

  RDom rdot(0, symN33);
  Expr acc = clamp(tA(oatom) - 1, 0, nelements - 1);
  e33(oatom) = Expr((double) 0.0);
  e33(oatom) += d33(rdot, oatom) * coeff33(rdot, acc);


  Func cf133("cf133");
  Var j("j");
  cf133(j, oatom) = Expr((double) 0.0);
  cf133(r.y, oatom) += d3(abfacc1, rbfacc1, meacc1, oatom) * coeff33(kp, acc);
  //  cf133.compute_root();

  RDom rn33(0, n33, 0, nijmax);
  rn33.where(rn33.y < offsets(oatom + 1) - offsets(oatom));
  Expr abfacc3 = unsafe_promise_clamped(ind33(rn33.x, 0), 0, nabf3 - 1);
  Expr rbfacc3 = unsafe_promise_clamped(ind33(rn33.x, 1), 0, nrbf3 - 1);
  Expr meacc3 = unsafe_promise_clamped(ind33(rn33.x, 2), 0, symMe - 1);
  Expr oatommax = min(offsets(oatom + 1), nijmax + offsets(oatom)) - 1;
  Expr rn33y = unsafe_promise_clamped(unsafe_promise_clamped(rn33.y + offsets(oatom), offsets(oatom), oatommax), 0, npairs - 1);
  

  
  fij(dim, rn33.y, oatom) += cf133(rn33.x, oatom) * dd3(dim, rn33.y, abfacc3, rbfacc3, meacc3, oatom);
  cf133.compute_at(fij, oatom);
  fij.update(4).reorder(rn33.x, rn33.y, oatom);

  Func cf233("cf233");
  Var i("j");
  cf233(i, oatom) = Expr((double) 0.0);
  cf233(r.x, oatom) += d3(abfacc2, rbfacc2, meacc2, oatom) * coeff33(kp, acc);
  //  cf233.compute_root();

  fij(dim, rn33.y, oatom) += cf233(rn33.x, oatom) * dd3(dim, rn33.y, abfacc3, rbfacc3, meacc3, oatom);
  fij.update(5).reorder(rn33.x, rn33.y, oatom);

  cf233.compute_at(fij, oatom);
  //  dd3.compute_at(fij, rn33.y);

  //  d33.compute_at(fij, oatom);

}

class  poddescOuter : public Halide::Generator<poddescOuter> {
public:

  Input<Buffer<double>> rijs{"rijs", 2};
  Input<Buffer<double>> besselparams{"besselparams", 1};
  Input<int> nbesselparams{"nbesselpars", 1};
  Input<int> bdegree{"bdegree", 1};
  Input<int> adegree{"adegree", 1};
  Input<int> npairs{"npairs", 1};
  Input<int> natoms{"natoms", 1};
  Input<int> nrbfmax{"nrbfmax", 1};
  Input<int> nijmax{"nijmax"};
  Input<double> rin{"rin", 1};
  Input<double> rcut{"rcut", 1};

  Input<Buffer<double>> Phi{"Phi", 2};
  Input<int> ns{"ns", 1};
  

  Input<Buffer<int>> ti{"ti", 1};
  Input<Buffer<int>> tj{"tj", 1};
  Input<Buffer<int>> ai{"ai", 1};
  Input<Buffer<int>> aj{"aj", 1};
  Input<Buffer<int>> offsets{"offsets", 1};
  Input<Buffer<int>> tA{"tA", 1};

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

  Input<int> nrbf2{"nrbf2", 1};
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

  Input<Buffer<double>> coeff1{"coeff1", 1};  
  Input<Buffer<double>> coeff2{"coeff2", 3};  
  Input<Buffer<double>> coeff3{"coeff3", 4};
  Input<Buffer<double>> coeff23{"coeff23", 3};
  Input<Buffer<double>> coeff33{"coeff33", 2};
  Input<Buffer<double>> coeff4{"coeff4", 4};
  Input<Buffer<double>> coeff34{"coeff34", 1};
  Input<Buffer<double>> coeff44{"coeff44", 1};

  Output<Buffer<double>> fij_o{"fij_o", 2};
  Output<Buffer<double>> e_o{"e_o", 1};
  Output<Buffer<double>> etot_o{"etot", 0};

  //  Output<Buffer<double>> fife_o{"fij_o", 2};


  void generate() {
    Var oatom("oatom");
    

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
    buildRBF(rbft, rijs, besselparams, offsets,
	     rin, rcut-rin,
	     bdegree, adegree, nbesselparams, npairs, ns, nijmax,
	     bfi, bfp, np, dim, oatom);

    // MatMul
    //rbft.dim(2).set_bounds(0, 4).set_stride(npairs * ns);
    //rbft.dim(1).set_bounds(0, ns).set_stride(npairs);
    //rbft.dim(0).set_bounds(0, npairs).set_stride(1);
    Var i("i");
    Var j("j");
    Var k("k");
    Var c("c");
    Func prod("prod");
    prod(c, k, i, j, oatom) = Phi(k, i) * rbft(j, k, c, oatom);
    Func rbf("rbf");
    rbf(j, i, c, oatom) = Expr((double) 0.0);
    RDom r(0, ns);
    rbf(j, i, c, oatom) += prod(c, r, i, j, oatom);
    //j is num pairs so we sum over basis functions here...
    //rbf.update(0).reorder(c, r, i, j); // TODO
    

    Func e2("e2");
    Func fijAtom("fijAtom");
    fijAtom(dim, np, oatom) = Expr((double) 0.0);
    fijAtom.bound(dim, 0, 3).bound(np, 0, nijmax);
    tallyTwoBodyLocalForce(fijAtom, e2,
			   coeff2, rbf, tj, tA, aj, offsets,
			   nrbf2, npairs, nelements, natoms, nijmax, oatom);


    Var n("n");
    Func abf4("abf4");
    Func tm("tm");
    Var abfi("abfi");
    Var abfip("abfip");
    buildAngularBasis(k3, npairs, nijmax,
		      pq, rijs, offsets,
		      abf4, tm,
		      c, np,  abfi, abfip, oatom
		      );


    Func sumU("sumU"), U("U");
    Var mrab("m"), ne("ne"), crab("crab");
    radialAngularBasis(sumU, U, rbf, abf4,
		       tj, aj, offsets,
		       n, k, mrab, ne, crab,
		       npairs, k3, nrbf3, nelements,
		       natoms,nijmax,
		       oatom);
    Var a0("a0");
    Var a1("a1");
    Var a2("a2");
    Var a3("a3");
    Func UW("UW");
    Expr offset2 = min(nijmax + offsets(oatom), offsets(oatom+1)) - 1;
    //unsafe_promise_clamped(unsafe_promise_clamped(a0 +offsets(oatom), offsets(oatom), offset2), 0, npairs-1)
    UW(a0,a1,a2,a3, oatom) = U(a0, a1, a2, a3, oatom);
    //    UW.bound()




    Func d2("d2"), dd2("dd2");
    twoBodyDescDeriv(d2, dd2,
		     rbf, tj, aj, offsets,
		     npairs, nelements, nrbf2, natoms, nijmax, oatom);


    Func d3("d3");
    Var abfThree("abfThree");
    Var rbfThree("rbfThree");
    Var kme("kme");
    threeBodyDesc(d3,
		  sumU, pn3, pc3,
		  npairs, nelements, nrbf3, nabf3, k3, natoms,
		  abfThree, rbfThree, kme, oatom);

    
    //    d3.compute_root();
    
    Func dd3("dd3");
    Var nj("nj");    
    RDom r3body(0, k3, 0, nabf3, 0, nelements, 0, npairs, "r3bodyRdom");
    threeBodyDescDeriv(dd3,
		       sumU, U, UW, tj, pn3, pc3,
		       elemindex, offsets,
		       npairs, k3, nelements,
		       dim, nj, abfThree,
		       nabf3, natoms, nijmax,
		       rbfThree, nrbf3, kme, me,
		       r3body, oatom);



    dd3.update(0).reorder(dim, r3body.x, r3body.y, rbfThree, r3body[3], oatom);
 
    Func cU("cU");
    Func e3("e3");
    Var k3var("k3var");



    threeBodyCoeff(cU, e3,
		   coeff3, sumU, pn3, pc3, ti, tA,
		   nj,
		   ne, k3var, rbfThree, oatom,
		   nelements, k3, nrbf3, nabf3, me, natoms);



    tallyLocalForceRev(fijAtom,
		       tj,aj, offsets,
		       cU, U, UW,
		       nrbf3, k3, npairs, nelements, natoms, nijmax,
		       oatom, dim, 1);

   

    Func e23("e23");
    Func d23("d23");
    fourbodystuff(fijAtom, e23, d23,
		  ind23, ind32, coeff23, d2, d3, dd3, dd2,
		  ti, tA, offsets,
		  npairs, n23, n32, nelements, nrbf2, nrbf3,
		  nabf3,me, natoms, nijmax,
		  n, oatom);


    Func e33("e33");
    Func d33("d33");
    fivebodystuff(fijAtom, e33, d33,
		  
		  ind33, coeff33, d3, ti, dd3, offsets, tA,
		  npairs, nijmax, n33, nelements,
		  nabf3, nrbf3, me, natoms,
		  n, oatom);




    Func sumU4("sumU4");
    Func U4("u4");
    Var m4v("m4");
    Var k4v("k4");
    Var e4v("e4");
    Var dim4v("dim4");


    sumU4(m4v, k4v, e4v, oatom) = sumU(m4v, k4v, e4v, oatom);
    sumU4.bound(m4v, 0, nelements).bound(k4v, 0, min(k3, k4)).bound(e4v, 0, min(nrbf4, nrbf3)); //.bound(oatom, 0, natoms);
    U4(m4v, k4v, e4v, dim4v, oatom) = U(m4v, k4v, e4v, dim4v, oatom);
    U4.bound(k4v, 0, min(k3, k4)).bound(e4v, 0, min(nrbf4, nrbf3)).bound(dim4v, 0, 4);

    Func cu4("cu4");
    Func e4("e4");

    fourbodycoeff(e4, cu4,
		  coeff4, sumU4, ti, pa4, pb4, pc4, tA,
		  nrbf4, nabf4, nelements, k4, q4, natoms, oatom);


    tallyLocalForceRev(fijAtom,
		       tj, aj, offsets,
		       cu4, U4, UW,
		       nrbf4, k4, npairs, nelements, natoms, nijmax,
		       oatom, dim, 6);


 

    Func etemp("etemp");
    etemp(oatom) =  e33(oatom) + e23(oatom) + e4(oatom) + e3(oatom) + e2(oatom) + coeff1(clamp(tA(oatom) - 1, 0, nelements-1));
    RDom rout(0, nijmax, 0, 7, 0, natoms, "finrdom");
    rout.where(rout.x < offsets(rout.z + 1) - offsets(rout.z));
    Expr npp = clamp(rout.x + offsets(rout.z), 0, npairs - 1);
    Expr app = clamp(aj(npp), 0, natoms - 1);
    
    //0,1,2 = force
    //3,4,5 = force again
    //6 = energy
    Expr dimp = rout.y % 3;

    Func fife_o("fife_o");
    fife_o(oatom, dim) = Expr((double) 0.0);
    Expr lhs = fife_o(select(rout.y == 6, rout.z,
			     select(rout.y > 2 && rout.y < 6, app,
				    select(rout.y <= 2, rout.z, 0))),
		      select(rout.y == 6, 3, dimp)
		      );
    fife_o(select(rout.y == 6, rout.z,
		  select(rout.y > 2 && rout.y < 6, app,
			 select(rout.y <= 2, rout.z, 0))),
	   select(rout.y == 6, 3, dimp)
	   ) = select(rout.y == 6, etemp(rout.z),
		      select(rout.y > 2 && rout.y < 6, lhs + -1 * fijAtom(dimp, rout.x, rout.z),
			     select(rout.y <= 2, lhs+fijAtom(dimp, rout.x, rout.z), Expr((double) -1.0))));
    

    fife_o.compute_root();
    fife_o.update(0).atomic(true).parallel(rout.z);
    fife_o.update(0).unroll(rout.y);
    etemp.compute_at(fife_o, rout.z);
    fijAtom.compute_at(fife_o, rout.z);
    cu4.compute_at(fife_o, rout.z);
    cU.compute_at(fife_o, rout.z);
    sumU.compute_at(fife_o, rout.z);
    d33.compute_at(fife_o, rout.z);
    d23.compute_at(fife_o, rout.z);
    d3.compute_at(fijAtom, oatom);
    d3.compute_at(fife_o, rout.z);
    dd3.compute_at(fife_o, rout.z);
    dd2.compute_at(fife_o, rout.z);
    d2.compute_at(fife_o, rout.z);
    //    UW.compute_at(fife_o, rout.z);
    //U.compute_at(fife_o, rout.z);
    ///    U.in(fijAtom).compute_at(, oatom);

    abf4.compute_at(fife_o, rout.z);
    tm.compute_at(fife_o, rout.z);
    rbf.compute_at(fife_o, rout.z);
    rbft.compute_at(fife_o, rout.z);

    fij_o.dim(0).set_bounds(0, natoms).set_stride(3);
    fij_o.dim(1).set_bounds(0, 3).set_stride(1);
    fij_o(oatom, dim) = fife_o(oatom, dim);
    fij_o.bound(dim, 0, 3);
    fij_o.reorder(dim, oatom);
    e_o(oatom) = fife_o(oatom, 3);
    e_o.dim(0).set_bounds(0, natoms).set_stride(1);

    RDom outrdom(0, natoms, "out");
    etot_o() += e_o(outrdom);


  }
};

HALIDE_REGISTER_GENERATOR(poddescOuter, poddescOuter);
