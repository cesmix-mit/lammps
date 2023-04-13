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
	       Func xij, Func besselparams, Expr rin, Expr rmax,
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
  drbf(bfp, bfi, np, dim) = (xij(dim, np)/dij) * drbfdr;
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
  dabf(bfi, np, dim) = (xij(dim, np)/dij) * drbfdr_a;
  // dabf.trace_stores();
  dabf.bound(dim, 0, 3);
  dabf.bound(bfi, 0, adegree);
  dabf.bound(np, 0, npairs);

  rbf.compute_root();

  drbf.compute_root(); // .021 ms (10%)
  //drbf.compute_at(xij, np); // .022 ms (11%)
  // nothing ? // Runtime looks like increases slightly and info gets pushed into `rbft`

  abf.compute_root();
  
  dabf.compute_root();

  // rbf.size() = nbparams * bdegree * npairs
  // drbf[x].size() = nbparams * bdegree * npairs
  // abf.size() = adegree * npairs 
  // dabf[x].size() = adegree * npairs
  // output.size() = nbparams * bdgree * npairs + adegree * npairs
  Var rbf_abf_info("rbf_abf_info"), drbf_dabf_info("drbf_dabf_info");
  Var rbfty("rbfty");
  RDom r1(0, nbparams, 0, bdegree, 0, npairs);
  RDom r2(0, npairs, 0, adegree);
  Var abf_index("abf_index");
  Expr rbf_info_length = nbparams * bdegree;
  Expr nsp = rbf_info_length + adegree;
  // Set up rbf_info
  rbfall(np, rbf_abf_info, rbfty) = zero;
  rbfall.bound(rbfty, 0, 4);
  rbfall.bound(np, 0, npairs);
  rbfall.bound(rbf_abf_info, 0, ns);

  rbfall(r1.z, r1.y + r1.x * bdegree, 0) = rbf(r1.x, r1.y, r1.z);
  rbfall(r2.x, get_abf_index(r2.y, rbf_info_length), 0) = abf(r2.y, r2.x);
  rbfall(r1.z, r1.y + r1.x * bdegree, 1) = drbf(r1.x, r1.y, r1.z, 0);
  rbfall(r2.x, get_abf_index(r2.y, rbf_info_length), 1) = dabf(r2.y, r2.x, 0);
  rbfall(r1.z, r1.y + r1.x * bdegree, 2) = drbf(r1.x, r1.y, r1.z, 1);
  rbfall(r2.x, get_abf_index(r2.y, rbf_info_length), 2) = dabf(r2.y, r2.x, 1);
  rbfall(r1.z, r1.y + r1.x * bdegree, 3) = drbf(r1.x, r1.y, r1.z, 2);
  rbfall(r2.x, get_abf_index(r2.y, rbf_info_length), 3) = dabf(r2.y, r2.x, 2);

// This seems like it was the most important thing that needed to be changed 
  // rbfall.compute_root();  // 19353.844 ms .001 ms 0%
  rbfall.store_root().compute_root();  // 19247.139 ms .001 ms 0%
  // nothing? TERRIBLE TERRIBLE TERRIBLE, possibly bottleneck from earlier? 295 seconds total! with rbft taking 2.378 ms (77%)
}


void radialAngularBasis(Func & sumU, Func & U,

			Func rbf, Func abf,  Func atomtype,
			Expr N, Expr K, Expr M, Expr Ne)
{
  Expr zero = Expr((double) 0.0);

  Var n("n"), k("k"), m("m"), ne("ne"), c("crab");
  sumU(ne, k, m) = zero;

  //Expr c1 = rbf(m, n);
  //Expr c2 = abf(k, n);
  Expr c1 = rbf(n, m, 0);
  Expr c2 = abf(n, k, 0);
    
  // U(m, k, n) = c1 * c2;
  // Ux(m, k, n) = abfx(k, n) * c1 + c2 * rbfx(m, n);
  // Uy(m, k, n) = abfy(k, n) * c1 + c2 * rbfy(m, n);
  // Uz(m, k, n) = abfz(k, n) * c1 + c2 * rbfz(m, n);
  U(n, k, m, c) = select(c == 0, c1 * c2,
			 select(c == 1, abf(n, k, 1) * c1 + c2 * rbf(n, m, 1),
				select(c== 2, abf(n, k, 2) * c1 + c2 * rbf(n, m, 2) ,
				       select(c==3, abf(n, k, 3) * c1+ c2 * rbf(n, m, 3), Expr((double) 0.0)))));
  // Ux(m, k, n) = abfx(n, k) * c1 + c2 * rbfx(n, m);
    
  // Uy(m, k, n) = abfy(n, k) * c1 + c2 * rbfy(n, m);
  // Uz(m, k, n) = abfz(n, k) * c1 + c2 * rbfz(n, m);

  RDom r(0, M, 0, K, 0, N);
  Expr in = atomtype(r.z) - 1;

  // sumU(r.x, r.y, clamp(in, 0, Ne - 1)) += rbf(r.x, r.z) * abf(r.y, r.z);
  sumU(clamp(in, 0, Ne - 1), r.y, r.x) += rbf(r.z, r.x, 0) * abf(r.z, r.y, 0);

  sumU.bound(m, 0, M);
  U.bound(m, 0, M);

  sumU.bound(k, 0, K);
  U.bound(k, 0, K);

  sumU.bound(ne, 0, Ne);
  U.bound(n, 0, N);

  U.bound(c, 0, 4);

  U.compute_root();
  //nothing?  18736.008 ms -- .011 ms 5%
  // U.compute_at(U, n);
  sumU.compute_root();

}

void twoBodyDescDeriv(Func & d2, Func & dd2, Func rbf,  Func tj, Expr N, Expr Ne, Expr nrbf2)
{
  Expr zero = Expr((double) 0.0);

  Var ne("ne"), m("m"), n("n"), dim("dim");
  d2(ne, m) = zero;
  dd2(ne, m, n, dim) = zero;

  RDom r(0, N, 0, nrbf2);

  d2(clamp(tj(r.x)-1, 0, Ne - 1), r.y) += rbf(r.x, r.y, 0);
  dd2(clamp(tj(r.x)-1, 0, Ne - 1), r.y, r.x, 0) += rbf(r.x, r.y, 1);
  dd2(clamp(tj(r.x)-1, 0, Ne - 1), r.y, r.x, 1) += rbf(r.x, r.y, 2);
  dd2(clamp(tj(r.x)-1, 0, Ne - 1), r.y, r.x, 2) += rbf(r.x, r.y, 3);

  d2.bound(ne, 0, Ne);
  d2.bound(m, 0, nrbf2);

  dd2.bound(ne, 0, Ne);
  dd2.bound(m, 0, nrbf2);
  dd2.bound(n, 0, N);
  dd2.bound(dim, 0, 3);
    
  d2.compute_root();
  dd2.compute_root();
}


void tallyTwoBodyLocalForce(Func & fij, Func & e, Func coeff2, Func rbf, Func tj, Expr nbf, Expr N)
{
  Expr zero = Expr((double) 0.0);

  Var n("n"), m("m"), dim("dim"), empty("empty");
  e() = zero;
  fij(n, dim) = zero;

  RDom r(0, N, 0, nbf);

  Expr c = coeff2(clamp(tj(r.x), 1, N - 1) - 1, r.y);
  e() += c * rbf(r.x, r.y, 0);
  fij(r.x, dim) += c * rbf(r.x, r.y, dim + 1);

  fij.bound(n, 0, N);
  fij.bound(dim, 0, 3);

  fij.compute_root();
  e.compute_root();
}

class poddescTallyTwoBodyLocalForce : public Halide::Generator<poddescTallyTwoBodyLocalForce> {
public:
  Output<Buffer<double>> fij_o{"fij_o", 2};
  Output<double> e_o{"e_o"};

  Input<Buffer<double>> coeff2{"coeff2", 2};
  Input<Buffer<double>> rbf{"rbf", 3};

  Input<Buffer<int>> tj{"tj", 1};

  Input<int> nbf{"nbf", 1};
  Input<int> N{"N", 1};
  Input<int> ns{"ns", 1};
  Input<int> nrbfmax{"nrbfmax", 1};

  void generate() {
    rbf.dim(0).set_bounds(0, N).set_stride(1);
    rbf.dim(1).set_bounds(0, nrbfmax).set_stride(N);
    rbf.dim(2).set_bounds(0, 4).set_stride(N * nrbfmax);

    coeff2.dim(0).set_bounds(0, N).set_stride(nbf);
    coeff2.dim(1).set_bounds(0, nbf).set_stride(1);

    Func fij("fij"), e("e");
    tallyTwoBodyLocalForce(fij, e, coeff2, rbf, tj, nbf, N);

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

  Input<Buffer<double>> rbf{"rbf", 3};

  Input<Buffer<int>> tj{"tj", 1};
    
  Input<int> N{"N", 1};
  Input<int> Ne{"Ne", 1};
  Input<int> nrbf2{"nrbf2", 1};
  Input<int> ns{"ns", 1};
  Input<int> nrbfmax{"nrbfmax", 1};

  void generate() {
    rbf.dim(0).set_bounds(0, N).set_stride(1);
    rbf.dim(1).set_bounds(0, nrbfmax).set_stride(N);
    rbf.dim(2).set_bounds(0, 4).set_stride(N * nrbfmax);

    Func d2("d2"), dd2("dd2");
    twoBodyDescDeriv(d2, dd2, rbf, tj, N, Ne, nrbf2);

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
  Output<Buffer<double>> U_o{"U", 4};
  
  Input<Buffer<double>> rbf{"rbf", 3};
  Input<Buffer<double>> abf4{"abf", 3};
  Input<Buffer<int>> tj{"tj", 1};
  Input<int> npairs{"npairs", 1};
  Input<int> k3{"k3", 1};
  Input<int> nrbf3{"nrbf3", 1};
  Input<int> nrbfmax{"nrbfmax", 1};
  Input<int> nelements{"nelements", 1};
  Input<int> ns{"ns", 1};

  void generate() {
    rbf.dim(2).set_bounds(0, 4).set_stride(nrbfmax * npairs);
    rbf.dim(1).set_bounds(0, nrbfmax).set_stride(npairs);
    rbf.dim(0).set_bounds(0, npairs).set_stride(1);
    abf4.dim(2).set_bounds(0, 4).set_stride(k3* npairs);
    abf4.dim(1).set_bounds(0, k3).set_stride(npairs);
    abf4.dim(0).set_bounds(0, npairs).set_stride(1);



    Func sumU("sumU"), U("U");
    radialAngularBasis(sumU, U,rbf,abf4,
		       tj, npairs, k3, nrbf3, nelements);

    Var m("m"), k("k"), n("n"), ne("ne"), c("c");
    sumU_o(ne, k, m) = sumU(ne, k, m);
    U_o(n, k, m, c) = U(n, k, m, c);
    sumU_o.dim(0).set_bounds(0, nelements).set_stride(1);
    sumU_o.dim(1).set_bounds(0, k3).set_stride(nelements);
    sumU_o.dim(2).set_bounds(0, nrbf3).set_stride(nelements * k3);
    U_o.dim(0).set_bounds(0, npairs).set_stride(1);
    U_o.dim(1).set_bounds(0, k3).set_stride(npairs);
    U_o.dim(2).set_bounds(0, nrbf3).set_stride(npairs * k3);
    U_o.dim(3).set_bounds(0, 4).set_stride(npairs * k3 * nrbf3);
  }
};

class poddescRBF : public Halide::Generator<poddescRBF> {
public:

  Output<Buffer<double>> rbf_o{"rbf_f", 3};
    
  Input<Buffer<double>> rijs{"rijs", 2};
  Input<Buffer<double>> besselparams{"besselparams", 1};
  
  Input<int> nbesselparams{"nbesselpars", 1};
  Input<int> bdegree{"bdegree", 1};
  Input<int> adegree{"adegree", 1};
  Input<int> npairs{"npairs", 1};
  Input<int> ns{"ns"};
  Input<double> rin{"rin", 1};
  Input<double> rcut{"rcut", 1};

  void generate() {
    rijs.dim(1).set_bounds(0, npairs).set_stride(3);
    rijs.dim(0).set_bounds(0, 3).set_stride(1);


    besselparams.dim(0).set_bounds(0, nbesselparams);
    Var bfi("basis function index");
    Var bfp("basis function param");
    Var np("pairindex");
    Var numOuts("numOuts");
    Var dim("dim");

    Func rbf_f("rbf_f");
    buildRBF(rbf_f,
             rijs, besselparams, rin, rcut-rin,
             bdegree, adegree, nbesselparams, npairs, ns,
             bfi, bfp, np, dim);

    Var rbf_output("rbf_output");
    Var rbf_outputp("rbf_outputp");
    Var rbf_outputpp("rbf_outputpp");

    rbf_o(rbf_outputpp, rbf_outputp, rbf_output) = rbf_f(rbf_outputpp, rbf_outputp, rbf_output);
    rbf_o.dim(2).set_bounds(0, 4).set_stride(npairs * ns);
    rbf_o.dim(1).set_bounds(0, ns).set_stride(npairs);
    rbf_o.dim(0).set_bounds(0, npairs).set_stride(1);

  }
};






void buildAngularBasis(Expr k3, Expr npairs, Func pq, Func rij,
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

  /*
  Func abf4tm("abf4tm");
  Var tmp("tmp");
  abf4tm(pair, abfi, abfip, c, tmp) = zero;
  abf4tm(pair, abfi, 0, 0, 1) = Expr((double) 1.0);
  RDom rn(1, k3 + 1, 0, 4);
  Expr m = clamp(pq(rn.x) - 1, 0, 3 * k3 - 1);
  Expr prev0 = abf4tm(pair, abfi, m, 0, 1);
  Expr d = clamp(pq(rn.x + k3), 1, 3);
  Var selected("selected");
  Func uvw("uvw");
  uvw(pair, selected) = select(selected == 1, u, select(selected==2, v, select(selected==3, w, Expr((double) 0.0))));
  abf4tm(pair, abfi, rn.x, rn.y, 1) = abf4tm(pair, abfi, m, rn.y, 1) * uvw(pair, d) + select(d == rn.y, abf4tm(pair, abfi, m, 0, 1), zero);
  */

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
  //abf4(pair, abfi, c) = zero;

  /*
  tm(pair, abfi, abfip, c) = zero;
  tm(pair, abfi, 0, 0) = Expr((double) 1.0);
  RDom rn(1, k3 + 1, 0, 4);
  Expr m = clamp(pq(rn.x) - 1, 0, 3 * k3 - 1);
  Expr prev0 = tm(pair, abfi, m, 0);
  Expr d = clamp(pq(rn.x + k3), 1, 3);
  Var selected("selected");
  Func uvw("uvw");
  uvw(pair, selected) = select(selected == 1, u, select(selected==2, v, select(selected==3, w, Expr((double) 0.0))));
  tm(pair, abfi, rn.x, rn.y) = tm(pair, abfi, m, rn.y) * uvw(pair, d) + select(d == rn.y, tm(pair, abfi, m, 0), zero);
  abf4(pair, abfi, c) = zero;

  uvw.bound(selected, 1, 3);
  abf4(pair, abfi, 0) = tm(pair, abfi, abfi, 0);
  abf4(pair, abfi, 1) = tm(pair, abfi, abfi, 1) * dudx + tm(pair, abfi, abfi, 2) * dvdx + tm(pair, abfi, abfi, 3) * dwdx;
  abf4(pair, abfi, 2) = tm(pair, abfi, abfi, 1) * dudy + tm(pair, abfi, abfi, 2) * dvdy + tm(pair, abfi, abfi, 3) * dwdy;
  abf4(pair, abfi, 3) = tm(pair, abfi, abfi, 1) * dudz + tm(pair, abfi, abfi, 2) * dvdz + tm(pair, abfi, abfi, 3) * dwdz;  */


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
  //abf4(pair, abfi, c) = select(c == 0, tm(pair, abfi, abfi, 0),
  //                      tm(pair, abfi, abfi, 1) * jacobian(pair, c-1, 0) + tm(pair, abfi, abfi, 2) * jacobian(pair, c-1, 1) + tm(pair, abfi, abfi, 3) * jacobian(pair, c-1, 2));   
  abf4(pair, abfi, c) = select(c == 0, tm(pair, abfi, 0),
                        tm(pair, abfi, 1) * jacobian(pair, c-1, 0) + tm(pair, abfi, 2) * jacobian(pair, c-1, 1) + tm(pair, abfi, 3) * jacobian(pair, c-1, 2));   
  /*
  abf4tm(pair, abfi, 0, c, 0) = select(c == 0, abf4tm(pair, abfi, abfi, 0, 1),
                        abf4tm(pair, abfi, abfi, 1, 1) * jacobian(pair, c-1, 0) + abf4tm(pair, abfi, abfi, 2, 1) * jacobian(pair, c-1, 1) + abf4tm(pair, abfi, abfi, 3, 1) * jacobian(pair, c-1, 2));   
  */

  /*
  abf4tm(pair, abfi, 0, 0, 0) = abf4tm(pair, abfi, abfi, 0, 1);
  abf4tm(pair, abfi, 0, 1, 0) = abf4tm(pair, abfi, abfi, 1, 1) * jacobian(pair, 0, 0) +
      abf4tm(pair, abfi, abfi, 2, 1) + jacobian(pair, 0, 1) + abf4tm(pair, abfi, abfi, 3, 1) * jacobian(pair, 0, 2);
  abf4tm(pair, abfi, 0, 2, 0) = abf4tm(pair, abfi, abfi, 1, 1) * jacobian(pair, 1, 0) +
      abf4tm(pair, abfi, abfi, 2, 1) * jacobian(pair, 1, 1) + abf4tm(pair, abfi, abfi, 3, 1) * jacobian(pair, 1, 2);
  abf4tm(pair, abfi, 0, 3, 0) = abf4tm(pair, abfi, abfi, 1, 1) * jacobian(pair, 2, 0) +
      abf4tm(pair, abfi, abfi, 2, 1) + jacobian(pair, 2, 1) + abf4tm(pair, abfi, abfi, 3, 1) * jacobian(pair, 2, 2);
  */
                       
    
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
  abf4.reorder(c, abfi, pair);
  */
  tm.store_root().compute_root();
  abf4.store_root().compute_root();
  //abf4.store_root().compute_root();
}

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

void threeBodyDescDeriv(Func & dd3, Func sumU, Func U, Func atomtype, Func pn3, Func pc3,
        Func elemindex, Expr npairs, Expr q, Expr nelements, Var dim, Var nj, Var abf3, Expr nabf3, 
        Var rbf3, Expr nrbf3, Var kme, Expr me)
{
    Expr zero = Expr((double) 0.0);
    
    Var rbfTres("rbftres");
    dd3(dim, nj, abf3, rbfTres, kme) = zero;
    
    dd3.bound(dim, 0, 3);
    dd3.bound(nj, 0, npairs);
    dd3.bound(abf3, 0, nabf3);
    dd3.bound(rbfTres, 0, nrbf3);
    dd3.bound(kme, 0, me);
    
    RDom r(0, nabf3, 0, q, 0, nelements, 0, npairs);
    Expr n1 = pn3(r.x);
    Expr n2 = pn3(r.x + 1);
    r.where(n1 <= r.y); 
    r.where(r.y < n2);
    RVar rx = r.x;
    RVar ry = r.y;
    RVar rz = r.z;
    RVar rzz = r[3];
    
    Expr t1 = pc3(ry) * sumU(rz, ry, rbfTres);
    Expr i2 = atomtype(rzz) - 1;
    Expr k = elemindex(clamp(i2, 0, nelements - 1), rz);
    Expr f = select(rz == i2, 2 * t1, t1);
    
    dd3(dim, rzz, rx, rbfTres, clamp(k, 0, me - 1)) += f * U(rzz, ry, rbfTres, dim + 1);
}

void threeBodyDesc(Func & d3,
		   Func sumU, Func pn3, Func pc3,
		   Expr npairs, Expr nelements, Expr nrbf3, Expr nabf3, Expr k3,
		   Var abf3, Var rbf3, Var kme)
{
  Expr zero = Expr((double) 0.0);
  Expr me = nelements * (nelements + 1)/2;
  Expr mem = nelements * (nelements - 1)/2;
  // Var abf3("abf3");
  // Var rbf3("rbf3");
  // var kme("kme");

  d3(abf3, rbf3, kme) = zero;

  d3.bound(abf3, 0, nabf3);
  d3.bound(rbf3, 0, nrbf3);
  d3.bound(kme, 0, me);

  RDom r(0, nabf3, 0, k3, 0, nelements, 0, nelements);
  Expr n1 = pn3(r.x);
  Expr n2 = pn3(r.x + 1);
  r.where(n1 <= r.y);
  r.where(r.y < n2);
  RVar rx = r.x;
  RVar ry = r.y;
  RVar rz = r.z;
  RVar rzz = r[3];
  r.where(rzz <= rz);

  Expr t1 = pc3(r.y) * sumU(rz, r.y, rbf3);
  //  Expr ki = (nelements - rz) * ((nelements - rz) - 1)/2;
  //  Expr kij = rzz - rz - 1;
  Expr k = (2 * nelements - 3 - rz) * (rz/ 2) + rzz - 1; //mem  - ki + kij;
  Expr t2 = sumU(rzz, r.y, rbf3);
  d3(r.x, rbf3, clamp(k, 0, me -1)) += t1 * t2;
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
  Expr i3 = kp % N1;

  indexmap(k, c) = mux(c, {i1, i2, i3});
  indexmap.bound(k, 0, n1 * n2 * n3);
  indexmap.bound(c, 0, 3);
}


void fourbodyfij23(Func & fij,
		   Func ind23, Func ind32, Func coeff23, Func d2, Func dd3,
		   Expr npairs, Expr n23, Expr n32,
		   Var pairindex)
{

  Expr zero = Expr((double) 0.0);
  Func cf1("cf1");
  Var j("j");
  Var dim("dim");
  Rdom r1(0, n23);
  cf1(j) = zero;
  cf1(j) += d2(ind23(r1.x,1), ind23(r1.x,2)) * coeff23(r.x, j);

  Rdom r2(0, n32);
  fij(pairindex, dim) += cf1(r2.x) * dd3(dim, ind32(r2.x,0), ind32(r2.x,1), ind32(r2.x,2));

  Func cf2("c2");
  Var i("i");
  cf2(i) = zero;
  cf2(i) += d3(dim, ind32(r2.x,0), ind32(r2.x,1), ind32(r2.x,2)) * coeff23(i, r2.x);

  fij(pairindex, dim) += cf2(r1.x) * dd2(dim, ind23(r1.x, 1), ind23(r1.x, 2));
    


  
  

    
    
    
  
}

class poddescTwoBody : public Halide::Generator<poddescTwoBody> {
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
    Input<Buffer<int>> pn3{"pn3", 1};
    Input<Buffer<int>> pc3{"pc3", 1};
    Input<Buffer<int>> elemindex{"elemindex", 2};


    Input<int> nrbf3{"nrbf3", 1};
    Input<int> nelements{"nelements", 1};

  Input<int> nd23{"nd23", 1};
  Input<int> nd33{"nd33", 1};
  Input<int> nd34{"nd34", 1};
  Input<int> nabf3{"nabf3", 1};
  
    Output<Buffer<double>> sumU_o{"sumU_o", 3};
    Output<Buffer<double>> U_o{"U_o", 4};

    Output<Buffer<double>> d2_o{"d2_o", 2};
    Output<Buffer<double>> dd2_o{"dd2_o", 4};

  Output<Buffer<double>> d3_o{"d3_o", 3};
  Output<Buffer<double>> dd3_o{"dd3_o", 5};

    void generate() {
        rijs.dim(0).set_bounds(0, 3).set_stride(1);
        rijs.dim(1).set_bounds(0, npairs).set_stride(3);

        besselparams.dim(0).set_bounds(0, nbesselparams);
        Var bfi("basis function index");
        Var bfp("basis function param");
        Var np("pairindex");
        Var numOuts("numOuts");
        Var dim("dim");

        Func rbft("rbft");
        buildRBF(rbft, rijs, besselparams, rin, rcut-rin,
		 bdegree, adegree, nbesselparams, npairs, ns,
		 bfi, bfp, np, dim);

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

	    // rbf.compute_root(); // 19410.314 ms -- .004 ms 2%
        rbf.store_root().compute_root(); // 19353.843750 ms -- .003 ms 1%
        // Nothing? 36 seconds .004 ms 1%

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

	Func abf4("abf4");
	Func tm("tm");
	Var abfi("abfi");
	Var abfip("abfip");
	buildAngularBasis(k3, npairs, pq, rijs,
			  abf4, tm,
			  c, np,  abfi, abfip
			  );
	// abf4.compute_root();

	Func sumU("sumU"), U("U");
	Var copy1, copy2, copy3, copy4;
	radialAngularBasis(sumU, U, rbf, abf4,
			   tj, npairs, k3, nrbf3, nelements);
	sumU_o(copy1, copy2, copy3) = sumU(copy1, copy2, copy3);
	U_o(copy1, copy2, copy3, copy4)= U(copy1, copy2, copy3, copy4);
	sumU_o.dim(0).set_bounds(0, nelements).set_stride(1);
	sumU_o.dim(1).set_bounds(0, k3).set_stride(nelements);
	sumU_o.dim(2).set_bounds(0, nrbf3).set_stride(nelements * k3);
	U_o.dim(0).set_bounds(0, npairs).set_stride(1);
	U_o.dim(1).set_bounds(0, k3).set_stride(npairs);
	U_o.dim(2).set_bounds(0, nrbf3).set_stride(npairs * k3);
	U_o.dim(3).set_bounds(0, 4).set_stride(npairs * k3 * nrbf3);

    //U_o.compute_root();
    sumU_o.compute_root();

    //     abf4.dim(2).set_bounds(0, 4).set_stride(k3* npairs);
    // abf4.dim(1).set_bounds(0, k3).set_stride(npairs);
    // abf4.dim(0).set_bounds(0, npairs).set_stride(1);
    // sumU_o.dim(0).set_bounds(0, nelements).set_stride(1);
    // sumU_o.dim(1).set_bounds(0, k3).set_stride(nelements);
    // sumU_o.dim(2).set_bounds(0, nrbf3).set_stride(nelements * k3);
    // U_o.dim(0).set_bounds(0, npairs).set_stride(1);
    // U_o.dim(1).set_bounds(0, k3).set_stride(npairs);
    // U_o.dim(2).set_bounds(0, nrbf3).set_stride(npairs * k3);
    // U_o.dim(3).set_bounds(0, 4).set_stride(npairs * k3 * nrbf3);

    // // if nd23 > 0
    Func d2("d2"), dd2("dd2");
    twoBodyDescDeriv(d2, dd2, rbf, tj, npairs, nelements, nrbf2);
    // d2.compute_root();
    dd2.compute_root();



    d2_o(copy1, copy2) = d2(copy1, copy2);
    dd2_o(copy1, copy2, copy3, copy4) = dd2(copy1, copy2, copy3, copy4);

    d2_o.dim(0).set_bounds(0, nelements).set_stride(nrbf2);
    d2_o.dim(1).set_bounds(0, nrbf2).set_stride(1);
    

    dd2_o.dim(0).set_bounds(0, nelements).set_stride(3 * npairs * nrbf2);
    dd2_o.dim(1).set_bounds(0, nrbf2).set_stride(3 * npairs);
    dd2_o.dim(2).set_bounds(0, npairs).set_stride(3);
    dd2_o.dim(3).set_bounds(0, 3).set_stride(1);


    Func d3("d3");
    Var abfThree("abfThree");
    Var rbfThree("rbfThree");
    Var kme("kme");
    threeBodyDesc(d3, sumU, pn3, pc3,
		  npairs, nelements, nrbf3, nabf3, k3,
		  abfThree, rbfThree, kme);

    Expr me = nelements * (nelements + 1)/2;
    d3_o(copy1, copy2, copy3) = d3(copy1, copy2, copy3);
    d3_o.dim(0).set_bounds(0, nabf3).set_stride(1);
    d3_o.dim(1).set_bounds(0, nrbf3).set_stride(nabf3);
    d3_o.dim(2).set_bounds(0, me).set_stride(nabf3 * nrbf3);
    
    Func dd3("dd3");
    Var nj("nj");    
    threeBodyDescDeriv(dd3, sumU, U, tj, pn3, pc3,
        elemindex, npairs, k3, nelements, dim, nj, abfThree, nabf3, 
        rbfThree, nrbf3, kme, me);
    dd3_o(dim, nj, copy1, copy2, copy3) = dd3(dim, nj, copy1, copy2, copy3);
    dd3_o.dim(0).set_bounds(0, 3).set_stride(1);
    dd3_o.dim(1).set_bounds(0, npairs).set_stride(3);
    dd3_o.dim(2).set_bounds(0, nabf3).set_stride(3 * npairs);
    dd3_o.dim(3).set_bounds(0, nrbf3).set_stride(3 * npairs * nabf3);
    dd3_o.dim(4).set_bounds(0, me).set_stride(3 * npairs * nabf3 * nrbf3);
    
    }
};

HALIDE_REGISTER_GENERATOR(poddescTwoBody, poddescTwoBody);
HALIDE_REGISTER_GENERATOR(poddescRBF, poddescRBF);
HALIDE_REGISTER_GENERATOR(poddescRadialAngularBasis, poddescRadialAngularBasis);
HALIDE_REGISTER_GENERATOR(poddescTwoBodyDescDeriv, poddescTwoBodyDescDeriv);
HALIDE_REGISTER_GENERATOR(poddescTallyTwoBodyLocalForce, poddescTallyTwoBodyLocalForce);
HALIDE_REGISTER_GENERATOR(poddescFourMult, poddescFourMult);
HALIDE_REGISTER_GENERATOR(poddescAngularBasis, poddescAngularBasis);
