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

  drbf.compute_root();

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

  rbfall.compute_root();
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

}

void twoBodyDescDeriv(Func & d2, Func & dd2, Func rbf, Func rbfx, Func rbfy, Func rbfz, Func tj, Expr N, Expr Ne, Expr nrbf2)
{
    Expr zero = Expr((double) 0.0);

    Var ne("ne"), m("m"), n("n"), dim("dim");
    d2(ne, m) = zero;
    dd2(ne, m, n, dim) = zero;

    RDom r(0, N, 0, nrbf2);

    d2(clamp(tj(r.x)-1, 0, Ne - 1), r.y) += rbf(r.x, r.y);
    dd2(clamp(tj(r.x)-1, 0, Ne - 1), r.y, r.x, 0) += rbfx(r.x, r.y);
    dd2(clamp(tj(r.x)-1, 0, Ne - 1), r.y, r.x, 1) += rbfy(r.x, r.y);
    dd2(clamp(tj(r.x)-1, 0, Ne - 1), r.y, r.x, 2) += rbfz(r.x, r.y);

    d2.bound(ne, 0, Ne);
    d2.bound(m, 0, nrbf2);

    dd2.bound(ne, 0, Ne);
    dd2.bound(m, 0, nrbf2);
    dd2.bound(n, 0, N);
    dd2.bound(dim, 0, 3);
    
}

void tallyTwoBodyLocalForce(Func & fij, Func & e, Func coeff2, Func rbf, Func rbfx, Func rbfy, Func rbfz, Func tj, Expr nbf, Expr N)
{
    Expr zero = Expr((double) 0.0);

    Var n("n"), m("m"), dim("dim"), empty("empty");
    e() = zero;
    fij(n, dim) = zero;

    RDom r(0, N, 0, nbf);

    Expr c = coeff2(clamp(tj(r.x), 1, N - 1) - 1, r.y);
    e() += c * rbf(r.x, r.y);
    fij(r.x, 0) += c * rbfx(r.x, r.y);
    fij(r.x, 1) += c * rbfy(r.x, r.y);
    fij(r.x, 2) += c * rbfz(r.x, r.y);

    fij.bound(n, 0, N);
    fij.bound(dim, 0, 3);
}

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
}

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

    // rijs.dim(1).set_bounds(0, npairs).set_stride(3);
    // rijs.dim(0).set_bounds(0, 3).set_stride(1);
    // besselparams.dim(0).set_bounds(0, nbesselparams);
    // Phi.dim(0).set_bounds(0, ns).set_stride(1);
    // Phi.dim(1).set_bounds(0, ns).set_stride(ns);
    // pq.dim(0).set_bounds(0, 3* k3).set_stride(1);

    
    // Var bfi("basis function index");
    // Var bfp("basis function param");
    // Var np("pairindex");
    // Var numOuts("numOuts");
    // Var dim("dim");

    // Func rbft_temp("rbf_f");
    // buildRBF(rbft_temp,
    //          rijs, besselparams, rin, rcut-rin,
    //          bdegree, adegree, nbesselparams, npairs, ns,
    //          bfi, bfp, np, dim);

    // Var i("i");
    // Var j("j");
    // Var k("k");
    // Var c("c");
    // Func prod("prod");
    // prod(c, k, i, j) = Phi(k, i) * rbf4_in(j, k, c);
    // prod.bound(c, 0, 4);
    // prod.bound(k, 0, nrbfmax);
    // prod.bound(j, 0, npairs);
    // prod.bound(i, 0, npairs);
    // Func rbf4("rbf4");
    // rbf4(j, i, c) = Expr((double) 0.0);
    // RDom r(0, ns);
    // rbf4(j, i, c) += prod(c, r, i, j);
    // rbf4.compute_root();


    // Func abf4("abf4");
    // Func tm("tm");
    // Var abfi("abfi");
    // Var abfip("abfip");
    // buildAngularBasis(k3, npairs, pq, rij,
    // 		      abf4, tm,
    // 		      c, np,  abfi, abfip
    // 		      );
    // abf4.compute_root();

    // Func sumU("sumU"), U("U");
    // radialAngularBasis(sumU, U, rbf4, abf4,
    // 		       tj, npairs, k3, nrbf3, nelements);
    // //     abf4.dim(2).set_bounds(0, 4).set_stride(k3* npairs);
    // // abf4.dim(1).set_bounds(0, k3).set_stride(npairs);
    // // abf4.dim(0).set_bounds(0, npairs).set_stride(1);

    // Var ne("ne"), m("m");
    // sumU_o(ne, k, m) = sumU(m, k, ne);
    // U_o(n, k, m, c) = U(m, k, n, c);

    // sumU_o.dim(0).set_bounds(0, nelements).set_stride(1);
    // sumU_o.dim(1).set_bounds(0, k3).set_stride(nelements);
    // sumU_o.dim(2).set_bounds(0, nrbf3).set_stride(nelements * k3);
    // U_o.dim(0).set_bounds(0, npairs).set_stride(1);
    // U_o.dim(1).set_bounds(0, k3).set_stride(npairs);
    // U_o.dim(2).set_bounds(0, nrbf3).set_stride(npairs * k3);
    // U_o.dim(3).set_bounds(0, 4).set_stride(npairs * k3 * nrbf3);
    // // if nd23 > 0
    // Func d2("d2"), dd2("dd2");
    // twoBodyDescDeriv(d2, dd2, rbf4, tj, npairs, nelements, nrbf2);
    // d2.compute_root();
    // dd2.compute_root();

    // d2_o(ne, m) = d2(ne, m);
    // dd2_o(ne, m, n, dim) = dd2(ne, m, n, dim);

    // d2_o.dim(0).set_bounds(0, nelements).set_stride(nrbf2);
    // d2_o.dim(1).set_bounds(0, nrbf2).set_stride(1);

    // dd2_o.dim(0).set_bounds(0, nelements).set_stride(3 * npairs * nrbf2);
    // dd2_o.dim(1).set_bounds(0, nrbf2).set_stride(3 * npairs);
    // dd2_o.dim(2).set_bounds(0, npairs).set_stride(3);
    // dd2_o.dim(3).set_bounds(0, 3).set_stride(1);

    
  }
};

HALIDE_REGISTER_GENERATOR(poddescRBF, poddescRBF);
HALIDE_REGISTER_GENERATOR(poddescRadialAngularBasis, poddescRadialAngularBasis);
HALIDE_REGISTER_GENERATOR(poddescTwoBodyDescDeriv, poddescTwoBodyDescDeriv);
HALIDE_REGISTER_GENERATOR(poddescTallyTwoBodyLocalForce, poddescTallyTwoBodyLocalForce);
HALIDE_REGISTER_GENERATOR(poddescFourMult, poddescFourMult);
HALIDE_REGISTER_GENERATOR(poddescAngularBasis, poddescAngularBasis);
