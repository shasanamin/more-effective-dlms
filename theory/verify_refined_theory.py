#!/usr/bin/env python3
"""Deterministic checks accompanying the paper's finite-step analysis.

No model weights, network access, or accelerator are required. This is NOT
proof-assistant certification. The mathematical proofs are in the paper.
Run: python theory/verify_refined_theory.py
Requires Python >=3.10 and NumPy; writes JSON to --output when requested.
"""
from __future__ import annotations
import argparse
import itertools as it
import json
import math
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import numpy as np
from numpy.polynomial import Polynomial as Poly

TOL = 3e-10
RESULT = {"seed": 20260922, "checks": {}, "max_absolute_residual": 0.0,
          "formal_proof_assistant": False}

def check(name, lhs, rhs, tol=TOL):
    error = abs(float(lhs) - float(rhs))
    if not np.isfinite(error) or error > tol:
        raise AssertionError(f"{name}: {lhs} versus {rhs}; residual={error}")
    RESULT["checks"][name] = RESULT["checks"].get(name, 0) + 1
    RESULT["max_absolute_residual"] = max(RESULT["max_absolute_residual"], error)

def require(name, condition):
    if not bool(condition):
        raise AssertionError(name)
    RESULT["checks"][name] = RESULT["checks"].get(name, 0) + 1

def basis(k, d):
    """Integer coefficients of a**k * (1-a)**d."""
    return {k+j: (-1)**j * math.comb(d, j) for j in range(d+1)}

def symbolic_entropy_identities():
    # Coefficient keys are (formal entropy variable H(X_A), power of a).
    # All arithmetic is integer/Fraction; no entropy values are substituted.
    for L in range(1, 9):
        full = (1 << L)-1
        F, h, D = defaultdict(int), defaultdict(int), defaultdict(int)
        for A in range(full+1):
            for p, coef in basis(A.bit_count(), L-A.bit_count()).items():
                if A: F[A, p] += coef
        for i in range(L):
            for A in range(full+1):
                if A & (1 << i): continue
                for p, coef in basis(A.bit_count(), L-1-A.bit_count()).items():
                    h[A | (1 << i), p] += coef
                    if A: h[A, p] -= coef
                for j in range(L):
                    if j == i or A & (1 << j): continue
                    for p, coef in basis(A.bit_count(), L-2-A.bit_count()).items():
                        # I(X_i;X_j | X_A)
                        for B, sign in ((A | (1 << i), 1), (A | (1 << j), 1),
                                        (A, -1), (A | (1 << i) | (1 << j), -1)):
                            if B: D[B, p] += sign*coef
        derivative = lambda f: { (A,p-1): p*c for (A,p),c in f.items() if p and c }
        clean = lambda f: {key: value for key,value in f.items() if value}
        require("symbolic_F_prime_equals_h", clean(derivative(F)) == clean(h))
        require("symbolic_minus_h_prime_equals_CMI_profile",
                clean({k:-v for k,v in derivative(h).items()}) == clean(D))
        area = defaultdict(Fraction)
        for (A,p),coef in h.items(): area[A] += Fraction(coef,p+1)
        require("exact_rational_entropy_area", clean(area) == {full: Fraction(1)})

class FiniteAudit:
    """Independent enumeration; does not import the original verifier."""
    def __init__(self, L, V, delta, seed, sparse=False):
        self.L, self.V = L, V
        self.delta = np.array(delta, dtype=float)
        self.N = len(delta)
        self.a = np.r_[0.0, np.cumsum(delta)]
        self.states = list(it.product(range(V), repeat=L))
        self.rng = np.random.default_rng(seed)
        self.P = self.rng.gamma(1.7, size=len(self.states))
        if sparse:
            self.P[self.rng.random(len(self.P)) < .35] = 0
            self.P[0] += .2
        self.P /= self.P.sum()
        self.ent = {}
        for A in range(1 << L):
            marg = defaultdict(float)
            for x, p in zip(self.states, self.P):
                marg[tuple(x[j] for j in range(L) if A & (1 << j))] += p
            self.ent[A] = -sum(p*math.log(p) for p in marg.values() if p)
        self.H = self.ent[(1 << L)-1]
        self.h = Poly([0.0])
        for i in range(L):
            for A in range(1 << L):
                if A & (1 << i): continue
                k = A.bit_count()
                b = Poly([0,1])**k * Poly([1,-1])**(L-1-k)
                self.h += b * (self.ent[A | (1 << i)] - self.ent[A])
        self.logits_cache = {}

    def logits(self, k, i, A, x):
        context = tuple(x[j] if A & (1 << j) else -1 for j in range(self.L))
        key = (k, i, context)
        if key not in self.logits_cache:
            self.logits_cache[key] = self.rng.normal(size=self.V)
        return self.logits_cache[key]

    def q(self, k, i, A, x, lam):
        r = self.logits(k,i,A,x)
        q = np.exp(lam*r - np.max(lam*r)); q /= q.sum()
        return q, r

    def objective(self, lam):
        J = J1 = J2 = 0.
        for k,d in enumerate(self.delta):
            a = self.a[k]
            for i in range(self.L):
                for A in range(1 << self.L):
                    if A & (1 << i): continue
                    m = A.bit_count()
                    w = d*a**m*(1-a)**(self.L-1-m)
                    if not w: continue
                    for x,p in zip(self.states,self.P):
                        q,r = self.q(k,i,A,x,lam)
                        mean = float(q@r)
                        J -= w*p*math.log(q[x[i]])
                        J1 += w*p*(mean-r[x[i]])
                        J2 += w*p*float(q@((r-mean)**2))
        return J, J1, J2

    def run(self, lam):
        # Force objective cache population independent of reveal enumeration.
        J,J1,J2 = self.objective(lam)
        out = np.zeros(len(self.states)); aug = Bdirect = 0.
        paths = []
        for K in it.product(range(self.N), repeat=self.L):
            nu = math.prod(self.delta[k] for k in K)
            Q = []
            for x,p in zip(self.states,self.P):
                qx = 1.
                for k in range(self.N):
                    A = sum(1 << j for j in range(self.L) if K[j] < k)
                    S = [i for i in range(self.L) if K[i] == k]
                    for i in S: qx *= self.q(k,i,A,x,lam)[0][x[i]]
                Q.append(qx)
                if p: aug += nu*p*math.log(p/qx)
            Q = np.array(Q)
            check("conditional_sampler_normalization", Q.sum(), 1.)
            out += nu*Q; paths.append((nu,Q))
            for k in range(self.N):
                A = sum(1 << j for j in range(self.L) if K[j] < k)
                S = [i for i in range(self.L) if K[i] == k]
                joint = A | sum(1 << i for i in S)
                tc = sum(self.ent[A | (1 << i)]-self.ent[A] for i in S)
                tc -= self.ent[joint]-self.ent[A]
                Bdirect += nu*tc
        D = sum(p*math.log(p/q) for p,q in zip(self.P,out) if p)
        R = sum(nu*sum(p*math.log(o/q) for p,o,q in zip(self.P,out,Q) if p)
                for nu,Q in paths)
        Hgrid = sum(d*self.h(a) for d,a in zip(self.delta,self.a[:-1]))
        B = Hgrid-self.H; C = J-Hgrid
        check("augmented_KL_equals_J_minus_entropy", aug, J-self.H)
        check("conditional_TC_equals_entropy_quadrature_gap", Bdirect, B)
        check("output_KL_equals_B_plus_C_minus_R", D, B+C-R)
        require("all_decomposition_terms_nonnegative", min(B,C,R,D) >= -TOL)
        require("Pinsker_bound", .5*np.abs(self.P-out).sum() <= math.sqrt(max(0,(B+C)/2))+TOL)
        integral = 0.
        for lo,hi in zip(self.a[:-1],self.a[1:]):
            antiderivative = ((Poly([hi,-1]))*(-self.h.deriv())).integ()
            integral += antiderivative(hi)-antiderivative(lo)
        check("exact_dependence_integral", B, integral)
        require("mesh_dependence_bound", B <= max(self.delta)*(self.h(0)-self.h(1))+TOL)
        require("h_nondecreasing_information", np.max(self.h.deriv()(np.linspace(0,1,51))) <= TOL)
        check("numeric_entropy_area", self.h.integ()(1)-self.h.integ()(0), self.H)
        refined = sum((hi-lo)/2*(self.h(lo)+self.h((hi+lo)/2))
                      for lo,hi in zip(self.a[:-1],self.a[1:]))-self.H
        require("nested_refinement_reduces_B", refined <= B+TOL)
        step=2e-4
        jp=self.objective(lam+step)[0]; jm=self.objective(lam-step)[0]
        check("temperature_gradient", (jp-jm)/(2*step), J1, 8e-8)
        check("temperature_hessian", (jp+jm-2*J)/step**2, J2, 1e-6)
        require("temperature_convexity", J2 >= 0.)
        h2=self.h.deriv(2)
        candidates=[0.,1.]
        for root in self.h.deriv(3).roots():
            if abs(root.imag)<1e-9 and 0 < root.real < 1: candidates.append(float(root.real))
        bound=max(abs(float(h2(x))) for x in candidates)
        for N in (1,2,3,8,32):
            BN=sum(self.h(k/N)/N for k in range(N))-self.H
            lead=(self.h(0)-self.h(1))/(2*N)
            require("finite_uniform_grid_remainder", abs(BN-lead) <= bound/(12*N*N)+TOL)


def bkl(p,q):
    return p*math.log(p/q)+(1-p)*math.log((1-p)/(1-q))

def collision_family():
    rng=np.random.default_rng(8675309)
    finite=boundary=0
    for N in range(2,8):
        for _ in range(30):
            delta=rng.dirichlet(np.ones(N)); c=float(delta@delta)
            s=float(rng.uniform(.51,.97)); ell=math.log(s/(1-s))
            lam=1.2; u=1/(1+math.exp(-lam*ell)); t=c/2+(1-c)*u
            # Independent enumeration of the complete 2-bit sampler.
            out=np.zeros(4)
            for k1,k2 in it.product(range(N),repeat=2):
                out += delta[k1]*delta[k2]*(np.full(4,.25) if k1==k2
                    else np.array([u,1-u,1-u,u])/2)
            check("collision_family_output", out[0]+out[3], t)
            if c < 2*(1-s):
                ustar=(s-c/2)/(1-c)
                lamstar=math.log(ustar/(1-ustar))/ell
                require("finite_repair_requires_sharpening", lamstar>1)
                check("exact_finite_temperature_repair", c/2+(1-c)*ustar,s)
                require("repair_worsens_posterior_certificate", (1-c)*bkl(s,ustar)>0)
                finite+=1
            else:
                for lam in (.1,1.,2.,4.):
                    u=1/(1+math.exp(-lam*ell)); t=c/2+(1-c)*u
                    derivative=(t-s)/(t*(1-t))*(1-c)*ell*u*(1-u)
                    require("boundary_regime_strict_decrease", derivative<0)
                boundary+=1
    RESULT["collision_cases"]={"finite_exact_repair":finite,"limit_optimum":boundary}
    s=.7;c=.5; ustar=(s-c/2)/(1-c)
    RESULT["illustrative_lambda_star"]=math.log(ustar/(1-ustar))/math.log(s/(1-s))
    RESULT["illustrative_tau_star"]=1/RESULT["illustrative_lambda_star"]


def bayes_two_bit_output(target, lam, c):
    """Enumerate general (possibly asymmetric) two-bit Bayes sampling."""
    target=np.asarray(target, dtype=float).reshape(2,2)
    def temper(probabilities):
        probabilities=np.asarray(probabilities, dtype=float)**lam
        return probabilities/probabilities.sum()
    marginal0=temper(target.sum(axis=1))
    marginal1=temper(target.sum(axis=0))
    simultaneous=np.outer(marginal0,marginal1)
    first0=np.zeros((2,2));first1=np.zeros((2,2))
    for x in range(2):
        first0[x,:]=marginal0[x]*temper(target[x,:])
        first1[:,x]=marginal1[x]*temper(target[:,x])
    return c*simultaneous+(1-c)*(first0+first1)/2


def collision_boundary_and_robustness():
    # Randomized tests above almost surely miss the equality boundary.
    s=.75;c=.5
    errors=[]
    for lam in (.25,1.,2.,4.,8.,16.):
        u=1/(1+math.exp(-lam*math.log(s/(1-s))))
        t=c/2+(1-c)*u
        require("critical_collision_boundary_below_target",t<s)
        errors.append(bkl(s,t))
    require("critical_collision_boundary_strict_improvement",
            all(x>y for x,y in zip(errors,errors[1:])))
    check("critical_collision_boundary_zero_limit",1-c/2,s)
    for lam in (.25,1.,2.,8.):
        target=np.array([[.35,.15],[.15,.35]])
        out=bayes_two_bit_output(target,lam,1.)
        check("single_round_no_temperature_effect",np.abs(out-.25).max(),0.)

    # Check the stated open-neighborhood claim without assuming bit symmetry.
    rng=np.random.default_rng(20260923)
    target0=np.array([[.35,.15],[.15,.35]])
    lamstar=math.log(9)/math.log(7/3)
    for _ in range(100):
        target=target0+rng.uniform(-.01,.01,size=(2,2))
        target/=target.sum()
        d=float(rng.uniform(.46,.54));c=d*d+(1-d)*(1-d)
        reference=bayes_two_bit_output(target,1.,c)
        sharpened=bayes_two_bit_output(target,lamstar,c)
        kl=lambda output: float(np.sum(target*np.log(target/output)))
        check("perturbed_Bayes_output_normalization",sharpened.sum(),1.)
        require("Bayes_sharpening_survives_asymmetric_perturbations",
                kl(sharpened)<kl(reference))

    # Equal-grid collision probability goes to zero; verify first-order term.
    for s in (.55,.7,.9):
        ell=math.log(s/(1-s))
        coefficient=(s-.5)/(s*(1-s)*ell)
        errors=[]
        for c in (1e-3,5e-4,2.5e-4):
            ustar=(s-c/2)/(1-c)
            lamstar=math.log(ustar/(1-ustar))/ell
            errors.append(abs(lamstar-1-coefficient*c))
        require("collision_expansion_quadratic_remainder",
                .2<errors[1]/errors[0]<.3 and .2<errors[2]/errors[1]<.3)


def clipped_schedule_checks():
    # Verify that all-mask initialization gives visible probability 1-t,
    # rather than the clipped training schedule alpha(t).
    eta=1e-3;epsilon=1e-5
    for M in (1,2,8,32,128,1024):
        times=np.linspace(1.,epsilon,M+1)
        alpha=1-(1-eta)*times
        survival=1.
        for k in range(M):
            new_survival=survival*(1-alpha[k+1])/(1-alpha[k])
            check("clipped_schedule_normalized_reveal_mass",
                  survival-new_survival,(1-epsilon)/M)
            survival=new_survival
        check("clipped_schedule_cleanup_mass",survival,epsilon)


def integer_partitions(n, minimum=1):
    if n==0: yield (); return
    for first in range(minimum,n+1):
        for rest in integer_partitions(n-first,first): yield (first,)+rest

def ged(counts,n):
    v=sum(counts)
    if v<2:return 0.
    return 10*(v*math.log(v)-sum(k*math.log(k) for k in counts if k))/(n*math.log(n))

def ged_checks():
    for n in range(2,13):
        for v in range(n+1):
            for counts in integer_partitions(v):
                value=ged(counts,n)
                require("GED_bounds", -TOL<=value<=10+TOL)
                require("GED_ceiling_characterization", (abs(value-10)<TOL)==(counts==(1,)*n))
                for i,k in enumerate(counts):
                    reduced=list(counts);reduced[i]-=1
                    require("invalidating_fixed_label_sample_cannot_increase_GED",ged(reduced,n)<=value+TOL)
                    for j in range(i+1,len(counts)):
                        merged=[x for t,x in enumerate(counts) if t not in (i,j)]+[k+counts[j]]
                        require("merging_modes_cannot_increase_GED",ged(merged,n)<=value+TOL)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    symbolic_entropy_identities()
    rng=np.random.default_rng(RESULT["seed"])
    for L,V in ((1,2),(2,2),(3,2),(4,2),(3,3)):
        for N in (1,2,3):
            for sparse in (False,True):
                delta=rng.dirichlet(np.ones(N))
                FiniteAudit(L,V,delta,int(rng.integers(2**31)),sparse).run(1.13)
    collision_family();collision_boundary_and_robustness();clipped_schedule_checks();ged_checks()
    RESULT['status']='PASS'
    RESULT['total_checks']=sum(RESULT['checks'].values())
    text=json.dumps(RESULT,indent=2,sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text+'\n')

if __name__=='__main__':main()
