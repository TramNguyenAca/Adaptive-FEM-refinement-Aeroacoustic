import numpy as np
from ngsolve_forward import *

import regpy.stoprules as rules
from regpy.solvers import RegularizationSetting
from regpy.solvers.linear.landweber_short import Landweber
#from regpy.hilbert import L2, Sobolev
from regpy.hilbert import L2 as regL2
from regpy.vecsps.ngsolve import NgsSpace
import time

from copy import deepcopy

class bilevel:
    def __init__(self, maxh = 5e-2, scatterers = 3, order = 2, omega = 50, domain_is_complex = False):

        self.maxh = maxh
        self.scatterers = scatterers
        self.order = order
        self.omega = omega

        maker = domain_maker(maxh = self.maxh, scatterers = self.scatterers, \
                     sob = True, order = self.order, PML = False, omega = self.omega, \
                     domain_is_complex = domain_is_complex)

        self.maker = maker
        self.mesh = maker.mesh
        self.fes = maker.fes
        self.solver = maker.solver
        self.domain_is_complex = domain_is_complex
        
        self.update(CF(0), refine = False)

    def update(self, fk, refine = True):

        with TaskManager():

            if refine:
                mmesh = deepcopy(self.mesh)
            
                mats = np.array([el.mat for el in mmesh.Elements()])
                #notmid = mats != "mid"

                elvol = Integrate(CF(1), mmesh, element_wise=True).NumPy()[:] #Integrate(CF(1), mmesh, VOL, element_wise=True).NumPy()[:]
                #maxerr = max(elvol[notmid])
                maxerr = max(elvol)

                #print(elvol)
                #print(maxerr)

                # mark for refinement (vectorized alternative)
                mmesh.ngmesh.Elements2D().NumPy()["refine"] = \
                    elvol > 0.33*maxerr
                    #np.logical_and(elvol > 0.5*maxerr, notmid)
                mmesh.Refine()
                
                mfes = L2(mmesh, order = self.order, complex = self.domain_is_complex, autoupdate = False, definedon = "inner")
                mfes = Compress(mfes)
                mfk = GridFunction(mfes)
                mfk.Set(fk)
            
                self.mesh = mmesh
                self.fes = mfes
                fk = mfk
        
            mats = np.array([el.mat for el in self.mesh.Elements()])
            #notmid = mats != "mid"
            #notmid = np.logical_or(notmid,np.logical_not(refine))

            elvol = Integrate(CF(1), self.mesh, element_wise=True).NumPy()[:]
            if np.any(np.isnan(elvol)):
                raise Exception("NaN elements...")
            #maxerr = np.nanmax(elvol[notmid])
            maxerr = max(elvol)
            self.h = np.sqrt(2*maxerr)
            
            print("Current maxh: " + str(self.h) + "...")

            fk_util = GridFunction(self.fes, autoupdate = False, nested = False)
            fk_util.Set(fk)
            
            self.cofes = L2(self.mesh, order = self.order, complex = True, autoupdate = False, definedon = "outer")
            self.cofes = Compress(self.cofes)
            
            self.allfes = H1(self.mesh, order = self.order, complex = True, autoupdate = False)
            self.allfes = Compress(self.allfes)
                
            # Trial-test pairs for the feses.

            u, v = self.fes.TnT()
            U, V = self.cofes.TnT()
            UU, VV = self.allfes.TnT()

            self.domain = NgsSpace(self.fes)
            self.codomain = NgsSpace(self.cofes)

            omega = self.omega
            self.PDE = BilinearForm(self.allfes)
            self.PDE += grad(UU)*grad(VV)*dx - omega**2*UU*VV*dx
            #a += -omega*1j*U*V * ds("pmlregion")
            self.PDE += -omega*1j*UU*VV * ds("outer")
            self.PDE.Assemble()
            self.iPDE = self.PDE.mat.Inverse(freedofs = self.allfes.FreeDofs(), inverse=self.solver)
        
            self.APDE = BilinearForm(self.allfes)
            self.APDE += grad(UU)*grad(VV)*dx - omega**2*UU*VV*dx
            #a += -omega*1j*U*V * ds("pmlregion")
            self.APDE += omega*1j*UU*VV * ds("outer")
            self.APDE.Assemble()
            self.iAPDE = self.APDE.mat.Inverse(freedofs = self.allfes.FreeDofs(), inverse=self.solver)

        def eval_PDE(f):
            with TaskManager():
                VV = self.allfes.TestFunction()

                G = GridFunction(self.allfes)
                G.vec.data = self.iPDE * LinearForm(f * VV * dx).Assemble().vec
                
                g = GridFunction(self.cofes)
                g.Set(G)
            return g

        def adjoint_PDE(g):
            with TaskManager():
                VV = self.allfes.TestFunction()
                
                F = GridFunction(self.allfes)
                F.vec.data = self.iAPDE * LinearForm(g * VV * dx).Assemble().vec

                f = GridFunction(self.fes)
                if self.domain_is_complex:
                    f.Set(F)
                else:
                    f.Set(np.real(F))
            return f
        
        self.eval_PDE = eval_PDE
        self.adjoint_PDE = adjoint_PDE

        class PDEs(NGSolveOperator):
            def __init__(self, fes, cofes, allfes, domain, codomain, solvePDE, solveAPDE):
                self.fes = fes
                self.cofes = cofes
                self.allfes = allfes
                self.solvePDE = solvePDE
                self.solveAPDE = solveAPDE
                super().__init__(domain=domain,codomain=codomain,linear=True)
            
            def _eval(self, f):
                F = self.domain.to_ngs(f)

                #g0 = GridFunction(self.allfes)
                #g0.vec.data = self.iPDE * LinearForm(F * VV * dx).Assemble().vec

                #g = GridFunction(self.cofes)
                #g.Set(g0)
                g = self.solvePDE(F)
                return self.codomain.from_ngs(g)

            def _adjoint(self, g):
                G = self.codomain.to_ngs(g)

                #f0 = GridFunction(self.allfes)
                #f0.vec.data = self.iAPDE * LinearForm(G * VV * dx).Assemble().vec
                #f = GridFunction(self.fes)
                #f.Set(f0)
                f = self.solveAPDE(G)
                return self.domain.from_ngs(f)
            
        #self.rPDE = PDEs(fes = self.fes, cofes = self.cofes, allfes = self.allfes, domain = self.domain, codomain = self.codomain, solvePDE = self.eval_PDE, solveAPDE = self.adjoint_PDE)

        #X = regL2#NegativeSobolev
        #Y = regL2 #Sobolev #

        #self.setting = RegularizationSetting(op=self.rPDE, penalty=X, data_fid=Y)
        #start_normtime = time.time()
        
        self.norm = 0.075 #1/10#
        #self.norm = self.setting.op_norm()
        print("Skipping norm estimation, warning...")
        #self.norm = self.setting.op_norm()
        #print("Norm " + str(self.norm) + "estimated in", time.time() - start_normtime,"seconds...")
        return fk, fk_util

    
