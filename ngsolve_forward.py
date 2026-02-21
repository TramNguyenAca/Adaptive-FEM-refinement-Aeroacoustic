from ngsolve import *
from ngsolve.webgui import Draw
from netgen.occ import *
from netgen.geom2d import SplineGeometry
import numpy as np
import scipy.sparse as sparse

from regpy.functionals import Functional
from regpy.operators import Operator
from regpy.operators.ngsolve import NGSolveOperator
from regpy.operators import PtwMultiplication
#from regpy.discrs.ngsolve import NgsSpace, Matrix
#from regpy.hilbert import HilbertSpace, Sobolev
from regpy import util
from regpy.vecsps.ngsolve import NgsSpace
from regpy.solvers import RegularizationSetting
from regpy.hilbert import L2 as regpyL2

class domain_maker():
    def __init__(self, PML = False, maxh = 0.05, omega = 50, scatterers = 2, sob = False, order = 2, solver = '', domain_is_complex = False):
        
        self.domain_is_complex = domain_is_complex
        
        outer_radius = 1
        self.outer_radius = outer_radius
        outer = MoveTo(-1, -1).Rectangle(2, 2).Face() # Circle((0.0, 0.0), outer_radius).Face() #
        outer.edges.name = 'outer'
        
        inner_radius = 0.50
        self.inner_radius = inner_radius
        inner = MoveTo(-1/2, -1/2).Rectangle(1, 1).Face() #Circle((0.0, 0.0), inner_radius).Face() #
        inner.faces.name = 'inner'
        inner.edges.name = 'inner'
        #outer = outer - inner

        mid_radius = 0.55
        self.mid_radius = mid_radius
        mid = MoveTo(-0.55, -0.55).Rectangle(1.1, 1.1).Face()  #Circle((0.0, 0.0), mid_radius).Face() #
        
        pmlregion = Circle((0.0, 0.0), 1.2).Face()
        pmlregion.faces.name = 'pmlregion'
        pmlregion = pmlregion - outer
        
        outer = outer - mid
        mid = mid - inner
        mid.faces.name = 'mid'
        mid.edges.name = 'mid'

        scatterer = []
        for i in range(scatterers):
            if i==0:
                scatter = MoveTo(-0.8, -0.7).Rectangle(0.1, 0.4).Face()
                scatter.edges.name = 'scat1'
            if i==1:
                scatter = MoveTo(0.6, -0.2).Rectangle(0.1, 0.5).Face()
                scatter.edges.name = 'scat2'
            if i==2:
                scatter = MoveTo(-0.3, -0.8).Rectangle(0.5, 0.2).Face()
                scatter.edges.name = 'scat3'
            scatterer.append(scatter)

        for scatter in scatterer:
            outer = outer - scatter
        outer.faces.name = 'outer'

        bigouter = pmlregion + outer
        bigouter.faces.name = 'bigouter'
        #bigouter.edges.name = 'bigouter'

        if PML:
            geo = Glue([inner, mid, outer, pmlregion])
        else:
            geo = Glue([inner, mid, outer])
        
        mesh = Mesh(OCCGeometry(geo, dim=2).GenerateMesh(maxh=maxh))
        if PML:
            mesh.SetPML(pml.Radial(rad=1,alpha=1j,origin=(0,0)), "pmlregion")

        #designmesh = Mesh(OCCGeometry(geo, dim=2).GenerateMesh(maxh=maxh/3))

        self.fes = L2(mesh, order = 2, complex = self.domain_is_complex, autoupdate = False, definedon = "inner")#, dirichlet = "inner")
        self.fes = Compress(self.fes) #, complex = True

        #self.fes_copy = L2(mesh, order = 1, complex = True, definedon = "inner")#, dirichlet = "inner")
        #self.fes_copy = Compress(self.fes) #, complex = True
        
        self.cofes = L2(mesh, order = 2, complex = True, autoupdate = False, definedon = "outer")#, dirichlet = 'pmlregion')
        self.cofes = Compress(self.cofes)

        self.allfes = H1(mesh, order = 2, complex = True, autoupdate = False)
        self.allfes = Compress(self.allfes)

        self.domain_is_complex = domain_is_complex
        
        self.mesh = mesh
        
        self.solver = solver

        # n is the number of free source dofs.
        # Big M is all dofs on the entire region.
        self.n = self.fes.ndof
        self.M = self.cofes.ndof

        # Trial-test pairs for the two feses.

        u, v = self.fes.TnT()
        U, V = self.cofes.TnT()
        UU, VV = self.allfes.TnT()

        # The linear operators we require are mass matrix, mass matrix inverse,
        # prior covariance (half power), PDE forward and PDE adjoint.

        self.Mass = BilinearForm(self.fes)
        self.Mass += u * v * dx
        self.Mass.Assemble()
        self.Mass = self.Mass.mat
        self.InvMass = self.Mass.Inverse(freedofs = self.fes.FreeDofs(), inverse=self.solver)

        self.coMass = BilinearForm(self.cofes)
        self.coMass += U * V * dx
        self.coMass.Assemble()
        self.coMass = self.coMass.mat
        self.coInvMass = self.coMass.Inverse(freedofs = self.cofes.FreeDofs(), inverse=self.solver)

        # Helmholtz equation as PDE
        self.omega = omega

        self.PDE = BilinearForm(self.allfes)
        self.PDE += grad(UU)*grad(VV)*dx - omega**2*UU*VV*dx
        #a += -omega*1j*U*V * ds("pmlregion")
        self.PDE += -omega*1j*UU*VV * ds("outer")
        self.PDE.Assemble()
        self.iPDE = self.PDE.mat.Inverse(freedofs = self.allfes.FreeDofs(), inverse=self.solver)

        # Adjoint Helmholtz equation as adjoint PDE
        self.APDE = BilinearForm(self.allfes)
        self.APDE += grad(UU)*grad(VV)*dx - omega**2*UU*VV*dx
        #a += -omega*1j*U*V * ds("pmlregion")
        self.APDE += omega*1j*UU*VV * ds("outer")
        self.APDE.Assemble()
        self.iAPDE = self.APDE.mat.Inverse(freedofs = self.allfes.FreeDofs(), inverse=self.solver)
    
    def update(self):
        for el in self.mesh.Elements():
            self.mesh.SetRefinementFlag(el, el.mat != "mid")
        self.mesh.Refine()
        
        self.fes.Update()
        self.cofes.Update()
        self.allfes.Update()

        self.PDE.Assemble()
        self.iPDE = self.PDE.mat.Inverse(freedofs = self.allfes.FreeDofs(), inverse=self.solver)
        self.APDE.Assemble()
        self.iAPDE = self.APDE.mat.Inverse(freedofs = self.allfes.FreeDofs(), inverse=self.solver)

        class regpyPDE(NGSolveOperator):
            def __init__(self, domain, codomain, forw):
                super().__init__(domain = domain, codomain = codomain, linear = True)
                self.forw = forw
            
            def _eval(self, f_vec):
                f = self.domain.to_ngs(f_vec)
                return self.forw(f)

            def _adjoint(self, g_vec):
                pass
        
        rPDE = regpyPDE(domain = NgsSpace(self.fes), codomain = NgsSpace(self.cofes), forw = self.eval_PDE)
        setting = RegularizationSetting(op=rPDE, penalty=regpyL2, data_fid=regpyL2)
        return #setting.op_norm()

    def coeff_to_ngs(self, f_vec):

        f = GridFunction(self.fes)
        if self.domain_is_complex:
            f.vec.FV().NumPy()[:] = f_vec
        else:
            f.vec.FV().NumPy()[:] = np.real(f_vec)
        return f
        
    def ngs_to_coeff(self, g):
        
        v = self.fes.TestFunction()

        Mg = GridFunction(self.fes)
        return Mg.vec.FV().NumPy()

    def eval_PDE(self, f):
        VV = self.allfes.TestFunction()

        G = GridFunction(self.allfes)
        G.vec.data = self.iPDE * LinearForm(f * VV * dx).Assemble().vec
        
        g = GridFunction(self.cofes)
        g.Set(G)

        return g

    def adjoint_PDE(self, g):
        VV = self.allfes.TestFunction()
        
        F = GridFunction(self.allfes)
        F.vec.data = self.iAPDE * LinearForm(g * VV * dx).Assemble().vec

        f = GridFunction(self.fes)
        if self.domain_is_complex:
            f.Set(F)
        else:
            f.Set(np.real(F))
        return f
