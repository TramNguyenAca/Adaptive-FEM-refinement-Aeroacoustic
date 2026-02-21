from regpy.solvers import RegSolver

import logging
#logging.basicConfig(level=logging.WARNING)
import numpy as np
from copy import deepcopy

class Landweber(RegSolver):
    r"""The linear Landweber method. Solves the linear, ill-posed equation
    \[
        T(x) = g^\delta,
    \]
    in Hilbert spaces by gradient descent for the residual
    \[
        \Vert T(x) - g^\delta\Vert^2,
    \]
    where \(\Vert\cdot\Vert)\ is the Hilbert space norm in the codomain, and gradients are computed with
    respect to the Hilbert space structure on the domain.

    The number of iterations is effectively the regularization parameter and needs to be picked
    carefully.

    Parameters
    ----------
    setting : regpy.solvers.RegularizationSetting
        The setting of the forward problem.
    data : array-like
        The measured data/right hand side.
    init : array-like
        The initial guess.
    stepsize : float, optional
        The step length; must be chosen not too large. If omitted, it is guessed from the norm of
        the derivative at the initial guess.
    """

    def __init__(self, setting, data, init, stepsize=None, error_func = None, perfect_error_func = None, accel = True):
        super().__init__(setting)
        self.setting = setting
        self.rhs = data
        """The right hand side gets initialized to measured data"""
        self.x = init
        self.y = self.op(self.x)
        norm = setting.op_norm()
        self.stepsize = stepsize or 0.9 / norm**2
        """The stepsize."""
        self.error_func = error_func
        self.perfect_error_func = perfect_error_func
        self.errors = []
        self.perfect_errors = []
        self.residuals = []
        self.iters = 0
        self.accel = accel

    def _next(self):
        self.iters += 1
        self._residual = self.y - self.rhs
        self._gy_residual = self._residual #self.h_codomain.gram(self._residual)
        self._update = self.op.adjoint(self._gy_residual)
        self.x_old = deepcopy(self.x)
        self.x -= self.stepsize * self._update #self.h_domain.gram_inv(self._update)
        if self.accel:
            self.y = self.op(self.x + (self.iters - 1)/(self.iters + 2) * (self.x - self.x_old))
        else:
            self.y = self.op(self.x)
        
        if self.log.isEnabledFor(logging.INFO):
            if self.error_func is not None:
                norm_error = self.error_func(self.x)
                self.log.info('|error| = {}'.format(norm_error))
                self.errors.append(norm_error)
            if self.perfect_error_func is not None:
                norm_error = self.perfect_error_func(self.x)
                self.log.info('|prfer| = {}'.format(norm_error))
                self.perfect_errors.append(norm_error)
            norm_residual = self.setting.h_codomain.norm(self._residual)
            self.log.info('|residual| = {}'.format(norm_residual))
            self.residuals.append(norm_residual)
            