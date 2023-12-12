#!/usr/bin/env python3

import multigrid
multigrid.g_reuse_multigridop = False
from multigrid import (
    Multigrid,
    MultigridOp,
    MultigridDecomp,
    SparseOperator,
    get_shift_csr,
    ModNumpy,
    ModTensorflow,
    ModCupy,
)
from multigrid_plot import plot_3d, plot_grid, plot_grid_matrix
import numpy as np

import scipy.sparse as sp
import scipy.sparse.linalg
from functools import partial
import unittest
import os
import sys


class Dummy:

    class TestMultigridBase(unittest.TestCase):

        def setUp(self):
            self.plot = os.environ.get("PLOT", False)
            self.log = os.environ.get("LOG", False)
            self.dtype = self.mod.float64

        def assertSmall(self, value, thres=1e-14):
            if not value <= thres:
                self.fail('Expected small value, got {:.8g}'.format(value))

        def assertClose(self, value, expected, thres=1e-14):
            if not abs(value - expected) <= thres:
                self.fail('Expected {:.8g}, got {:.8g}'.format(
                    expected, value))

        def cast(self, value, dtype=None):
            if dtype is None:
                dtype = self.dtype
            try:
                return self.mod.cast(value, dtype)
            except:
                return self.mod.array(value, dtype=dtype)

        def rand(self, shape, seed):
            mod = self.mod
            return mod.sin(seed * 0.123 +
                           mod.linspace(self.cast(0), self.cast(1), shape))

        def test_sparse_operator(self):
            mod = self.mod
            dtype = self.dtype
            nw = np.array((9, 9))
            nwh = nw // 2 + 1

            def p(matr, name, nw=nw, scale=16, stride=2):
                if not self.plot:
                    return
                plot_grid_matrix(matr.tocsr() * scale,
                                 nw,
                                 suff='mgrid_' + name,
                                 width=3,
                                 title=name + ' * {:}'.format(scale),
                                 stride=stride)

            restriction = 'full'
            R = Multigrid.get_R(nw,
                                mod=mod,
                                dtype=dtype,
                                restriction=restriction)
            Rop = MultigridOp.get_R(nw,
                                    mod=mod,
                                    dtype=dtype,
                                    restriction=restriction)
            p(R, 'R')
            p(Rop, 'Rop')
            self.assertSmall(mod.spnorm(R.tocsr() - Rop.tocsr(), 'fro'))

            dwa = np.array((-1, 2))  # Shift on base grid.
            dwc = np.array((1, -1))  # Shift on coarse grid.

            Ia = get_shift_csr(dwa, nw, mod=self.mod, dtype=self.dtype)
            Ic = get_shift_csr(dwc, nwh, mod=self.mod, dtype=self.dtype)

            TT = Multigrid.get_T(nw, mod=mod, dtype=dtype).T
            TTop = MultigridOp.get_TT(nw, mod=mod, dtype=dtype)
            p(TT, 'T')
            p(TTop, 'Top')
            self.assertSmall(mod.spnorm(TT.tocsr() - TTop.tocsr(), 'fro'))

            A = R.multiply(Ic.T @ TT @ Ia.T)
            Aop = Rop.mul_elementwise(TTop.shift_left(-dwc).shift_right(-dwa))

            p(A, 'A')
            p(Aop, 'Aop')
            self.assertSmall(mod.spnorm(A.tocsr() - Aop.tocsr(), 'fro'))

        def test_mul_transpose_field(self):
            mod = self.mod
            nw = np.array((5, 5))
            for stride in [1, 2]:
                with self.subTest(stride=stride):
                    nwh = (nw - 1) // stride + 1

                    u0 = self.rand(np.prod(nwh), 0)
                    u1 = self.rand(np.prod(nwh), 1)
                    u2 = self.rand(np.prod(nwh), 2)
                    u3 = self.rand(np.prod(nwh), 3)
                    u4 = self.rand(np.prod(nwh), 4)
                    Aop = SparseOperator(
                        {
                            (0, 0): u0,
                            (-1, 0): u1,
                            (0, -1): u2,
                            (1, 0): u3,
                            (1, 1): u4,
                        },
                        nw,
                        mod=mod,
                        stride=stride,
                        dtype=u0.dtype,
                    )
                    A = Aop.tocsr()

                    u = self.rand(np.prod(nwh), 9)

                    vop = Aop.mul_transpose_field(u)
                    v = A.T @ u
                    self.assertSmall(mod.norm(vop - v))

        def test_mul_transpose_field_mg(self):
            mod = self.mod
            nw = np.array((9, 9))
            nwh = nw // 2 + 1

            TTop = MultigridOp.get_TT(nw, mod=self.mod, dtype=self.dtype)
            T = Multigrid.get_T(nw, mod=self.mod, dtype=self.dtype)

            def p(u, nw, suff):
                if not self.plot:
                    return
                plot_grid(u, nw, suff=suff, vmin=0, vmax=1, cmap='viridis')

            uh = mod.sin(mod.linspace(0, 1, np.prod(nwh)))
            p(uh, nwh, suff='uh')
            u = T @ mod.spnative(uh).flatten()
            p(u, nw, suff='u')
            uop = TTop.mul_transpose_field(uh)
            p(uop, nw, suff='uop')
            self.assertSmall(mod.norm(u - uop))

        def test_normal(self):
            mod = self.mod
            dtype = self.dtype
            nw = np.array((5, 5))

            def p(matr, name, nw=nw, scale=1, stride=1):
                if not self.plot:
                    return
                plot_grid_matrix(matr.tocsr() * scale,
                                 nw,
                                 suff='mgrid_' + name,
                                 width=2,
                                 title=name + ' * {:}'.format(scale),
                                 stride=stride)

            # Dicrete laplacian on base grid, as SparseOperator.
            u0 = self.rand(np.prod(nw), 1)
            u1 = self.rand(np.prod(nw), 2)
            u2 = self.rand(np.prod(nw), 3)
            Aop = SparseOperator(
                {
                    (0, 0): u0,
                    (-1, 0): u1,
                    (0, -1): u2,
                },
                nw,
                mod=mod,
                dtype=dtype,
            )
            Bop = Aop.mul_self_transpose()
            p(Bop.tocsr(), 'Bop')

            # Dicrete laplacian on base grid, as a sparse matrix.
            A = Aop.tocsr()
            B = A.T @ A
            p(B, 'B')
            self.assertSmall(mod.spnorm((B.tocsr() - Bop.tocsr()), 'fro'))

        def test_coarsen(self):
            mod = self.mod
            dtype = self.dtype
            nw = np.array((11, 11))
            nwh = (nw - 1) // 2 + 1

            def p(matr, name, nw=nw, scale=16, stride=1):
                if not self.plot:
                    return
                plot_grid_matrix(matr.tocsr() * scale,
                                 nw,
                                 suff='mgrid_' + name,
                                 width=2,
                                 title=name + ' * {:}'.format(scale),
                                 stride=stride)

            R = Multigrid.get_R(nw, mod=mod, dtype=dtype)
            Rop = MultigridOp.get_R(nw, mod=mod, dtype=dtype)
            TT = Multigrid.get_T(nw, mod=mod, dtype=dtype).T
            TTop = MultigridOp.get_TT(nw, mod=mod, dtype=dtype)

            # Dicrete laplacian on base grid, as SparseOperator.
            pad = [[1, 1]] * 2
            Aop = SparseOperator(
                {
                    (0, 0): 4 * mod.ones(nw, dtype),
                    (-1, 0): -mod.pad(mod.ones(nw - 2, dtype), pad),
                    (1, 0): -mod.pad(mod.ones(nw - 2, dtype), pad),
                    (0, -1): -mod.pad(mod.ones(nw - 2, dtype), pad),
                    (0, 1): -mod.pad(mod.ones(nw - 2, dtype), pad),
                },
                nw,
                mod=mod,
                dtype=dtype)
            # Dicrete laplacian on base grid, as a sparse matrix.
            A = Aop.tocsr()

            p(A, 'A', stride=1, scale=1)
            p(Aop, 'Aop', stride=1, scale=1)
            self.assertSmall(mod.spnorm(A.tocsr() - Aop.tocsr(), 'fro'))

            # Dicrete laplacian on coarse grid, as SparseOperator.
            Ahop = MultigridOp.coarsen_sparse_operator(Aop,
                                                       Rop,
                                                       TTop,
                                                       mod=mod,
                                                       dtype=dtype)
            # Dicrete laplacian on coarse grid, as a sparse matrix.
            Ah = R @ A @ TT.T

            p(Ah, 'Ah', nw=nwh, scale=16)
            p(Ahop, 'Ahop', nw=nwh, scale=16)
            self.assertSmall(mod.spnorm(Ah.tocsr() - Ahop.tocsr(), 'fro'))

        def test_multigrid_poisson(self,
                                   nx=16,
                                   dim=2,
                                   plot=False,
                                   use_mg=True,
                                   use_mgop=True,
                                   maxiter=10,
                                   use_exact=True):
            mod = self.mod
            dtype = self.dtype
            nw = np.array((nx, ) * dim) + 1
            nwh = (nw - 1) // 2 + 1

            dx = 1 / (nw[0] - 1)
            denom = dx**2

            def printlog(m):
                if not self.log:
                    return
                sys.stderr.write(str(m) + '\n')
                sys.stderr.flush()

            printlog("\ndim={:}".format(dim))
            printlog("Constructing SparseOperator...")
            stf = dict()
            dwz = (0, ) * dim
            stf[dwz] = 2 * dim * mod.ones(nw, dtype) / denom
            pad = [[1, 1]] * 2
            for i in range(dim):
                dw = tuple(1 if j == i else 0 for j in range(dim))
                stf[dw] = -mod.pad(mod.ones(nw - 2, dtype), pad) / denom
                dw = tuple(-1 if j == i else 0 for j in range(dim))
                stf[dw] = -mod.pad(mod.ones(nw - 2, dtype), pad) / denom
            # Dicrete laplacian on base grid as SparseOperator.
            Aop = SparseOperator(stf, nw, mod=mod, dtype=dtype)

            if plot:
                assert dim == 2, "Plotting requires dim=2"

            if plot or use_mg:
                # Dicrete laplacian on base grid, as a sparse matrix.
                A = Aop.tocsr()

            # Right-hand side.
            rhs = mod.ones(nw, dtype=dtype) * 10
            rhs = mod.reshape(rhs, -1)

            if plot or use_exact:
                printlog("Exact solution...")
                u_exact = mod.spsolve(Aop.tocsr(), rhs)

            if plot:
                mxone = np.linspace(0, 1, nw[0])
                myone = np.linspace(0, 1, nw[1])
                mx, my = np.meshgrid(mxone, myone)
                plot_3d(mx, my, u_exact, suff='u_exact')

            if use_mgop:
                # Multigrid with SparseOperator.
                printlog("Initializing MultigridOp...")
                mgop = MultigridOp(nw, mod=mod, dtype=dtype)
                printlog("update_A()...")
                mgop.update_A([[Aop]])
                u = mod.zeros_like(rhs)
                frame = 0
                for it in range(maxiter):
                    if plot:
                        plot_3d(
                            mx,
                            my,
                            u,
                            suff='uop_frame{:03d}'.format(frame),
                            title='uop_frame{:03d} it={:03d}'.format(
                                frame, it),
                        )
                        frame += 1
                    u = mgop.step(
                        [u],
                        [rhs],
                        smoother=partial(mgop.smoother_jacobi, omega=0.7),
                    )[0]
                    printlog('it={:d} res={:.5g}'.format(  #
                        it, np.linalg.norm(rhs - Aop.mul_field(u))))
                self.assertSmall(np.linalg.norm(rhs - Aop.mul_field(u)),
                                 thres=1e-6)
                if use_exact:
                    printlog('|u-u_exact|={:.5g}'.format(mod.norm(u -
                                                                  u_exact)))

            if use_mg:
                # Multigrid with sparse matrices.
                printlog("Initializing Multigrid...")
                mg = Multigrid(nw, mod=self.mod, dtype=self.dtype)
                printlog("update_A()")
                mg.update_A(A)
                u = mod.zeros_like(rhs)
                frame = 0
                for it in range(maxiter):
                    if plot:
                        plot_3d(
                            mx,
                            my,
                            u,
                            suff='u_frame{:03d}'.format(frame),
                            title='u_frame{:03d} it={:03d}'.format(frame, it),
                        )
                        frame += 1
                    u = mg.step(
                        u,
                        rhs,
                        smoother=partial(mg.smoother_jacobi, omega=0.7),
                    )
                    printlog('it={:d} res={:.5g}'.format(  #
                        it, np.linalg.norm(rhs - A @ u)))
                self.assertSmall(np.linalg.norm(rhs - A @ u), thres=1e-6)
                if use_exact:
                    printlog('|u-u_exact|={:.5g}'.format(
                        np.linalg.norm(u - u_exact)))


class TestMultigridNumpy(Dummy.TestMultigridBase):

    def setUp(self):
        self.mod = ModNumpy(np, sp)
        super().setUp()


try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
except ImportError:
    print("Skipping Tensorflow tests, module tensorflow not found")
else:

    class TestMultigridTensorflow(Dummy.TestMultigridBase):

        def setUp(self):
            self.mod = ModTensorflow(tf, sp)
            super().setUp()

    class TestMultigridDecomp(unittest.TestCase):

        def setUp(self):
            self.mod = ModTensorflow(tf, sp)
            self.dtype = self.mod.float64

        def assertSmall(self, value, thres=1e-14):
            if not value <= thres:
                self.fail('Expected small value, got {:.8g}'.format(value))

        def assertClose(self, value, expected, thres=1e-14):
            if not abs(value - expected) <= thres:
                self.fail('Expected {:.8g}, got {:.8g}'.format(
                    expected, value))

        def cast(self, value, dtype=None):
            if dtype is None:
                dtype = self.dtype
            try:
                return self.mod.cast(value, dtype)
            except:
                return self.mod.array(value, dtype=dtype)


        def rand(self, shape, seed=0):
            mod = self.mod
            res = mod.linspace(self.cast(0), self.cast(1), np.prod(shape))
            res = mod.sin(seed * 0.123 + res)
            res = mod.reshape(res, shape)
            return res

        def test_interp_1d(self):
            mod = self.mod
            dtype = self.dtype
            nw = np.array((9,))
            nwh = nw // 2 + 1
            uh = self.rand(nwh)
            # First.
            ua = MultigridDecomp.interp_field(uh, mod=self.mod)
            # Second.
            T = Multigrid.get_T(nw, mod=mod, dtype=dtype)
            ub = mod.reshape(T @ mod.reshape(uh, [-1]), nw)
            self.assertSmall(mod.norm(ua - ub))

        def test_interp_2d(self):
            mod = self.mod
            dtype = self.dtype
            nw = np.array((9, 9))
            nwh = nw // 2 + 1
            uh = self.rand(nwh)
            # First.
            ua = MultigridDecomp.interp_field(uh, mod=self.mod)
            # Second.
            T = Multigrid.get_T(nw, mod=mod, dtype=dtype)
            ub = mod.reshape(T @ mod.reshape(uh, [-1]), nw)
            self.assertSmall(mod.norm(ua - ub))

        def test_interp_3d(self):
            mod = self.mod
            dtype = self.dtype
            nw = np.array((9, 9, 9))
            nwh = nw // 2 + 1
            uh = self.rand(nwh)
            # First.
            ua = MultigridDecomp.interp_field(uh, mod=self.mod)
            # Second.
            T = Multigrid.get_T(nw, mod=mod, dtype=dtype)
            ub = mod.reshape(T @ mod.reshape(uh, [-1]), nw)
            self.assertSmall(mod.norm(ua - ub))


try:
    import cupy as cp
    import cupyx.scipy.sparse
    import cupyx.scipy.sparse.linalg
except ImportError:
    print("Skipping CuPy tests, module cupy not found")
else:

    class TestMultigridCupy(Dummy.TestMultigridBase):

        def setUp(self):
            self.mod = ModCupy(cp, cupyx.scipy.sparse)
            super().setUp()


if __name__ == "__main__":
    unittest.main()
