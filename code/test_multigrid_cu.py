#!/usr/bin/env python3

from multigrid import (
    Multigrid,
    MultigridOp,
    SparseOperator,
    MultiIndex,
    StencilDict,
    get_shift_csr,
    ModNumpy,
    ModTensorflow,
    ModCupy,
)
import multigrid
from multigrid_plot import plot_3d, plot_grid, plot_grid_matrix
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
from functools import partial
from time import time
import cupy as cp


def test_normal(mod=np):
    nw = np.array((5, 5))
    nwh = nw // 2 + 1

    def p(matr, name, nw=nw, scale=1, stride=1):
        plot_grid_matrix(matr.tocsr() * scale,
                         nw,
                         suff='mgrid_' + name,
                         width=2,
                         title=name + ' * {:}'.format(scale),
                         stride=stride)

    dtype = mod.float64

    # Dicrete laplacian on base grid, as SparseOperator.
    u0 = mod.sin(0.1 + mod.linspace(0, 1, np.prod(nwh)))
    u1 = mod.sin(0.2 + mod.linspace(0, 1, np.prod(nwh)))
    u2 = mod.sin(0.3 + mod.linspace(0, 1, np.prod(nwh)))
    u3 = mod.sin(0.4 + mod.linspace(0, 1, np.prod(nwh)))
    Aop = SparseOperator(
        StencilDict(
            [
                (0, 0),
                (-1, 0),
                (0, -1),
                (1, 0),
                (-1, 0),
            ],
            [
                u0,
                u1,
                u2,
                u3 * 0,
                u0,
            ],
            mod=mod,
        ),
        nw,
        mod=mod,
        stride=2,
        dtype=u0.dtype,
    )

    print('Aop', len(Aop.shift_to_field))
    Aop.eliminate_zeros()
    print('eliminate_zeros()', len(Aop.shift_to_field))
    Aop.shift_to_field.merge_duplicates()
    print('merge_duplicates()', len(Aop.shift_to_field))

    A = Aop.tocsr()
    #u = mod.sin(0.4 + mod.linspace(0, 1, np.prod(nwh)))
    uh = mod.ones(np.prod(nwh), dtype=dtype)
    u = mod.ones(np.prod(nw), dtype=dtype)

    vop = mod.numpy(Aop.mul_field(u))
    v = A @ mod.numpy(u)
    print(np.reshape(vop, nwh))
    print(np.reshape(v, nwh))
    print('|vop - v| =', np.linalg.norm(vop - v))

    vop = mod.numpy(Aop.mul_transpose_field(uh))
    v = A.T @ mod.numpy(uh)
    print(np.reshape(vop, nw))
    print(np.reshape(v, nw))
    print('|vop - v| =', np.linalg.norm(vop - v))
    return

    Bop = Aop.mul_self_transpose()
    p(Bop.tocsr(), 'Bop')

    # Dicrete laplacian on base grid, as a sparse matrix.
    A = Aop.tocsr()
    B = A.T @ A
    p(B, 'B')
    print('|B - Bop| =', sp.linalg.norm(B.tocsr() - Bop.tocsr(), 'fro'))


def test_multigrid_sparse_operator(mod=None):
    dtype = mod.float64
    nw = np.array((9, 9))
    nwh = nw // 2 + 1

    def p(matr, name, nw=nw, scale=16, stride=2):
        plot_grid_matrix(matr.tocsr() * scale,
                         nw,
                         suff='mgrid_' + name,
                         width=3,
                         title=name + ' * {:}'.format(scale),
                         stride=stride)

    R = Multigrid.get_R(nw)
    Rop = MultigridOp.get_R(nw, mod=mod, dtype=dtype)
    p(R, 'R')
    p(Rop, 'Rop')
    print('|R - Rop| =', sp.linalg.norm(R.tocsr() - Rop.tocsr(), 'fro'))

    dwa = np.array((-1, 2))  # Shift on base grid.
    dwc = np.array((1, -1))  # Shift on coarse grid.
    Ia = get_shift_csr(dwa, nw)
    Ic = get_shift_csr(dwc, nwh)

    TT = Multigrid.get_T(nw).T
    TTop = MultigridOp.get_TT(nw, mod=mod, dtype=dtype)
    p(TT, 'T')
    p(TTop, 'Top')
    print('|T - Top| =', sp.linalg.norm(TT.tocsr() - TTop.tocsr(), 'fro'))

    A = R * (Ic.T @ TT @ Ia.T)
    Aop = Rop.mul_elementwise(TTop.shift_left(-dwc).shift_right(-dwa))
    p(A, 'A')
    p(Aop, 'Aop')
    print('|A - Aop| =', sp.linalg.norm(A.tocsr() - Aop.tocsr(), 'fro'))



if __name__ == "__main__":
    modcp = multigrid.ModCupy(cp)
    modnp = multigrid.ModNumpy(np)
    test_multigrid_sparse_operator(mod=modcp)
    test_normal(mod=modcp)
