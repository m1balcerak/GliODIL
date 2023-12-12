#!/usr/bin/env python3

import tensorflow as tf
from multigrid import (
    Multigrid,
    MultigridOp,
    SparseOperator,
    MultiIndex,
    StencilDict,
    get_shift_csr,
)
import multigrid
from multigrid_plot import plot_3d, plot_grid, plot_grid_matrix
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
from functools import partial
from time import time


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

    vop = Aop.mul_field(u)
    v = A @ u
    print(np.reshape(vop, nwh))
    print(np.reshape(v, nwh))
    print('|vop - v| =', np.linalg.norm(vop - v))

    vop = Aop.mul_transpose_field(uh)
    v = A.T @ uh
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


@tf.function(jit_compile=False)
def test_runtf(seed):
    nw = np.array((1025,)  * 2)
    nwh = nw // 2 + 1

    mod = tf
    dtype = mod.float64

    # Dicrete laplacian on base grid, as SparseOperator.
    u0 = mod.sin(seed + 0.1 + mod.linspace(0, 1, np.prod(nwh)))
    u1 = mod.sin(seed + 0.2 + mod.linspace(0, 1, np.prod(nwh)))
    u2 = mod.sin(seed + 0.3 + mod.linspace(0, 1, np.prod(nwh)))
    u3 = mod.sin(seed + 0.4 + mod.linspace(0, 1, np.prod(nwh)))
    A = SparseOperator(
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

    uh = mod.ones(np.prod(nwh), dtype=dtype)
    u = mod.ones(np.prod(nw), dtype=dtype)
    vh = A.mul_field(u)

    v = A.mul_transpose_field(uh)

    R = MultigridOp.get_R(nw, mod=mod, dtype=dtype)
    vh += R.mul_field(u)

    TT = MultigridOp.get_TT(nw, mod=mod, dtype=dtype)
    v += TT.mul_transpose_field(uh)

    dwa = np.array((-1, 2))  # Shift on base grid.
    dwc = np.array((1, -1))  # Shift on coarse grid.
    A = R.mul_elementwise(TT.shift_left(-dwc).shift_right(-dwa))
    vh += A.mul_field(u)

    u0 = mod.sin(seed + 0.1 + mod.linspace(0, 1, np.prod(nw)))
    u1 = mod.sin(seed + 0.2 + mod.linspace(0, 1, np.prod(nw)))
    u2 = mod.sin(seed + 0.3 + mod.linspace(0, 1, np.prod(nw)))
    u3 = mod.sin(seed + 0.4 + mod.linspace(0, 1, np.prod(nw)))
    A = SparseOperator(
        StencilDict(
            [
                (0, 0),
                (-1, 0),
                (0, -1),
                (1, 0),
            ],
            [
                u0,
                u1,
                u2,
                u3,
            ],
            mod=mod,
        ),
        nw,
        mod=mod,
        stride=1,
        dtype=u0.dtype,
    )
    u = mod.ones(np.prod(nw), dtype=dtype)

    A = A.mul_self_transpose()
    v += A.mul_field(u)

    Ah = MultigridOp.coarsen_sparse_operator(A, R, TT, mod=mod, dtype=dtype)
    vh += Ah.mul_field(uh)

    return vh, v


def test_tf(mod):
    v = mod.random.uniform((5, 5))
    #print(multigrid.noncircular_shift(v, (1, -1), mod=tf))
    #print(multigrid.noncircular_shift(v, (1, -1), mod=np))
    a00 = mod.ones((3, 3)) * 0
    a01 = mod.ones((3, 3)) * 1
    a10 = mod.ones((3, 3)) * 2
    a11 = mod.ones((3, 3)) * 3
    a = mod.stack([a00, a01, a10, a11])
    print(a.shape)
    a = mod.batch_to_space(a, block_shape=[2, 2], crops=[[0, 1]] * 2)[0]
    print(a)
    print(a.shape)


if __name__ == "__main__":
    #test_multigrid_sparse_operator(mod=np)
    #test_multigrid_sparse_operator(mod=tf)
    for seed in range(10):
        t = time()
        test_runtf(tf.constant(tf.cast(seed, tf.float64)))
        print("test_runtf(): {:.3}s".format(time() - t))
    #test_normal(mod=np)
    #test_normal(mod=tf)
    #print(np.array(test_runtf()))
    #print(np.array(test_runtf()))
