#!/usr/bin/env python3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from multigrid import get_shift_csr
import scipy.sparse as sp


def get_path(ext, suff=''):
    return os.path.splitext(os.path.basename(
        sys.argv[0]))[0] + suff + '.' + ext


def savefig(fig, suff='', ext='pdf', path=None, **kwargs):
    if path is None:
        path = get_path(ext, suff)
    elif ext is not None:
        path = os.path.splitext(path)[0] + '.' + ext
    print(path)
    metadata = {
        'Date': None
    } if ext == 'svg' else {
        'DateModified': None
    } if ext == 'pdf' else {}
    fig.savefig(path, metadata=metadata, **kwargs)


def plot_3d(mx,
            my,
            u,
            suff='sol',
            vmin=0,
            vmax=1,
            title=None,
            hook=None,
            xlabel='x',
            ylabel='y'):
    if vmin is None:
        vmin = u.min()
    if vmax is None:
        vmax = u.max()
    u = u.reshape(mx.shape)
    fig, ax = plt.subplots(figsize=(5, 5),
                           subplot_kw={
                               'projection': '3d',
                               'computed_zorder': False,
                           })
    ax.plot_surface(
        mx,
        my,
        u,
        cmap='jet',
        edgecolor='k',
        vmin=vmin,
        vmax=vmax,
        lw=0.1,
        alpha=0.8,
        rstride=1,
        cstride=1,
    )
    if hook:
        hook(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlim(vmin, vmax)
    title = suff if title is None else title
    ax.set_title(title)
    savefig(fig, suff='_' + suff, ext='png', dpi=250)
    plt.close(fig)


def plot_grid(u,
              nw,
              suff='grid',
              vmin=None,
              vmax=None,
              title=None,
              hook=None,
              cmap='bwr'):
    '''
    Visualizes a discrete field on a 2D grid
    '''
    u = np.reshape(u, nw)
    if vmax is None:
        vmax = abs(u).max()
    if vmin is None:
        vmin = -vmax
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.pcolormesh(u.T,
                  vmin=vmin,
                  vmax=vmax,
                  lw=0.2,
                  edgecolors='k',
                  cmap=cmap,
                  clip_on=False)
    if hook:
        hook(ax)
    title = suff if title is None else title
    ax.spines.left.set_visible(False)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.bottom.set_visible(False)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_xticks([])
    ax.set_yticks([])
    savefig(fig, suff='_' + suff, ext='png', dpi=250)
    plt.close(fig)


def plot_grid_matrix(matr,
                     nw,
                     suff='mgrid',
                     title=None,
                     stride=1,
                     scale=1,
                     width=2,
                     fontsize=4,
                     nodeindex=False):
    '''
    Visualizes stencil-based sparse matrix on a 2D grid.
    '''
    nw = np.array(nw)
    nwh = (nw - 1) // stride + 1
    fig, ax = plt.subplots(figsize=(3, 3))
    assert np.all(np.prod(nwh) == matr.shape[0]), (
        "Dimensions do not match, expected equal {:} and {:}".format(
            np.prod(nwh), matr.shape[0]))
    assert np.all(np.prod(nw) == matr.shape[1]), (
        "Dimensions do not match, expected equal {:} and {:}".format(
            np.prod(nw), matr.shape[1]))

    def flat(ix, iy):
        return ix * nw[0] + iy

    def flath(ixh, iyh):
        return ixh * nwh[0] + iyh

    xs = np.linspace(0, 1, nwh[0])
    ys = np.linspace(0, 1, nwh[1])
    hxs = 0.4 * scale / nw[0]
    hys = 0.4 * scale / nw[1]

    def entry(irow, icol):
        if 0 <= irow < matr.shape[0] and 0 <= icol < matr.shape[1]:
            return matr[irow, icol]
        return 0

    def put(ixh, iyh):
        x = xs[ixh]
        y = xs[iyh]
        ix = ixh * stride
        iy = iyh * stride
        ax.scatter(x, y, edgecolor='k', facecolor='w', lw=0.1, s=60, zorder=2)
        if nodeindex:
            ax.text(x + hxs * 0.5,
                    y + hys * 0.5,
                    '{:},{:}'.format(ixh, iyh),
                    va='center_baseline',
                    ha='center',
                    c='C0',
                    fontsize=fontsize)
        for dx in range(-width, width + 1):
            for dy in range(-width, width + 1):
                ixd = (ix + dx + nw[0]) % nw[0]
                iyd = (iy + dy + nw[1]) % nw[1]
                s = entry(flath(ixh, iyh), flat(ixd, iyd))
                s = "{:.2g}".format(s)
                if s == '0':
                    continue
                    s = '⋅'
                ax.text(x + dx * hxs,
                        y + dy * hys,
                        s,
                        va='center_baseline',
                        ha='center',
                        fontsize=fontsize,
                        zorder=5,
                        c='k')

    for iyh in range(len(ys)):
        for ixh in range(len(xs)):
            put(ixh, iyh)
    ax.pcolormesh(xs,
                  ys,
                  np.empty(nwh - 1),
                  lw=1,
                  edgecolors='0.9',
                  facecolors='none',
                  clip_on=False)
    title = suff if title is None else title
    ax.set_aspect('equal')
    ax.spines.left.set_visible(False)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.bottom.set_visible(False)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(xs.min() - hxs * 2, xs.max() + hxs * 2)
    ax.set_ylim(ys.min() - hys * 2, ys.max() + hys * 2)
    savefig(fig, suff='_' + suff, ext='png', dpi=400)
    plt.close(fig)


def example_plot_grid():
    nw = (9, 9)
    u = np.random.rand(*nw)
    plot_grid(u, nw)


def example_plot_grid_matrix():
    nw = np.array((9, 9))
    dc = sp.diags([np.ones(nw).flatten()], [0])
    dxm = sp.diags([np.ones(nw).flatten()], [0])
    dxp = sp.diags([np.ones(nw).flatten()], [0])
    dym = sp.diags([np.ones(nw).flatten()], [0])
    dyp = sp.diags([np.ones(nw).flatten()], [0])
    matr = sum([
        dc * get_shift_csr((0, 0), nw),
        dxm * get_shift_csr((1, 0), nw),
        dxp * get_shift_csr((-1, 0), nw),
        dym * get_shift_csr((0, 1), nw),
        dyp * get_shift_csr((0, -1), nw),
    ])
    plot_grid_matrix(matr, nw, stride=1)

    # Restrict to rectangular matrix
    # with columns over points `nw` and rows over coarse points `nwh`.
    nwh = nw // 2 + 1
    ii = np.array(range(matr.shape[0])).reshape(nw)
    ii = ii[::2, ::2].flatten()
    matr = matr[ii]
    plot_grid_matrix(matr, nw, stride=2, suff='mgrid2')


def example_plot_restriction():
    import multigrid
    nw = np.array((9, 9))
    nwh = (nw - 1) // 2 + 1


    dc = sp.diags([-4 * np.ones(nw).flatten()], [0])
    dxm = sp.diags([np.ones(nw).flatten()], [0])
    dxp = sp.diags([np.ones(nw).flatten()], [0])
    dym = sp.diags([np.ones(nw).flatten()], [0])
    dyp = sp.diags([np.ones(nw).flatten()], [0])
    A = sum([
        dc * get_shift_csr((0, 0), nw),
        dxm * get_shift_csr((1, 0), nw),
        dxp * get_shift_csr((-1, 0), nw),
        dym * get_shift_csr((0, 1), nw),
        dyp * get_shift_csr((0, -1), nw),
    ])

    R = multigrid.Multigrid.get_R(nw)
    T = multigrid.Multigrid.get_T(nw)

    plot_grid_matrix(R * 8, nw, stride=2, suff='R')
    plot_grid_matrix(T.T * 8, nw, stride=2, suff='TT')
    plot_grid_matrix(A, nw, stride=1, suff='A')
    plot_grid_matrix(R @ A @ T, nwh, stride=1, suff='RAT')


if __name__ == "__main__":
    example_plot_grid()
    example_plot_grid_matrix()
    example_plot_restriction()
