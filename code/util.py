#!/usr/bin/env python3

import sys
import os

import inspect
import time
import optimizer
import linsolver
import numpy as np
from collections import defaultdict
from util_plot import *
from re import findall
from tfwrap import tf

from multigrid import (
    Multigrid,
    MultigridOp,
    SparseOperator,
    StencilDict,
    ModNumpy,
    ModTensorflow,
    ModCupy,
)
from functools import partial

g_log_file = sys.stderr  # File used by printlog()
g_log_echo = False  # True if printlog() should print to stderr.

cupy = None
cupyx = None


def import_cupy(args):
    global cupy, cupyx
    import cupy
    import cupyx
    import cupyx.scipy.sparse
    import cupyx.scipy.sparse.linalg
    printlog("Using CuPy with memory limit {:.3f} GiB".format(
        cupy.get_default_memory_pool().get_limit() / (1 << 30)))
    mod = ModCupy(cupy, cupyx.scipy.sparse)
    return mod


def set_log_file(f, echo=False):
    global g_log_file, g_log_echo
    g_log_file = f
    g_log_echo = echo


def printlog(*msg):
    m = ' '.join(map(str, msg)) + '\n'
    if g_log_echo and g_log_file != sys.stderr:
        sys.stderr.write(m)
        sys.stderr.flush()
    g_log_file.write(m)
    g_log_file.flush()


def TIMECLEAR():
    global time_prev
    time_prev = time.time()


def TIME(pattern="{file}:{line}"):
    global time_prev
    if 'time_prev' not in globals():
        time_prev = time.time()
    file = os.path.basename(inspect.stack()[1][1])
    line = inspect.stack()[1][2]
    t = "{:.3f}s".format(time.time() - time_prev)
    d = {'file': file, 'line': line}
    printlog((pattern + ' ' + t).format(**d))
    time_prev = time.time()


class Timer():

    def __init__(self):
        self._starts = []  # Stack of pairs (key, time).
        self.counters = dict()  # Key to time.

    def push(self, key=None):
        self._starts.append((key, time.time()))

    def pop(self, key=None):
        start = self._starts.pop()
        assert start[0] is None or key is None or start[0] == key, \
                "Inconsistent keys passed to push() and pop(): "\
                "{:} and {:}".format(start[0], key)
        if key is None:
            key = start[0]
        dt = time.time() - start[1]
        self.counters[key] = self.counters.get(key, 0.) + dt

    def append(self, timer):
        for k in timer.counters:
            self.counters[k] = self.counters.get(k, 0.) + timer.counters[k]


def get_error(u, v):
    e1 = np.mean(abs(u - v))
    e2 = np.mean((u - v)**2)**0.5
    einf = np.max(abs(u - v))
    return e1, e2, einf


def add_arguments(parser):
    parser.add_argument('--epochs',
                        type=int,
                        default=None,
                        help="Maximum epochs, "
                        "defaults to product of plot_every and frames")
    parser.add_argument('--every_factor',
                        type=int,
                        default=1,
                        help="Multiplier for all *_every options")
    parser.add_argument('--plot_every',
                        type=int,
                        default=5,
                        help="Epochs between plots")
    parser.add_argument('--report_every',
                        type=int,
                        default=10,
                        help="Epochs between reports to stdout")
    parser.add_argument('--history_every',
                        type=int,
                        default=1,
                        help="Epochs between entries of training history")
    parser.add_argument('--checkpoint_every',
                        type=int,
                        default=5,
                        help="Epochs between checkpoints")
    parser.add_argument('--frames',
                        type=int,
                        default=10,
                        help="Frames to plot. Zero disables first frame.")
    parser.add_argument('--outdir',
                        type=str,
                        default='.',
                        help='Output directory')
    parser.add_argument('--optimizer',
                        type=str,
                        default='newton',
                        help="Optimizer")
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help="Seed for numpy.random and tensorflow.random")
    parser.add_argument('--plot_title',
                        type=int,
                        default=0,
                        help="Enable title in plots")
    parser.add_argument('--plotext',
                        type=str,
                        default='pdf',
                        help="Extension of plots")
    parser.add_argument('--history_full',
                        type=int,
                        default=0,
                        help="Number of epochs to write "
                        "history at every point")
    parser.add_argument('--montage',
                        type=int,
                        default=1,
                        help="Run montage after plotting")
    parser.add_argument('--double',
                        type=int,
                        default=0,
                        help="Double precision")
    parser.add_argument('--epoch_start',
                        type=int,
                        default=0,
                        help="Initial value of epoch")
    parser.add_argument('--frame_start',
                        type=int,
                        default=0,
                        help="Initial value of frame")
    parser.add_argument('--checkpoint',
                        type=str,
                        help="Continue from checkpoint in state_*.pickle")
    parser.add_argument('--checkpoint_train',
                        type=str,
                        help="Continue from history in state_*_train.pickle"
                        ". By default, infers the name from --checkpoint"
                        ". Set to '' to disable default behavior")
    parser.add_argument('--bfgs_m',
                        type=int,
                        default=50,
                        help="History size for L-BFGS")
    parser.add_argument('--multigrid',
                        type=int,
                        default=0,
                        help="Use multigrid decomposition")
    parser.add_argument('--mg_interp',
                        type=str,
                        default='conv',
                        choices=['roll', 'conv', 'manual'],
                        help="Multigrid interpolation method")
    parser.add_argument('--dump_data',
                        type=int,
                        default=1,
                        help="Dump data_*.pickle with every plot")
    parser.add_argument('--jac_nsmp0',
                        type=int,
                        default=50,
                        help="Number of samples "
                        "for initialization of Jacobi optimizer")
    parser.add_argument('--jac_nsmp1',
                        type=int,
                        default=1,
                        help="Number of samples "
                        "for each step of Jacobi optimizer")
    parser.add_argument('--jac_factor',
                        type=float,
                        default=1,
                        help="Factor for the diagonal update"
                        "for each step of Jacobi optimizer. "
                        "Increase above 1 for more weight to recent values")


def optimize_newton(args, problem, state, callback):
    opt = optimizer.Optimizer(name='newton', displayname='Newton')
    printlog("Running {} optimizer".format(opt.displayname))
    packed = problem.pack_state(state)
    dhistory = defaultdict(lambda: [])
    for epoch in range(args.epoch_start, args.epochs + 1):
        s = problem.unpack_state(packed)
        problem.domain.assign_active_state(state, s)
        const, m = problem.linearize(state, epoch=epoch)

        # Compute loss and residuals with initial state, to be used by callback.
        opt.last_loss = sum([np.mean(c**2) for c in const])
        opt.last_residual = [np.mean(c**2)**0.5 for c in const]
        callback(packed, epoch, dhistory=dhistory, opt=opt)
        if epoch == args.epochs:
            break

        opt.evals += 1
        const = np.hstack([eq.flatten() for eq in const])
        problem.timer_total.push('tt_linsolver')
        timer = Timer()
        timer.push('linsolver')
        dpacked = linsolver.solve(m, -const, args, dhistory, args.linsolver)
        timer.pop()
        problem.timer_total.append(timer)
        problem.timer_last.append(timer)
        packed += dpacked


def optimize_opt(args, optname, problem, state, callback, **kwargs):

    def loss_grad(packed, epoch):
        nonlocal opt
        packed = tf.cast(packed, problem.domain.dtype)
        s = problem.unpack_state(packed)
        problem.domain.assign_active_state(state, s)
        loss, grads, wgrads, loss_split = problem.eval_loss_grad(state,
                                                                 epoch=epoch)
        # FIXME: Pass Problem.Raw here.
        last_residual = [v**0.5 if v >= 0 else v for v in loss_split]
        g = problem.pack_fields(grads)
        wg = problem.pack_weights(wgrads)
        g = tf.concat([g, wg], axis=-1)
        return loss, g, last_residual

    packed = problem.pack_state(state)
    if args.bfgs_m is not None:
        kwargs['m'] = args.bfgs_m
    opt = optimizer.make_optimizer(optname,
                                   dtype=problem.domain.dtype,
                                   **kwargs)
    printlog("Running {} optimizer".format(opt.displayname))

    # Compute loss and residuals with initial state, to be used by callback.
    loss, _, residual = loss_grad(packed, args.epoch_start)
    opt.last_loss = loss.numpy()
    opt.last_residual = [r.numpy() for r in residual]
    callback(packed, epoch=args.epoch_start, opt=opt)

    packed, info = opt.run(packed,
                           loss_grad=loss_grad,
                           epochs=args.epochs - args.epoch_start,
                           callback=callback,
                           epoch_start=args.epoch_start,
                           lr=args.lr,
                           **kwargs)
    printlog(info)




def optimize_jacobi(args, problem, state, callback):
    opt = optimizer.Optimizer(name='jacobi', displayname='Jacobi')
    printlog("Running {} optimizer".format(opt.displayname))
    packed = problem.pack_state(state)
    dhistory = defaultdict(lambda: [])

    nsmp0 = args.jac_nsmp0
    nsmp1 = args.jac_nsmp1
    factor = args.jac_factor
    diag = problem.eval_diag(state, epoch=args.epoch_start, nsmp=nsmp0)
    ntotal = nsmp0

    for epoch in range(args.epoch_start, args.epochs + 1):
        s = problem.unpack_state(packed)
        problem.domain.assign_active_state(state, s)

        loss, grads, wgrads, loss_split = problem.eval_loss_grad(state,
                                                                 epoch=epoch)
        diag1 = problem.eval_diag(state, epoch=epoch, nsmp=nsmp1)

        def upd(avg0, n0, avg1, n1):
            return (avg0 * n0 + avg1 * n1 * factor) / (n0 + n1 * factor)

        diag = upd(diag, ntotal, diag1, nsmp1)
        ntotal += nsmp1

        g = problem.pack_fields(grads)
        wg = problem.pack_weights(wgrads)
        g = tf.concat([g, wg], axis=-1)
        dpacked = -(g / diag) * (args.lr * 0.5)

        # Compute loss and residuals with initial state, to be used by callback.
        opt.last_loss = loss.numpy()
        loss_split = [v.numpy() for v in loss_split]
        # FIXME: Pass Problem.Raw here.
        opt.last_residual = [v**0.5 if v >= 0 else v for v in loss_split]
        callback(packed, epoch, dhistory=dhistory, opt=opt)
        if epoch == args.epochs:
            break

        opt.evals += 1

        packed += dpacked


def optimize(args, optname, problem, state, callback, **kwargs):
    if optname == 'newton':
        return optimize_newton(args, problem, state, callback)
    elif optname == 'jacobi':
        return optimize_jacobi(args, problem, state, callback)
    return optimize_opt(args, optname, problem, state, callback)


def get_memory_usage_kb():
    '''
    Returns current memory usage in KiB.
    '''
    #with open('/proc/self/status') as f:
    #    res = findall('VmRSS:[^0-9]*([0-9]*)', f.read())[0]
    return 0
    #return int(res)


def optimize_multigrid_base(opt,
                            args,
                            problem,
                            state,
                            callback=None,
                            mod=None,
                            datalevels=None):

    verbose = args.linsolver_verbose

    def printv(*m):
        if verbose:
            printlog(*m)

    printv("Running {} optimizer".format(opt.displayname))
    packed = problem.pack_state(state)
    dhistory = defaultdict(lambda: [])
    printv("Constructing Multigrid...")
    mg = Multigrid(problem.domain.shape,
                   restriction=args.restriction,
                   nvar=len(state.fields),
                   mod=mod,
                   dtype=problem.domain.dtype,
                   nlevels=args.nlvl)
    printv('levels: {}'.format(', '.join(map(str, mg.nnw))))
    for epoch in range(args.epoch_start, args.epochs + 1):
        s = problem.unpack_state(packed)
        problem.domain.assign_active_state(state, s)
        if callback:
            callback(packed, epoch, dhistory=dhistory, opt=opt)
        if epoch == args.epochs:
            break

        timer = Timer()
        timer.push('linsolver')
        if datalevels is not None:
            AA = [None] * mg.nlevels
            for level in range(mg.nlevels):
                printv("level=", level)
                lproblem = datalevels[level].problem
                lstate = datalevels[level].state

                def coarsen(u):
                    R = mg.RRsingle[level - 1]
                    u = mod.reshape(u, [-1])
                    return mod.reshape(R @ u, lproblem.domain.shape)

                if level > 0:
                    for k in problem.domain.fieldnames:
                        lstate.fields[k].assign(
                            coarsen(datalevels[level - 1].state.fields[k]))

                const, m = lproblem.linearize(lstate,
                                              epoch=epoch,
                                              mod=mod.mod,
                                              modsp=mod.modsp)
                opt.last_residual = [np.mean(c**2)**0.5 for c in const]
                const = mod.stack([c for c in const]).flatten()
                if level == 0:
                    rhs = -m.T @ const
                AA[level] = m.T @ m
                if level == 0:
                    matr = AA[0]
            opt.evals += 1
            mg.update_A(AA)
        else:
            printv("Evaluating gradients...")
            const, m = problem.linearize(state,
                                         epoch=epoch,
                                         mod=mod.mod,
                                         modsp=mod.modsp)
            opt.last_residual = [mod.numpy(mod.norm(c)) for c in const]
            opt.evals += 1
            const = mod.stack([c for c in const]).flatten()
            printv("Computing rhs...")
            rhs = -m.T @ const
            printv("Computing matr...")
            matr = m.T @ m

            printv("Calling update_A()...")
            mg.update_A(matr)

        sol = np.zeros_like(rhs)
        for it in range(args.linsolver_maxiter):
            sol = mg.step(sol,
                          rhs,
                          ndirect=args.ndirect,
                          pre=args.smooth_pre,
                          post=args.smooth_post,
                          smoother=partial(mg.smoother_jacobi,
                                           omega=args.omega,
                                           full=False))
            if verbose > 1:
                printv('it={:d} r={:.5g}'.format(
                    it,
                    np.mean((rhs - matr @ sol)**2)**0.5))
        dpacked = mod.numpy(sol)
        timer.pop()
        problem.timer_total.append(timer)
        problem.timer_last = timer
        packed += dpacked


def optimize_multigrid(args, problem, state, callback, datalevels=None):
    import scipy.sparse
    mod = ModNumpy(np, scipy.sparse)
    opt = optimizer.Optimizer(name='multigrid', displayname='Multigrid')
    return optimize_multigrid_base(opt,
                                   args,
                                   problem,
                                   state,
                                   callback,
                                   mod=mod,
                                   datalevels=datalevels)


def optimize_multigridcp(args, problem, state, callback):
    mod = import_cupy(args)
    opt = optimizer.Optimizer(name='multigridcp', displayname='MultigridCupy')
    return optimize_multigrid_base(opt,
                                   args,
                                   problem,
                                   state,
                                   callback,
                                   mod=mod)


def optimize_multigridop_base(opt,
                              args,
                              problem,
                              state,
                              callback=None,
                              mod=None):
    verbose = args.linsolver_verbose
    domain = problem.domain

    def printv(m):
        if verbose:
            printlog(m)

    printv("Running {} optimizer".format(opt.displayname))
    packed = problem.pack_state(state)
    dhistory = defaultdict(lambda: [])
    printv("Constructing MultigridOp...")
    nvar = len(state.fields)
    mg = MultigridOp(domain.shape,
                     nvar=nvar,
                     restriction=args.restriction,
                     mod=mod,
                     dtype=domain.dtype,
                     nlevels=args.nlvl)
    printv('levels: {}'.format(', '.join(map(str, mg.nnw))))
    for epoch in range(args.epochs + 1):
        s = problem.unpack_state(packed)
        domain.assign_active_state(state, s)
        if callback:
            callback(packed, epoch, dhistory=dhistory, opt=opt)
        if epoch == args.epochs:
            break

        printv("Evaluating gradients...")
        epoch = tf.constant(epoch, dtype=domain.dtype)
        const, grads, field_desc, wgrads = problem._eval_grad(
            state.fields, state.weights, epoch)
        opt.last_residual = [np.mean(c**2)**0.5 for c in const]

        neqn = len(const)  # Number of equations.

        nw = domain.shape
        stf = [[dict() for _ in range(nvar)] for _ in range(neqn)]
        for i in range(neqn):  # Loop over equations.
            for j in range(len(field_desc)):  # Loop over grid field variables.
                varindex, dw = field_desc[j]
                g = grads[i][j]
                if g is None:
                    continue
                if tf.reduce_sum(g**2) == 0:
                    continue
                stf[i][varindex][tuple(np.array(dw))] = mod.native(g.numpy())
        const = [mod.native(c.numpy()) for c in const]
        printv("Constructing SparseOperator...")
        mm = [[
            SparseOperator(s, nw, mod=mod, dtype=domain.dtype) for s in row
        ] for row in stf]
        del stf

        opt.evals += 1
        timer = Timer()
        timer.push('linsolver')
        printv("Computing rhs...")
        rhs = [
            -sum([mm[i][j].mul_transpose_field(const[i]) for i in range(neqn)])
            for j in range(nvar)
        ]
        printv("Computing matr...")
        matr = [[None for _ in range(nvar)] for _ in range(nvar)]
        for i in range(nvar):
            for j in range(nvar):
                matr[i][j] = mm[0][i].mul_transpose_op(mm[0][j])
                for e in range(1, neqn):
                    matr[i][j] = matr[i][j].add_elementwise(
                        mm[e][i].mul_transpose_op(mm[e][j]))
        del mm
        printv("Calling update_A()...")
        mg.update_A(matr)
        sol = [np.zeros_like(rhs[i]) for i in range(nvar)]
        for it in range(args.linsolver_maxiter):
            sol = mg.step(sol,
                          rhs,
                          pre=args.smooth_pre,
                          post=args.smooth_post,
                          ndirect=args.ndirect,
                          smoother=partial(mg.smoother_jacobi,
                                           omega=args.omega))
            if verbose > 1:
                printv('it={:d} r={}'.format(
                    it, ', '.join('{:.5g}'.format(np.mean(r**2)**0.5)
                                  for r in mg.residual(matr, sol, rhs))))
        dpacked = mod.numpy(mod.reshape(mod.stack(sol), [-1]))
        timer.pop()
        problem.timer_total.append(timer)
        problem.timer_last = timer
        packed += dpacked


def optimize_multigridop(args, problem, state, callback):
    import scipy.sparse
    mod = ModNumpy(np, scipy.sparse)
    opt = optimizer.Optimizer(name='multigridop', displayname='MultigridOp')
    return optimize_multigridop_base(opt,
                                     args,
                                     problem,
                                     state,
                                     callback,
                                     mod=mod)


def optimize_multigridopcp(args, problem, state, callback):
    mod = import_cupy(args)
    opt = optimizer.Optimizer(name='multigridopcp',
                              displayname='MultigridOpCupy')
    return optimize_multigridop_base(opt,
                                     args,
                                     problem,
                                     state,
                                     callback,
                                     mod=mod)
