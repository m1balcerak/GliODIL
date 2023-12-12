#!/usr/bin/env python3

import scipy.sparse
import numpy as np
import os
import math
from util import TIME, TIMECLEAR, Timer
import pickle
from argparse import Namespace
import util_min
from multigrid import Multigrid, MultigridDecomp

from tfwrap import tf


class Domain:

    def __init__(
        self,
        ndim=None,
        lower=0.,
        upper=1.,
        shape=None,
        varnames=['x', 'y', 'z'],  # Independent variables.
        fieldnames=[],  # Unknown fields.
        neuralnets=dict(),  # Neural networks dict(name:layers)
        frozen_weights=[],
        frozen_fields=[],
        dtype=np.float64,
        multigrid=False,
        mg_nlvl=None,
        mg_factors=None,
        mg_axes=None,
        mg_interp=None,
        mg_cell=False,
    ):
        '''
        multigrid: `bool`
            Use multigrid decomposition for fields in `fieldnames`.
        mg_nlvl: `int`
            Number of multigrid levels. Defaults to maximum possible.
        mg_factors: `list` of `int`
            Factors of each level. Defaults to ones.
        mg_axes: `list` of `bool`
            Axes in which to use multigrid decomposition. Defaults to all.
        mg_interp: `str`
            Multigrid interpolation method. See `MultigridDecomp.interp_field()`.
        mg_cell: `bool`
            Use cell-based interpolation if True, otherwise node-based.
        '''
        if ndim is None:
            ndim = len(varnames)
        self.ndim = ndim
        if shape is not None and ndim is not None:
            assert len(shape) == ndim
        self.lower = np.ones(ndim, dtype=dtype) * lower
        self.upper = np.ones(ndim, dtype=dtype) * upper
        self.varnames = varnames[:ndim]
        self.dtype = dtype
        self.shape = shape

        # Multigrid decomposition
        self.multigrid = multigrid
        self.mg_factors = mg_factors
        mg_axes = [True] * ndim if mg_axes is None else mg_axes
        nlvl_max = min(
            round(np.log2(n if mg_cell else n - 1)) if ax else max(shape)
            for n, ax in zip(shape, mg_axes))
        mg_nlvl = nlvl_max if mg_nlvl is None else min(mg_nlvl, nlvl_max)
        self.mg_nlvl = mg_nlvl
        self.mg_cell = mg_cell
        self.mg_nnw = [
            tuple(
                ((n >> lvl) if mg_cell else ((n - 1) >> lvl) + 1) if ax else n
                for n, ax in zip(shape, mg_axes))
            for lvl in range(self.mg_nlvl)
        ]
        MultigridDecomp.check_nnw(self.mg_nnw,
                                  use_axes=mg_axes,
                                  cell=self.mg_cell)
        self.mg_axes = mg_axes
        self.mg_interp = mg_interp

        mg_fieldnames = []
        if multigrid:
            netsize = sum([np.prod(nw) for nw in self.mg_nnw])
            for name in fieldnames:
                if name in neuralnets:
                    raise RuntimeError(
                        "Name collision of field and neuralnet '{}'".format(
                            name))
                neuralnets[name] = [0, netsize]
                mg_fieldnames.append(name)
        self.fieldnames = fieldnames
        self.mg_fieldnames = mg_fieldnames
        self.neuralnets = neuralnets
        self.frozen_weights = frozen_weights
        # Names of active weights.
        self.aweights = [k for k in neuralnets if k not in frozen_weights]
        # Compute shapes of weights.
        self.weight_shapes = dict()
        for name in neuralnets:
            layers = neuralnets[name]
            res = []
            for ni, no in zip(layers[:-1], layers[1:]):
                res.append((ni, no))  # Weights.
                res.append((no, ))  # Biases.
            self.weight_shapes[name] = res

    def fields_size(self):
        return len(self.fieldnames) * np.prod(self.shape)

    def get_minimal(self):
        return util_min.Domain(self)

    def aweights_size(self):
        res = 0
        for name in self.aweights:
            res += sum(np.prod(s) for s in self.weight_shapes[name])
        return res

    def weights_size(self):
        res = 0
        for name in self.neuralnets:
            res += sum(np.prod(s) for s in self.weight_shapes[name])
        return res

    def step_by_dim(self, i):
        return (self.upper[i] - self.lower[i]) / self.shape[i]

    def cell_center_1d(self, i):
        x = np.linspace(self.lower[i],
                        self.upper[i],
                        self.shape[i],
                        endpoint=False,
                        dtype=self.dtype)
        x += (x[1] - x[0]) * 0.5
        return x

    def cell_center_all(self):
        xx = [self.cell_center_1d(i) for i in range(self.ndim)]
        res = np.meshgrid(*xx, indexing='ij')
        return res

    def cell_center_by_dim(self, i):
        # TODO Only create meshgrid for one component.
        return self.cell_center_all()[i]

    def cell_center_by_name(self, name):
        '''
        Returns cell centers in direction `name` (e.g. 'x').
        '''
        if name in self.varnames:
            i = self.varnames.index(name)
            return self.cell_center_by_dim(i)
        return None

    def cell_node_1d(self, i):
        x = np.linspace(self.lower[i],
                        self.upper[i],
                        self.shape[i] + 1,
                        dtype=self.dtype)
        return x

    def cell_node_all(self):
        xx = [self.cell_node_1d(i) for i in range(self.ndim)]
        res = np.meshgrid(*xx, indexing='ij')
        return res

    def cell_node_by_dim(self, i):
        # TODO Only create meshgrid for one component.
        return self.cell_node_all()[i]

    def cell_node_by_name(self, name):
        '''
        Returns cell nodes in direction `name` (e.g. 'x').
        '''
        if name in self.varnames:
            i = self.varnames.index(name)
            return self.cell_node_by_dim(i)
        return None

    def cell_index_all(self):
        return np.meshgrid(*[np.arange(s) for s in self.shape], indexing='ij')

    def cell_index_by_dim(self, i):
        # TODO Only create meshgrid for one component.
        return self.cell_index_all()[i]

    def cell_index_by_name(self, name):
        '''
        Returns cell index in direction `name` (e.g. 'ix').
        '''
        # TODO Only create meshgrid for one component.
        names = ['i' + n for n in self.varnames]
        if name in names:
            i = names.index(name)
            return self.cell_index_by_dim(i)
        return None

    def node_index_all(self):
        return np.meshgrid(*[np.arange(s + 1) for s in self.shape],
                           indexing='ij')

    def node_index_by_dim(self, i):
        # TODO Only create meshgrid for one component.
        return self.node_index_all()[i]

    # Aliases consistent with Context.
    def step(self, i):
        if isinstance(i, str):
            i = self.varnames.index(i)
        return self.step_by_dim(i)

    def cell_index(self, i):
        if isinstance(i, str):
            i = self.varnames.index(i)
        return self.cell_index_by_dim(i)

    def node_index(self, i):
        if isinstance(i, str):
            i = self.varnames.index(i)
        return self.node_index_by_dim(i)

    def size(self, i):
        if isinstance(i, str):
            i = self.varnames.index(i)
        return self.shape[i]

    def custom_by_name(self, name):
        if name == "zeros":
            return tf.zeros(self.shape, dtype=self.dtype)
        if name == "ones":
            return tf.ones(self.shape, dtype=self.dtype)
        return None

    def random_inner(self, size):
        res = latin_hypercube(self.ndim, size).T
        for i in range(self.ndim):
            res[i] = self.lower[i] + (self.upper[i] - self.lower[i]) * res[i]
        res = [p for p in res]
        return res

    def random_boundary(self, normal, side, size):
        '''
        Returns random points from boundary (domain face).
        normal: `int`
            Direction of normal to boundary, [0, ndim)
        side: `int`
            Side of boundary (0 or 1).
        size: `int`
            Number of samples.
        '''
        assert normal < self.ndim
        assert side == 0 or side == 1
        res = latin_hypercube(self.ndim - 1, size).T
        const = np.ones(size) * side
        res = np.vstack((res[:normal], const, res[normal:]))
        for i in range(self.ndim):
            res[i] = self.lower[i] + (self.upper[i] - self.lower[i]) * res[i]
        res = [p for p in res]
        return res

    def assign_active_state(self, dst, src):
        for name in self.fieldnames:
            dst.fields[name].assign(src.fields[name])
        for name in self.aweights:
            for i in range(len(src.weights[name])):
                dst.weights[name][i].assign(src.weights[name][i])

    def multigrid_to_field(self,
                           uw,
                           mod=tf,
                           factors=None,
                           use_axes=None,
                           method=None,
                           cell=None):
        '''
        Converts multigrid components to field.
        uw: `ndarray`
            Multigrid components on levels `nnw`.
        '''
        factors = self.mg_factors if factors is None else factors
        use_axes = self.mg_axes if use_axes is None else use_axes
        method = self.mg_interp if method is None else method
        cell = self.mg_cell if cell is None else cell
        u = MultigridDecomp.weights_to_field(uw,
                                             self.mg_nnw,
                                             mod,
                                             factors=factors,
                                             use_axes=use_axes,
                                             method=method,
                                             cell=cell)

        return u

    def field_to_multigrid(self, u, mod=tf, factors=None):
        '''
        Converts field to multigrid components.
        u: `ndarray`
            Field on the fine grid `self.mg_nnw[0]`.
        '''
        factors = self.mg_factors if factors is None else factors
        uw = MultigridDecomp.field_to_weights(u,
                                              self.mg_nnw,
                                              mod=mod,
                                              factors=factors)
        return uw

    def state_to_field(self, key, state, mod=tf):
        if key in state.weights:
            uw = state.weights[key][1]
            u = self.multigrid_to_field(uw, mod=mod)
        else:
            u = state.fields[key]
        return u
    
    def add_field_to_state(self, u, key, state, mod=tf):
        if key in state.weights:
            uw = self.field_to_multigrid(u, mod=mod)
            state_uw = state.weights[key][1]
            if hasattr(state_uw, "assign_add"):
                state_uw.assign_add(uw)
            else:
                state_uw += uw
        else:
            state_u = state.fields[key]
            if hasattr(state_u, "assign_add"):
                state_u.assign_add(u)
            else:
                state_u += u


class State:

    def __init__(self):
        self.fields = dict()
        self.weights = dict()


def convert_to_tf_variable(domain, state):
    # Convert to tf.Variable if needed.
    for name in state.fields:
        if not isinstance(state.fields[name], tf.Variable):
            state.fields[name] = tf.Variable(state.fields[name],
                                             dtype=domain.dtype)
    for name in domain.aweights:
        if len(state.weights[name]) and \
                not isinstance(state.weights[name][0], tf.Variable):
            state.weights[name] = [
                tf.Variable(w, dtype=domain.dtype) for w in state.weights[name]
            ]


def eval_neural_net(wb, *uu):
    '''
    Evaluates a fully connected neural network.
    wb: `list`
        List of weights (rank 2) and biases (rank 1) interleaved.
    uu: `list`
        Input arrays. All arrays must have the same shape.
        The number of arrays `len(uu)` must coincide with the number
        of inputs of the neural network `wb[0].shape[0]`.
    Returns:
        Output array or a list of them if multiple outputs are expected.
        The output arrays have the same shape as the input arrays.
        The number of output arrays  conincides with the number of
        outputs of the neural network `wb[-2].shape[1]`.
    '''

    ni = wb[0].shape[0]  # Number of inputs.
    no = wb[-2].shape[1]  # Number of outputs.
    assert ni == len(uu), \
            "Got {:} arguments but "\
            "neural network expects {:} inputs".format(len(uu), ni)

    if ni:  # Neural network with inputs.
        shape = uu[0].shape
        n = len(wb) // 2
        uu = [tf.reshape(u, (-1, )) for u in uu]
        u = tf.transpose(tf.stack(uu, 0))
        for i in range(n):
            w = wb[2 * i]  # Weights.
            b = wb[2 * i + 1]  # Biases.
            u = tf.linalg.matmul(u, w) + b[None, :]
            if i == n - 1:
                break
            u = tf.tanh(u)
        u = tf.transpose(u)
        uu = tf.reshape(u, (no, ) + shape)
        uu = [uu[i] for i in range(no)]
        return uu
    # Neural network without inputs.
    assert len(wb) == 2 and wb[0].shape[0] == 0 and wb[1].shape != 0, \
            "Neural network without input can only have biases " \
            "but has wb={:}".format(wb)
    return wb[1]


class NeuralNet:

    def __init__(self, weights, *inputs):
        self.weights = weights
        self.inputs = [
            x if tf.is_tensor(x) else tf.constant(x) for x in inputs
        ]

    def __call__(self, *deriv):
        f = eval_neural_net(self.weights, *self.inputs)
        assert len(deriv) == 0 or len(deriv) == len(self.inputs)

        def grad(f):
            for i in range(len(deriv)):
                for _ in range(deriv[i]):
                    f = tf.gradients(f, self.inputs[i])[0]
            return f

        if not isinstance(f, list):
            f = [f]
        res = [grad(f[i]) for i in range(len(f))]
        return res


class Context:

    def __init__(self,
                 domain,
                 args,
                 field_desc,
                 state_fields,
                 state_weights,
                 epoch=0,
                 extra=None,
                 split_shifts=True):
        '''
        split_shifts: Treat gradients over shifted fields independently.
        '''
        self.domain = domain
        self.args = args
        self.field_desc = field_desc
        self.state_fields = state_fields
        self.state_weights = state_weights
        self.mgfields = dict()
        self.split_shifts = split_shifts
        self.epoch = epoch
        self.extra = extra
        self.dtype = domain.dtype

    def cast(self, value, dtype=None):
        dtype = self.dtype if dtype is None else dtype
        return tf.cast(value, dtype)

    def size(self, i):
        domain = self.domain
        if isinstance(i, str):
            i = domain.varnames.index(i)
        return domain.shape[i]

    def step(self, i):
        domain = self.domain
        if isinstance(i, str):
            i = domain.varnames.index(i)
        return domain.step_by_dim(i)

    def cell_index(self, i):
        domain = self.domain
        if isinstance(i, str):
            i = domain.varnames.index(i)
        return domain.cell_index_by_dim(i)

    def node_index(self, i):
        domain = self.domain
        if isinstance(i, str):
            i = domain.varnames.index(i)
        return domain.node_index_by_dim(i)

    def cell_center(self, i):
        domain = self.domain
        if isinstance(i, str):
            i = domain.varnames.index(i)
        return domain.cell_center_by_dim(i)

    # TODO: Add `shift` parameter to cell_index(), cell_center()

    def neural_net(self, name, *inputs, freeze=False):
        try:
            wb = self.state_weights[name]  # Weights and biases interleaved.
        except KeyError:
            raise KeyError(
                "Weights with name '{:}' not found in domain.neuralnets".
                format(name))

        return NeuralNet(wb, *inputs)

    def field(self, name, *shift, freeze=False):
        domain = self.domain
        szero = (0, ) * domain.ndim
        assert len(shift) <= domain.ndim, "Shift is too long {:}".format(shift)
        shift = shift + (0, ) * (domain.ndim - len(shift))

        # Custom fields.
        u = domain.custom_by_name(name)
        if u is not None:
            return u

        # Multigrid fields.
        if name in domain.mg_fieldnames:
            if (name, shift) in self.mgfields:
                u = self.mgfields[(name, shift)]
            else:
                if (name, szero) in self.mgfields:
                    u = self.mgfields[(name, szero)]
                else:
                    uw = self.neural_net(name)()[0]
                    u = domain.multigrid_to_field(uw, mod=tf)
                    self.mgfields[(name, szero)] = u
                if shift != szero:
                    u = tf.roll(u, np.negative(shift), range(domain.ndim))
                    self.mgfields[(name, shift)] = u
            return u

        # Regular grid fields.
        try:
            j = domain.fieldnames.index(name)
        except ValueError:
            raise ValueError("Unknown field name '" + name + "'")

        if freeze:
            return tf.roll(tf.stop_gradient(self.state_fields[name]),
                           np.negative(shift), range(domain.ndim))

        if self.split_shifts:
            # New field argument created for each shift.
            if (j, shift) in self.field_desc:
                u = self.args[self.field_desc.index((j, shift))]
            else:
                u = tf.roll(self.state_fields[name], np.negative(shift),
                            range(domain.ndim))
                self.args.append(u)
                self.field_desc.append((j, shift))
            return u
        else:
            # One field argument reused for all shifted fields.
            if (j, szero) in self.field_desc:
                uc = self.args[self.field_desc.index((j, szero))]
            else:
                uc = self.state_fields[name]
                self.args.append(uc)
                self.field_desc.append((j, szero))
            return tf.roll(uc, np.negative(shift), range(domain.ndim))


class Problem:

    def __init__(self, operator, domain, extra=None):
        '''
        operator: callable(mod, ctx)
            Discrete operator returning fields on grid.
            Each field corresponds to an equation to be solved.
            Arguments are:
                mod: module with mathematical functions (tensorflow, numpy)
                ctx: instance of Context
        domain: instance of Domain
        '''
        self.domain = domain
        self.operator = operator
        self.timer_total = Timer()
        self.timer_last = Timer()
        self.extra = extra

    def _eval_grad0(self, state_fields, state_weights, epoch):
        domain = self.domain
        args = []  # Field arguments for which gradients are computed.
        field_desc = []  # Field descriptors: (i, shift) for `fieldnames[i]`.
        wargs = [w for name in domain.aweights for w in state_weights[name]]
        with tf.GradientTape(persistent=True,
                             watch_accessed_variables=True) as tape:
            tape.watch(wargs)
            ctx = Context(domain,
                          args,
                          field_desc,
                          state_fields,
                          state_weights,
                          epoch=epoch,
                          extra=self.extra,
                          split_shifts=True)
            ff = self.operator(tf, ctx)
            if not isinstance(ff, tuple) and not isinstance(ff, list):
                ff = (ff, )
        grads = [tape.gradient(f, args) for f in ff]
        # Having experimental_use_pfor=True leads to excessive memory usage,
        # sufficient to store a dense matrix of size `prod(domain.shape)**2`.
        if len(wargs):
            wgrads = [
                tape.jacobian(ff[i], wargs, experimental_use_pfor=False)
                for i in range(len(ff))
            ]
        else:
            wgrads = [[] for _ in range(len(ff))]
        return ff, grads, field_desc, wgrads

    # XXX _eval_grad() must take members of State explicitly.
    #     Otherwise, the operator is not computed correctly (frozen).
    @tf.function(jit_compile=False)
    def _eval_grad(self, state_fields, state_weights, epoch):
        return self._eval_grad0(state_fields, state_weights, epoch)

    class Raw:

        def __init__(self, value):
            self.value = value

    @tf.function(jit_compile=False)
    def _eval_loss(self, state_fields, state_weights, epoch):
        domain = self.domain
        args = []  # Field arguments for which gradients are computed.
        field_desc = []  # Index in `fieldnames` of the field argument.
        wargs = [state_weights[name] for name in domain.aweights]
        with tf.GradientTape(persistent=True,
                             watch_accessed_variables=True) as tape:
            ctx = Context(domain,
                          args,
                          field_desc,
                          state_fields,
                          state_weights,
                          epoch=epoch,
                          extra=self.extra,
                          split_shifts=False)
            ff = self.operator(tf, ctx)
            if not isinstance(ff, tuple) and not isinstance(ff, list):
                ff = (ff, )
            loss_split = [
                tf.reduce_mean(f.value)
                if isinstance(f, Problem.Raw) else tf.reduce_mean(tf.square(f))
                for f in ff
            ]
            loss = sum(loss_split)
        grads = tape.gradient(loss, args)
        wgrads = tape.gradient(loss, wargs)
        return loss, grads, field_desc, wgrads, loss_split

    @tf.function(jit_compile=False)
    def _eval_diag_tf(self, state_fields, state_weights, epoch):
        domain = self.domain
        args = []  # Field arguments for which gradients are computed.
        field_desc = []  # Index in `fieldnames` of the field argument.
        wargs = [state_weights[name] for name in domain.aweights]
        with tf.GradientTape(persistent=True,
                             watch_accessed_variables=True) as tape:
            ctx = Context(domain,
                          args,
                          field_desc,
                          state_fields,
                          state_weights,
                          epoch=epoch,
                          extra=None,
                          split_shifts=False)
            ff = self.operator(tf, ctx)
            if not isinstance(ff, tuple) and not isinstance(ff, list):
                ff = (ff, )
            lossn = sum([
                f.value if isinstance(f, Problem.Raw) else tf.reduce_sum(
                    f * tf.random.normal(f.shape, dtype=domain.dtype)) /
                (tf.cast(tf.size(f), domain.dtype)**0.5) for f in ff
            ])
        grads = tape.gradient(lossn, args)
        wgrads = tape.gradient(lossn, wargs)

        res = dict()
        for g, (i, shift) in zip(grads, field_desc):
            assert shift == (0, ) * domain.ndim
            res[domain.fieldnames[i]] = g
        for name in domain.fieldnames:
            if name not in res:
                g[name] = tf.zeros(domain.shape, dtype=domain.dtype)

        resw = dict()
        for i, name in enumerate(domain.aweights):
            resw[name] = wgrads[i]

        res = self.pack_fields(res)
        resw = self.pack_weights(resw)
        res = tf.concat([res, resw], axis=-1)
        return res
    
    @tf.function
    def _eval_loss_terms(self, state_fields, state_weights, epoch):
        '''
        Returns individual terms of the loss function.
        '''
        domain = self.domain
        args = []  # Field arguments for which gradients are computed.
        field_desc = []  # Index in `fieldnames` of the field argument.
        wargs = [state_weights[name] for name in domain.aweights]
        with tf.GradientTape(persistent=True,
                             watch_accessed_variables=True) as tape:
            ctx = Context(domain,
                          args,
                          field_desc,
                          state_fields,
                          state_weights,
                          epoch=epoch,
                          extra=self.extra,
                          split_shifts=False)
            ff = self.operator(tf, ctx)
            if not isinstance(ff, tuple) and not isinstance(ff, list):
                ff = (ff, )
            loss_terms = [tf.reduce_mean(tf.square(f)) for f in ff]
        return loss_terms
    
    def eval_loss_terms(self, state, epoch=0):
        domain = self.domain
        self.init_missing(state)
        timer = Timer()
        epoch = tf.constant(epoch, dtype=domain.dtype)
        timer.push("eval_grad")
        loss_terms = self._eval_loss_terms(state.fields, state.weights, epoch)
        timer.pop("eval_grad")
        self.timer_total.append(timer)
        self.timer_last = timer
        return [v.numpy() for v in loss_terms]

    def eval_diag(self, state, epoch=0, nsmp=10):
        domain = self.domain
        self.init_missing(state)
        epoch = tf.constant(epoch, dtype=domain.dtype)
        gsum = None
        for smp in range(nsmp):
            g = self._eval_diag_tf(state.fields, state.weights, epoch)
            if smp == 0:
                gsum = tf.square(g)
            else:
                gsum += tf.square(g)

        return gsum / nsmp

    def pack_fields(self, fields):
        res = []
        for name in self.domain.fieldnames:
            res.append(tf.reshape(fields[name], [-1]))
        if len(res):
            res = tf.concat(res, axis=-1)
        return res

    def unpack_fields(self, packed):
        fields = dict()
        shape = self.domain.shape
        i = 0
        N = np.prod(shape)
        for name in self.domain.fieldnames:
            fields[name] = tf.reshape(packed[i:i + N], shape)
            i += N
        return fields

    def pack_weights(self, weights, with_frozen=False):
        res = []
        domain = self.domain
        names = domain.neuralnets if with_frozen else domain.aweights
        for name in names:
            ww = weights[name]
            for i, s in enumerate(domain.weight_shapes[name]):
                if ww[i] is not None:
                    w = ww[i]
                else:
                    w = tf.zeros(s, dtype=domain.dtype)
                res.append(tf.reshape(w, [-1]))
        if len(res):
            res = tf.concat(res, axis=-1)
        return res

    def unpack_weights(self, packed, with_frozen=False):
        weights = dict()
        shape = self.domain.shape
        i = 0
        names = self.domain.neuralnets if with_frozen else self.domain.aweights
        for name in names:
            layers = self.domain.neuralnets[name]
            ww = []
            for ni, no in zip(layers[:-1], layers[1:]):
                # Weights.
                ww.append(tf.reshape(packed[i:i + ni * no], (ni, no)))
                i += ni * no
                # Biases.
                ww.append(packed[i:i + no])
                i += no
            weights[name] = ww
        return weights

    def pack_state(self, state, with_frozen=False):
        packed = tf.concat(
            [
                self.pack_fields(state.fields),
                self.pack_weights(state.weights, with_frozen),
            ],
            axis=-1,
        )
        return packed

    def unpack_state(self, packed, with_frozen=False):
        state = State()
        i = 0
        n = self.domain.fields_size()
        state.fields = self.unpack_fields(packed[i:i + n])
        i += n
        state.weights = self.unpack_weights(packed[i:], with_frozen)
        return state

    def init_missing(self, state):
        domain = self.domain
        for name in domain.fieldnames:
            if name not in state.fields:
                # Initialize fields with zeros by defult.
                state.fields[name] = tf.zeros(domain.shape, dtype=domain.dtype)
        for name, layers in domain.neuralnets.items():
            assert layers[-1], \
                "Neural network '{}' must have output, " \
                "but has layers={:}".format(name, layers)
            if name not in state.weights:
                ww = []
                for ni, no in zip(layers[:-1], layers[1:]):
                    # Weights
                    if ni:
                        scale = 1 / math.sqrt(ni)
                        w = tf.random.uniform((ni, no),
                                              -scale,
                                              scale,
                                              dtype=domain.dtype)
                    else:
                        w = tf.zeros((ni, no), dtype=domain.dtype)
                    ww.append(w)
                    # Biases
                    ww.append(tf.zeros(no, dtype=domain.dtype))
                state.weights[name] = ww
            else:
                ww = state.weights[name]
                if not isinstance(ww, list):
                    raise TypeError(
                        "Expected a list of arrays for "
                        "weights of neural network '{}'".format(name))
                for i in range(len(ww)):
                    assert len(ww) == len(domain.weight_shapes[name]),\
                            "Invalid number of weights, " \
                            "expected {:}, got {:}".format(
                                len(domain.weight_shapes[name]), hlen(ww))
                    if not tf.is_tensor(ww[i]):
                        ww[i] = np.array(ww[i], dtype=domain.dtype).reshape(
                            domain.weight_shapes[name][i])

        convert_to_tf_variable(domain, state)

    def linearize(self, state, epoch=0, mod=np, modsp=scipy.sparse):
        domain = self.domain
        self.init_missing(state)

        timer = Timer()

        timer.push("eval_grad")
        epoch = tf.constant(epoch, dtype=domain.dtype)
        const, grads, field_desc, wgrads = self._eval_grad(
            state.fields, state.weights, epoch)
        timer.pop("eval_grad")

        Nu = len(domain.fieldnames)  # Number of unknown fields.
        N = np.prod(domain.shape)  # Number of cells.
        # Total number of scalar equations.
        Ne = sum(np.prod(eq.shape, dtype=int) for eq in const)
        mshape = (Ne, Nu * N)  # Shape of sparse matrix from fields.

        timer.push("sparse_fields")
        linear = modsp.csr_matrix(mshape, dtype=domain.dtype)
        row = mod.arange(N)
        col_diag = mod.reshape(row, domain.shape)
        for i in range(len(const)):  # Loop over equations.
            for j in range(len(field_desc)):  # Loop over grid field variables.
                index, shift = field_desc[j]
                index = index.numpy()
                col = mod.roll(col_diag,
                               shift=np.negative(shift),
                               axis=range(domain.ndim)).flatten()
                g = grads[i][j]
                if g is not None:
                    g = mod.array(g.numpy().flatten())
                    a = modsp.csr_matrix((g, (row + i * N, col + index * N)),
                                         dtype=domain.dtype,
                                         shape=mshape)
                    linear += a

        timer.pop()

        timer.push("sparse_weights")
        Nw = domain.aweights_size()  # Size of weights of neural networks.
        mmw = []  # Matrices with gradients for each equation.
        for i in range(len(const)):  # Loop over equations.
            # Shape of equation.
            # Equals domain.shape for fields and arbitrary for neural networks.
            eqshape = tuple(const[i].shape)
            Nei = np.prod(eqshape, dtype=int)  # Number of scalar equations.
            ww = wgrads[i]
            wshapes = [
                s for name in domain.aweights
                for s in domain.weight_shapes[name]
            ]
            assert len(ww) == len(wshapes)
            # Replace None with zeros.
            ww = [
                tf.zeros(eqshape + s, dtype=domain.dtype) if w is None else w
                for w, s in zip(ww, wshapes)
            ]
            ww = [tf.reshape(w, eqshape + (-1, )) for w in ww]
            if Nw:
                mw = tf.concat(ww, axis=len(eqshape))
                mw = tf.reshape(mw, (Nei, -1))
                # Dense matrix with gradients.
                mw = modsp.csr_matrix(mw, shape=(Nei, Nw))
            else:
                # Empty matrix.
                mw = modsp.csr_matrix((Nei, Nw), dtype=domain.dtype)
            mmw.append(mw)
        mmw = modsp.vstack(mmw)
        # Append gradients from weights to gradients from fields.
        linear = modsp.hstack([linear, mmw])
        timer.pop()

        const = [mod.array(eq.numpy()) for eq in const]

        self.timer_total.append(timer)
        self.timer_last = timer

        return const, linear

    def eval_loss_grad(self, state, epoch=0):
        domain = self.domain
        self.init_missing(state)

        timer = Timer()

        epoch = tf.constant(epoch, dtype=domain.dtype)
        timer.push("eval_grad")
        loss, raw_grads, field_desc, raw_wgrads, loss_split = self._eval_loss(
            state.fields, state.weights, epoch)
        timer.pop("eval_grad")

        grads = dict()
        for g, (i, shift) in zip(raw_grads, field_desc):
            assert shift == (0, ) * domain.ndim
            grads[domain.fieldnames[i]] = g
        for name in domain.fieldnames:
            if name not in grads:
                grads[name] = tf.zeros(domain.shape, dtype=domain.dtype)

        wgrads = dict()
        for i, name in enumerate(domain.aweights):
            wgrads[name] = raw_wgrads[i]

        self.timer_total.append(timer)
        self.timer_last = timer

        return loss, grads, wgrads, loss_split


def checkpoint_save(state, path):
    s = dict()
    fields = dict()
    for k in state.fields:
        fields[k] = np.array(state.fields[k])
    s['fields'] = fields

    weights = dict()
    for k in state.weights:
        weights[k] = [np.array(w) for w in state.weights[k]]
    s['weights'] = weights

    with open(path, 'wb') as f:
        pickle.dump(s, f)


def checkpoint_load(state,
                    path,
                    fields_to_load=None,
                    weights_to_load=None,
                    skip_missing=True):
    with open(path, 'rb') as f:
        s = pickle.load(f)
    fields = s.get('fields', None)
    if fields is not None:
        if fields_to_load is None:
            fields_to_load = fields.keys()
        for k in fields_to_load:
            if not skip_missing and k not in fields:
                raise RuntimeError("Missing field {}".format(k))
            state.fields[k] = fields[k]
    weights = s.get('weights', None)
    if weights is not None:
        if weights_to_load is None:
            weights_to_load = weights.keys()
        for k in weights_to_load:
            if not skip_missing and k not in weights:
                raise RuntimeError("Missing weights {}".format(k))
            state.weights[k] = weights[k]


def extrap_quadh(u0, u1, u1p):
    '''
    Quadratic extrapolation from points 0, 1, 1.5 to point 2.
    Suffix `h` means half.
    '''
    u2 = (u0 - 6 * u1 + 8 * u1p) / 3
    return u2


def extrap_quad(u0, u1, u2):
    'Quadratic extrapolation from points 0, 1, 2 to point 3.'
    u3 = u0 - 3 * u1 + 3 * u2
    return u3


def extrap_linear(u0, u1):
    'Linear extrapolation from points 0, 1 to point 2.'
    u2 = 2 * u1 - u0
    return u2


def operator_laplace(mod, ctx):
    dx = ctx.step('x')
    dy = ctx.step('y')
    ix = ctx.cell_index('x')
    iy = ctx.cell_index('y')
    zeros = ctx.field('zeros')
    nx = ctx.size('x')
    ny = ctx.size('y')

    def stencil_var(key, freeze=False):
        'Returns: q, qxm, qxp, qym, qyp'
        st = [
            ctx.field(key, freeze=freeze),
            ctx.field(key, -1, 0, freeze=freeze),
            ctx.field(key, 1, 0, freeze=freeze),
            ctx.field(key, 0, -1, freeze=freeze),
            ctx.field(key, 0, 1, freeze=freeze)
        ]
        return st

    def laplace(st):
        q, qxm, qxp, qym, qyp = st
        q_xx = (qxp - 2 * q + qxm) / dx**2
        q_yy = (qyp - 2 * q + qym) / dy**2
        q_lap = q_xx + q_yy
        return q_lap

    def apply_bc_extrap(st):
        'Linear extrapolation from inner cells to halo cells.'
        st[1] = mod.where(ix == 0, extrap_linear(st[2], st[0]), st[1])
        st[2] = mod.where(ix == nx - 1, extrap_linear(st[1], st[0]), st[2])
        st[3] = mod.where(iy == 0, extrap_linear(st[4], st[0]), st[3])
        st[4] = mod.where(iy == ny - 1, extrap_linear(st[3], st[0]), st[4])
        return st

    u_st = stencil_var('u')
    v_st = stencil_var('v')
    apply_bc_extrap(u_st)
    apply_bc_extrap(v_st)
    u_lap = laplace(u_st)
    v_lap = laplace(v_st)
    return u_lap, v_lap, zeros


def latin_hypercube(ndim, size):
    '''
    Returns `size` points from the unit cube.
    '''
    # XXX Copied from pyDOE.
    cut = np.linspace(0, 1, size + 1)
    u = np.random.rand(size, ndim)
    a = cut[:size]
    b = cut[1:size + 1]
    rdpoints = np.zeros_like(u)
    for j in range(ndim):
        rdpoints[:, j] = u[:, j] * (b - a) + a
    H = np.zeros_like(rdpoints)
    for j in range(ndim):
        order = np.random.permutation(range(size))
        H[:, j] = rdpoints[order, j]
    return H
