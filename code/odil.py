import os, sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import util_op
import util
import util_history
import linsolver
import optimizer

from multigrid import MultigridDecomp

Domain = util_op.Domain
State = util_op.State
Problem = util_op.Problem
History = util_history.History
tf = util_op.tf

from util_io import parse_raw_xmf, read_raw, write_raw_xmf, write_raw_with_xmf
