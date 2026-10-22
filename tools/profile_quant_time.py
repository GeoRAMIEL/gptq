# Wall-time breakdown of the repo's GPTQ pipeline (opt.opt_sequential) on an OPT model.
#
# Splits quantization time into: capturing calibration inputs, Hessian accumulation (add_batch),
# GPTQ.fasterquant sub-stages, and the remainder (layer forward passes + CPU<->GPU moves).
# fasterquant is replaced by an instrumented copy that runs the same math as gptq.py but
# synchronizes at stage boundaries (once per 128-column block, not per column), and prints the
# same 'error' value per layer so it can be checked against an unmodified run.
#
#   .venv/bin/python tools/profile_quant_time.py facebook/opt-125m
#   .venv/bin/python tools/profile_quant_time.py facebook/opt-6.7b --layers 2
import argparse
import builtins
import os
import sys
import time
from collections import defaultdict

import torch
import torch.nn as nn
import transformers

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import gptq
import opt
from datautils import get_loaders
from quant import quantize

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('--layers', type=int, default=0, help='Only quantize the first N decoder layers (0 = all).')
parser.add_argument('--nsamples', type=int, default=128)
parser.add_argument('--wbits', type=int, default=4)
cli = parser.parse_args()

T = defaultdict(float)  # seconds per stage
COLS = defaultdict(int)  # columns processed by the column loop, per layer width


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def fasterquant(self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False):
    # Instrumented copy of gptq.GPTQ.fasterquant for the options used here
    # (no groups, no act-order, no static groups).
    assert groupsize == -1 and not actorder and not static_groups
    t0 = sync()
    W = self.layer.weight.data.clone()
    if isinstance(self.layer, transformers.Conv1D):
        W = W.t()
    W = W.float()
    if not self.quantizer.ready():
        self.quantizer.find_params(W, weight=True)
    H = self.H
    del self.H
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    Losses = torch.zeros_like(W)
    Q = torch.zeros_like(W)
    t1 = sync()
    T['fasterquant: setup (copies, find_params)'] += t1 - t0

    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(self.columns, device=self.dev)
    H[diag, diag] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    Hinv = H
    t2 = sync()
    T['fasterquant: Cholesky -> inverse -> Cholesky'] += t2 - t1

    for i1 in range(0, self.columns, blocksize):
        ta = sync()
        i2 = min(i1 + blocksize, self.columns)
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Losses1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i]
            q = quantize(
                w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
            ).flatten()
            Q1[:, i] = q
            Losses1[:, i] = (w - q) ** 2 / d ** 2
            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1
        Q[:, i1:i2] = Q1
        Losses[:, i1:i2] = Losses1 / 2
        tb = sync()
        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
        tc = sync()
        T['fasterquant: column loop (inside 128-col blocks)'] += tb - ta
        T['fasterquant: block update W[:, i2:] -= Err1 @ Hinv'] += tc - tb
    COLS[(self.rows, self.columns)] += self.columns

    t3 = sync()
    print('error', torch.sum(Losses).item())
    if isinstance(self.layer, transformers.Conv1D):
        Q = Q.t()
    self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
    T['fasterquant: write back'] += sync() - t3


orig_add_batch = gptq.GPTQ.add_batch


def add_batch(self, inp, out):
    t0 = sync()
    orig_add_batch(self, inp, out)
    T['add_batch (Hessian H += X X^T)'] += sync() - t0


gptq.GPTQ.fasterquant = fasterquant
gptq.GPTQ.add_batch = add_batch

model = opt.get_opt(cli.model)
model.eval()
nlayers_total = len(model.model.decoder.layers)
if cli.layers:
    model.model.decoder.layers = model.model.decoder.layers[:cli.layers]
nlayers = len(model.model.decoder.layers)
data, _ = get_loaders('wikitext2', nsamples=cli.nsamples, seed=0, model=cli.model, seqlen=model.seqlen)
opt.args = argparse.Namespace(
    nsamples=cli.nsamples, wbits=cli.wbits, sym=False, trits=False, percdamp=.01,
    groupsize=-1, act_order=False, static_groups=False
)

_print = builtins.print
errors = []
marks = {}


def quiet_print(*a, **k):
    if a and a[0] == 'error':
        errors.append(a[1])
    if a and a[0] == 'Ready.':
        marks['ready'] = sync()


builtins.print = quiet_print
t_start = sync()
opt.opt_sequential(model, data, opt.DEV)
t_end = sync()
builtins.print = _print

capture = marks['ready'] - t_start
total = t_end - t_start
per_layer_total = total - capture
accounted = sum(T.values())
T['remainder: layer forward passes + CPU<->GPU moves'] = per_layer_total - accounted

print(f'\n{cli.model}: {nlayers}/{nlayers_total} layers, nsamples={cli.nsamples}, wbits={cli.wbits}')
print(f'first-layer errors: {[round(e, 2) for e in errors[:6]]}')
print(f'capture calibration inputs (once): {capture:7.2f} s')
print(f'per-layer work ({nlayers} layers):      {per_layer_total:7.2f} s  = 100%')
for k, v in sorted(T.items(), key=lambda kv: -kv[1]):
    print(f'  {k:52s} {v:7.2f} s  {100*v/per_layer_total:5.1f}%')
fq = sum(v for k, v in T.items() if k.startswith('fasterquant'))
print(f'  {"(fasterquant total)":52s} {fq:7.2f} s  {100*fq/per_layer_total:5.1f}%')
cols = sum(COLS.values())
print(f'column loop: {cols} columns, {1e6*T["fasterquant: column loop (inside 128-col blocks)"]/cols:.1f} us/column')
if nlayers < nlayers_total:
    est = capture + per_layer_total * nlayers_total / nlayers
    print(f'estimated full-model quantization time: {est:.0f} s')
