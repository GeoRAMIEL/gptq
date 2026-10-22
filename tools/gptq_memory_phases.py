# Breaks down VRAM use of the repo's GPTQ pipeline (opt.opt_sequential) by phase for one decoder
# layer: after capturing the calibration inputs, during Hessian accumulation, and around each
# GPTQ.fasterquant call (peaks are reset before each call).
#
#   .venv/bin/python tools/gptq_memory_phases.py 128 [--model facebook/opt-6.7b]
import argparse
import builtins
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import gptq
import opt
from datautils import get_loaders

parser = argparse.ArgumentParser()
parser.add_argument('nsamples', type=int)
parser.add_argument('--model', default='facebook/opt-6.7b')
cli = parser.parse_args()

GiB = 2**30
_print = builtins.print


def mem():
    return f'alloc {torch.cuda.memory_allocated()/GiB:5.2f} GiB, peak {torch.cuda.max_memory_allocated()/GiB:5.2f} GiB'


orig_fasterquant, orig_add_batch = gptq.GPTQ.fasterquant, gptq.GPTQ.add_batch


def fasterquant(self, **kwargs):
    torch.cuda.reset_peak_memory_stats()
    _print(f'  before fasterquant {self.rows}x{self.columns}: {mem()}')
    orig_fasterquant(self, **kwargs)
    _print(f'  after  fasterquant {self.rows}x{self.columns}: {mem()}')


seen = set()


def add_batch(self, inp, out):
    orig_add_batch(self, inp, out)
    if self.columns not in seen:
        seen.add(self.columns)
        _print(f'  first add_batch, {self.columns} input columns: {mem()}')


def quiet_print(*a, **k):
    # opt_sequential prints a lot; keep only the phase marker.
    if a and a[0] == 'Ready.':
        _print(f'  after capturing calibration inputs: {mem()}')


gptq.GPTQ.fasterquant = fasterquant
gptq.GPTQ.add_batch = add_batch

model = opt.get_opt(cli.model)
model.eval()
model.model.decoder.layers = model.model.decoder.layers[:1]
data, _ = get_loaders('wikitext2', nsamples=cli.nsamples, seed=0, model=cli.model, seqlen=model.seqlen)
opt.args = argparse.Namespace(
    nsamples=cli.nsamples, wbits=4, sym=False, trits=False, percdamp=.01,
    groupsize=-1, act_order=False, static_groups=False
)
builtins.print = quiet_print
opt.opt_sequential(model, data, opt.DEV)
builtins.print = _print
print(f'final: {mem()}')
