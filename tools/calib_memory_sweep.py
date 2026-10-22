# Peak VRAM of the repo's GPTQ pipeline (opt.opt_sequential) vs number of calibration segments.
# All decoder layers of an OPT model have identical shapes, so only the first 2 are quantized;
# the per-layer peak equals that of a full run.
#
# Measure one size per process: memory left over from a previous run in the same process inflates
# later peaks and produces false OOMs. E.g.
#   for n in 128 192 256 288 304; do .venv/bin/python tools/calib_memory_sweep.py $n; done
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import opt
from datautils import get_loaders

parser = argparse.ArgumentParser()
parser.add_argument('nsamples', type=int)
parser.add_argument('--model', default='facebook/opt-6.7b')
parser.add_argument('--dataset', default='wikitext2', choices=['wikitext2', 'ptb', 'c4'])
parser.add_argument('--layers', type=int, default=2)
cli = parser.parse_args()

GiB = 2**30
model = opt.get_opt(cli.model)
model.eval()
model.model.decoder.layers = model.model.decoder.layers[:cli.layers]
data, _ = get_loaders(cli.dataset, nsamples=cli.nsamples, seed=0, model=cli.model, seqlen=model.seqlen)

opt.args = argparse.Namespace(
    nsamples=cli.nsamples, wbits=4, sym=False, trits=False, percdamp=.01,
    groupsize=-1, act_order=False, static_groups=False
)
torch.cuda.reset_peak_memory_stats()
tick = time.time()
try:
    opt.opt_sequential(model, data, opt.DEV)
    res = f'peak {torch.cuda.max_memory_allocated()/GiB:6.2f} GiB, {(time.time()-tick)/cli.layers:.1f} s/layer'
except torch.OutOfMemoryError:
    res = 'OOM'
print(f'RESULT nsamples={cli.nsamples:4d} ({cli.nsamples*model.seqlen:7d} tokens): {res}', flush=True)
