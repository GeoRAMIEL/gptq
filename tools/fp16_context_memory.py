# Peak VRAM of an OPT model in FP16, fully resident on the GPU, vs prompt length and batch size.
# Prints a fixed grid, then binary-searches the longest prompt that fits for each batch size.
#
#   .venv/bin/python tools/fp16_context_memory.py [facebook/opt-6.7b]
import sys

import torch
from transformers import OPTForCausalLM

GiB = 2**30
name = sys.argv[1] if len(sys.argv) > 1 else 'facebook/opt-6.7b'
model = OPTForCausalLM.from_pretrained(name, dtype=torch.float16).cuda().eval()
max_len = model.config.max_position_embeddings
vocab = model.config.vocab_size
torch.cuda.synchronize()
base = torch.cuda.memory_allocated()
print(f'{name}: attn_implementation={model.config._attn_implementation}, max positions {max_len}')
print(f'weights resident: {base/GiB:.2f} GiB; device total {torch.cuda.get_device_properties(0).total_memory/GiB:.2f} GiB')


def run(bs, L, full_logits=False):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ids = torch.randint(2, vocab - 1, (bs, L), device='cuda')
    try:
        with torch.no_grad():
            out = model(ids, use_cache=True, logits_to_keep=0 if full_logits else 1)
        kv = out.past_key_values
        del out
        torch.cuda.synchronize()
        kv_gib = (torch.cuda.memory_allocated() - base) / GiB
        peak = torch.cuda.max_memory_allocated() / GiB
        del kv
        return f'peak {peak:6.2f} GiB  (KV cache {kv_gib:.2f} GiB)'
    except torch.OutOfMemoryError:
        return 'OOM'
    finally:
        torch.cuda.empty_cache()


for full in (False, True):
    tag = 'eval-style (logits for all tokens)' if full else 'generation-style prefill (last-token logits)'
    print(f'\n== {tag}')
    for L in (512, 1024, max_len):
        print(f'bs=1  L={L:5d}: {run(1, L, full)}')
    for bs in (2, 4, 8):
        r = run(bs, max_len, full)
        print(f'bs={bs}  L={max_len:5d}: {r}')
        if r == 'OOM':
            break

print('\n== longest prompt that fits (64-token resolution)')
for bs in (1, 2, 4, 8):
    lo, hi = 0, max_len
    while lo < hi:
        mid = max((lo + hi + 64) // 128 * 64, lo + 64)
        if run(bs, mid) == 'OOM':
            hi = mid - 64
        else:
            lo = mid
    print(f'bs={bs}: max prompt length {lo} (total cached tokens {bs*lo}) -> {run(bs, lo) if lo else "-"}')
