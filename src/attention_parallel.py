import argparse
import torch
import torch.distributed as dist

S, H, Q, D = 3, 8, 4, 2

def attn(q, k, v):
    return torch.softmax(q @ k.transpose(-2, -1) / D**0.5, -1) @ v

def main():
    p = argparse.ArgumentParser(); p.add_argument('--mode', choices=['mha', 'gqa'], default='mha'); mode = p.parse_args().mode
    dist.init_process_group('gloo'); rank, world = dist.get_rank(), dist.get_world_size()
    kv = Q if mode == 'mha' else 2
    if Q % world or kv % world: raise ValueError('Head counts must divide evenly across ranks')
    torch.manual_seed(31); x = torch.randn(1, S, H)
    qw, kw, vw, ow = [torch.randn(n, H) for n in (H, kv*D, kv*D, H)]
    def project(w, heads): return (x @ w.T).reshape(1, S, heads, D).transpose(1, 2)
    q, k, v = project(qw, Q), project(kw, kv), project(vw, kv)
    if mode == 'gqa': k = k.repeat_interleave(Q//kv, 1); v = v.repeat_interleave(Q//kv, 1)
    reference = attn(q, k, v).transpose(1, 2).reshape(1, S, H) @ ow.T
    nq, nk = Q//world, kv//world; qs, ks = rank*nq, rank*nk
    q, k, v = project(qw[qs*D:(qs+nq)*D], nq), project(kw[ks*D:(ks+nk)*D], nk), project(vw[ks*D:(ks+nk)*D], nk)
    if mode == 'gqa': k = k.repeat_interleave(nq//nk, 1); v = v.repeat_interleave(nq//nk, 1)
    local = attn(q, k, v).transpose(1, 2).reshape(1, S, nq*D)
    shards = [torch.empty_like(local) for _ in range(world)]; dist.all_gather(shards, local)
    output = torch.cat(shards, -1) @ ow.T; torch.testing.assert_close(output, reference)
    print(f'Rank {rank}: {mode.upper()} local context {tuple(local.shape)}')
    dist.barrier()
    if rank == 0: print(f'{mode.upper()} head-sharded attention matches baseline')
    dist.destroy_process_group()

if __name__ == '__main__': main()
