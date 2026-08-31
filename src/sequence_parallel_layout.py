import torch
import torch.distributed as dist


BATCH_SIZE = 1
SEQUENCE_LENGTH = 4
HIDDEN_FEATURES = 4


def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if SEQUENCE_LENGTH % world_size != 0:
        raise ValueError("SEQUENCE_LENGTH must divide evenly across ranks")
    if HIDDEN_FEATURES % world_size != 0:
        raise ValueError("HIDDEN_FEATURES must divide evenly across ranks")

    # A deterministic activation with 4 tokens, each containing 4 hidden values.
    full_input = torch.arange(
        BATCH_SIZE * SEQUENCE_LENGTH * HIDDEN_FEATURES, dtype=torch.float32
    ).reshape(BATCH_SIZE, SEQUENCE_LENGTH, HIDDEN_FEATURES)

    # SP layout: each rank owns different token positions but every hidden value.
    local_tokens = full_input.chunk(world_size, dim=1)[rank].contiguous()

    # G: all-gather token chunks to enter the TP region.
    token_chunks = [torch.empty_like(local_tokens) for _ in range(world_size)]
    dist.all_gather(token_chunks, local_tokens)
    gathered_sequence = torch.cat(token_chunks, dim=1)

    # A row-parallel linear layer inside the TP region.
    torch.manual_seed(11)
    full_weight = torch.randn(HIDDEN_FEATURES, HIDDEN_FEATURES)
    local_hidden = gathered_sequence.chunk(world_size, dim=-1)[rank]
    local_weight = full_weight.chunk(world_size, dim=1)[rank]
    partial_output = local_hidden @ local_weight.T

    # G*: reduce-scatter sums TP contributions and restores token ownership.
    partial_token_chunks = list(partial_output.chunk(world_size, dim=1))
    local_output = torch.empty_like(partial_token_chunks[rank])
    dist.reduce_scatter(local_output, partial_token_chunks, op=dist.ReduceOp.SUM)

    baseline_output = full_input @ full_weight.T
    expected_local_output = baseline_output.chunk(world_size, dim=1)[rank]
    torch.testing.assert_close(gathered_sequence, full_input)
    torch.testing.assert_close(local_output, expected_local_output)

    print(
        f"Rank {rank}: SP tokens {tuple(local_tokens.shape)} "
        f"-> TP sequence {tuple(gathered_sequence.shape)} "
        f"-> SP output {tuple(local_output.shape)}"
    )
    if rank == 0:
        print("All-gather and reduce-scatter match the unsharded baseline")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
