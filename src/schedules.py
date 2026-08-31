from dataclasses import dataclass


@dataclass(frozen=True)
class Event:
    stage: int
    kind: str
    micro_batch: int
    virtual_chunk: int = 0


def _stage_events(stage: int, stages: int, micro_batches: int) -> list[Event]:
    warmup = min(stages - stage - 1, micro_batches)
    events = [Event(stage, "F", i) for i in range(warmup)]
    for i in range(warmup, micro_batches):
        events.extend((Event(stage, "F", i), Event(stage, "B", i - warmup)))
    events.extend(Event(stage, "B", i) for i in range(micro_batches - warmup, micro_batches))
    return events


def afab_schedule(stages: int, micro_batches: int) -> list[Event]:
    """Per-stage AFAB order: all forwards, then all backwards."""
    return [Event(stage, kind, mb) for kind in ("F", "B") for mb in range(micro_batches) for stage in range(stages)]


def one_f_one_b_schedule(stages: int, micro_batches: int) -> dict[int, list[Event]]:
    """Return each stage's warmup, alternating steady state, and drain events."""
    if stages < 1 or micro_batches < 1:
        raise ValueError("stages and micro_batches must be positive")
    return {stage: _stage_events(stage, stages, micro_batches) for stage in range(stages)}


def interleaved_schedule(stages: int, micro_batches: int, virtual_chunks: int) -> dict[int, list[Event]]:
    """Run 1F1B over virtual stages, mapping separated chunks back to devices."""
    if virtual_chunks < 1:
        raise ValueError("virtual_chunks must be positive")
    virtual = one_f_one_b_schedule(stages * virtual_chunks, micro_batches)
    result = {stage: [] for stage in range(stages)}
    for virtual_stage, events in virtual.items():
        physical_stage = virtual_stage % stages
        for event in events:
            result[physical_stage].append(
                Event(physical_stage, event.kind, event.micro_batch, virtual_stage // stages)
            )
    return result
