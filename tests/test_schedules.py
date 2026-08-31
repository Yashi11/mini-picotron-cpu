from src.schedules import interleaved_schedule, one_f_one_b_schedule


def test_one_f_one_b_alternates_after_warmup_and_drains():
    events = one_f_one_b_schedule(stages=3, micro_batches=4)
    assert [(e.kind, e.micro_batch) for e in events[0]] == [
        ("F", 0), ("F", 1), ("F", 2), ("B", 0), ("F", 3), ("B", 1), ("B", 2), ("B", 3)
    ]


def test_interleaving_revisits_each_physical_stage():
    events = interleaved_schedule(stages=2, micro_batches=3, virtual_chunks=2)
    assert {event.virtual_chunk for event in events[0]} == {0, 1}
    assert {event.virtual_chunk for event in events[1]} == {0, 1}
