import argparse
from schedules import interleaved_schedule, one_f_one_b_schedule

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("1f1b", "interleaved"), default="1f1b")
args = parser.parse_args()
schedule = one_f_one_b_schedule(2, 3) if args.mode == "1f1b" else interleaved_schedule(2, 3, 2)
for stage, events in schedule.items():
    print(f"Stage {stage}: " + " ".join(f"{e.kind}{e.micro_batch}" for e in events))
