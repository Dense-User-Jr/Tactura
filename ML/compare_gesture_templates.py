"""Compare wrist templates offline using calibration CSVs and a labeled Classify round."""

import argparse
import csv
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import blueprint_reference as B
from evaluate_blueprints import Sample, percent

NEGATIVES = {"NOTHING", "STILL"}
WARP_RADIUS = B.SLIDE  # 100 ms at the reference's 50 Hz


def medoid(clips):
    """Choose the actual calibration rep closest to the others by shifted RMS."""
    totals = [0.0] * len(clips)
    for first, second in combinations(range(len(clips)), 2):
        miss = (B.distance(clips[first], clips[second])[0]
                + B.distance(clips[second], clips[first])[0]) / 2
        totals[first] += miss
        totals[second] += miss
    return clips[min(range(len(clips)), key=totals.__getitem__)]


def warped_distance(clip, template):
    """Band-limited DTW with one shared alignment across signed gyro axes."""
    axes, length = clip.shape
    costs = ((clip[:, :, None] - template[:, None, :]) ** 2).sum(axis=0)
    previous = [math.inf] * (length + 1)
    previous[0] = 0.0
    for row in range(1, length + 1):
        current = [math.inf] * (length + 1)
        for column in range(max(1, row - WARP_RADIUS), min(length, row + WARP_RADIUS) + 1):
            cost = costs[row - 1, column - 1]
            current[column] = min(
                previous[column - 1] + 2 * cost,
                previous[column] + cost,
                current[column - 1] + cost,
            )
        previous = current
    return math.sqrt(previous[length] / (2 * axes * length))


def classify_warped(clips, templates):
    if not clips:
        return None
    miss, label, index = min(
        (warped_distance(clip, template.mean), label, index)
        for label, template in templates.items()
        for index, clip in enumerate(clips)
    )
    clip = clips[index]
    template = templates[label]
    # Keep the same strength guard when alignment rescues a candidate.
    return label if B._passes(clip, template, miss) else None


def read_sample(path, truth=None):
    label, events = B.read_gyro_events(path)
    return Sample(path.name, truth or label, events)


def report(title, tests, predict):
    outcomes = Counter((sample.truth, predict(sample.clips)) for sample in tests)
    print(f"\n{title}")
    positives = correct = misses = negatives = false_activations = 0
    for truth in sorted({sample.truth for sample in tests} - NEGATIVES):
        predictions = {label: count for (actual, label), count in outcomes.items() if actual == truth}
        total = sum(predictions.values())
        right = predictions.pop(truth, 0)
        missed = predictions.pop(None, 0)
        wrong = ", ".join(f"{label} {count}" for label, count in sorted(predictions.items()))
        print(f"  {truth:12} {right:3}/{total:<3} correct, {missed:2} missed"
              + (f", called {wrong}" if wrong else ""))
        positives += total
        correct += right
        misses += missed
    for (truth, label), count in outcomes.items():
        if truth in NEGATIVES:
            negatives += count
            false_activations += count if label is not None else 0
    print(f"  Gesture accuracy {correct}/{positives} ({percent(correct, positives)}), {misses} missed")
    print(f"  No-gesture false activations {false_activations}/{negatives} "
          f"({percent(false_activations, negatives)})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path, nargs="+", help="calibration CSVs from one sitting")
    parser.add_argument("--labels", type=Path, required=True, help="answer key beside the Classify CSVs")
    args = parser.parse_args()

    calibration_paths = [path.resolve() for path in args.calibration]
    if len(set(calibration_paths)) != len(calibration_paths):
        parser.error("duplicate calibration recordings")
    reps = defaultdict(list)
    for path in calibration_paths:
        sample = read_sample(path)
        if not sample.truth or sample.truth in NEGATIVES or not sample.clips:
            parser.error(f"expected a labeled gesture with a detectable stroke in {path}")
        chosen = B.choose_clip(sample.clips, reps[sample.truth])
        reps[sample.truth].append(sample.clips[chosen])

    tests = []
    test_paths = set()
    with args.labels.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            truth = row["truth"].strip().upper()
            if not truth:
                continue
            path = (args.labels.parent / row["file"]).resolve()
            if path in calibration_paths or path in test_paths:
                parser.error(f"duplicate or overlapping evaluation recording {path}")
            test_paths.add(path)
            tests.append(read_sample(path, truth))
    if not tests:
        parser.error("answer key has no labeled evaluation recordings")
    missing = {sample.truth for sample in tests} - NEGATIVES - reps.keys()
    if missing:
        parser.error(f"no calibration for: {', '.join(sorted(missing))}")

    baseline = {label: B.build_blueprint(clips) for label, clips in reps.items()}
    means = {label: B.Blueprint(bp.mean, bp.count, bp.size) for label, bp in baseline.items()}
    medoids = {}
    for label, clips in reps.items():
        rep = medoid(clips)
        medoids[label] = B.Blueprint(rep, len(clips), B.blueprint_size(rep))

    report("Aligned mean + eigengestures (reference baseline)", tests,
           lambda clips: B.classify_clips(clips, baseline)[0])
    report("Aligned mean + shifted RMS", tests, lambda clips: B.classify_clips(clips, means)[0])
    report("Recorded medoid + shifted RMS", tests, lambda clips: B.classify_clips(clips, medoids)[0])
    report("Recorded medoid + bounded DTW", tests, lambda clips: classify_warped(clips, medoids))


if __name__ == "__main__":
    main()
