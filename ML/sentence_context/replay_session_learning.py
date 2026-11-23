"""Run ten document-held-out replays of on-watch sentence learning."""

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from train import FOLDS, choices, combine, document_events, fit, rank, read_lexicon, read_sentences, spellable_spans


def learn(memory, span):
    for index, word in enumerate(span[1:], start=1):
        for length in range(1, min(index, 2) + 1):
            memory[tuple(span[index - length:index])][word] += 1


def rerank(prefix, baseline, memory):
    for length in range(len(prefix), 0, -1):
        counts = memory.get(prefix[-length:], {})
        if any(word in counts for word in baseline):
            return sorted(baseline, key=lambda word: -counts.get(word, 0))
    return baseline


def replay(document, encoding, groups, counts):
    memory = defaultdict(Counter)
    result = Counter()
    for sentence in document:
        spans = list(spellable_spans(sentence, encoding))
        for span in spans:
            for prefix, word, key in choices(span, encoding, groups):
                baseline = rank(prefix, groups[key], counts)
                adapted = rerank(prefix, baseline, memory)
                result["trials"] += 1
                result["changed"] += baseline != adapted
                for k in (1, 3):
                    before = word in baseline[:k]
                    after = word in adapted[:k]
                    result[f"baseline_top{k}"] += before
                    result[f"adapted_top{k}"] += after
                    result[f"top{k}_gained"] += after and not before
                    result[f"top{k}_lost"] += before and not after
        # Done becomes visible to the ranker only after this sentence is scored.
        for span in spans:
            learn(memory, span)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("brown_zip", type=Path)
    args = parser.parse_args()
    encoding, groups = read_lexicon()
    documents = list(read_sentences(args.brown_zip))
    if len(documents) != 500:
        parser.error(f"expected 500 Brown documents, got {len(documents)}")
    random.Random(2026).shuffle(documents)
    events = [document_events(doc, encoding, groups) for doc in documents]

    fields = ["trials", "changed", "baseline_top1", "adapted_top1", "top1_gained",
              "top1_lost", "baseline_top3", "adapted_top3", "top3_gained", "top3_lost"]
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["fold", *fields])
    pooled = Counter()
    for fold in range(FOLDS):
        counts = fit(combine(event for index, event in enumerate(events) if index % FOLDS != fold))
        result = Counter()
        for document in documents[fold::FOLDS]:
            result.update(replay(document, encoding, groups, counts))
        writer.writerow([fold + 1, *(result[field] for field in fields)])
        pooled.update(result)
    writer.writerow(["pooled", *(pooled[field] for field in fields)])


if __name__ == "__main__":
    main()
