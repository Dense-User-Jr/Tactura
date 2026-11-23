"""Train and cross-validate Listen's word ranker from an NLTK Brown Corpus zip.

Input: https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/brown.zip
Evaluated SHA-256: 9b275f9b3b95d7bd66ccfb7cd259f445a13bbe5d1f4107aba09fd3e8364bafa6

From the repository root:
    python3 ML/sentence_context/train.py /path/to/brown.zip \
        --model wear/src/main/assets/sentence_context.tsv

Each of ten folds trains on 450 documents and scores the other 50. Scored
positions have at least one immediately preceding, spellable word and multiple
exact matches for the intended word's first usable lexicon encoding.
The exported deployment model is subsequently fit on all 500 documents.
"""

import argparse
import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "wear/src/main/assets"
FOLDS = 10
TRIGRAM_WEIGHT = 4
MIN_COUNT = 2
# Preserve the existing demo rules. These also define the pre-training baseline.
PREFERENCES = {
    ("rent", "is"): ("due",),
    ("this", "is"): ("too",),
    ("it", "is"): ("too",),
    ("need",): ("to",),
    ("in",): ("two", "the"),
}


def read_lexicon():
    groups = defaultdict(list)
    encoding = {}
    with (ASSETS / "lexicon.csv").open(newline="", encoding="utf-8") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: -float(row["zipf"]))
    for row in rows:
        word, key = row["word"], row["gestures"]
        if int(row["rank"]) > 20_000 or not key or "6" in key:
            continue
        if word not in groups[key]:
            groups[key].append(word)
        encoding.setdefault(word, key)
    return encoding, groups


def read_sentences(path):
    """Yield Brown documents as ordered sentences of lowercase tokens."""
    with ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if re.fullmatch(r"brown/c[a-z]\d{2}", name):
                yield [
                    [token.rsplit("/", 1)[0].lower() for token in line.split()]
                    for line in archive.read(name).decode("ascii").splitlines() if line.strip()
                ]


def spellable_spans(sentence, encoding):
    """Keep real word adjacency and reset when a word cannot be entered."""
    span = []
    for word in sentence:
        if word in encoding:
            span.append(word)
        elif word in {".", "!", "?", ";", ":"} or re.search("[a-z]", word):
            if span:
                yield span
                span = []
    if span:
        yield span


def choices(span, encoding, groups):
    for index, word in enumerate(span[1:], start=1):
        key = encoding[word]
        if len(groups[key]) > 1:
            yield tuple(span[max(0, index - 2):index]), word, key


def document_events(sentences, encoding, groups):
    events = Counter(
        choice
        for sentence in sentences
        for span in spellable_spans(sentence, encoding)
        for choice in choices(span, encoding, groups)
    )
    words = sum(bool(re.search("[a-z]", word)) for sentence in sentences for word in sentence)
    return events, words


def read_documents(path, encoding, groups):
    for sentences in read_sentences(path):
        yield document_events(sentences, encoding, groups)


def fit(events):
    counts = Counter()
    for (prefix, word, _), count in events.items():
        counts[prefix[-1:], word] += count
        if len(prefix) == 2:
            counts[prefix, word] += count
    return {pair: count for pair, count in counts.items() if count >= MIN_COUNT}


def rank(prefix, candidates, counts):
    ranked = sorted(
        candidates,
        key=lambda word: -(counts.get((prefix[-1:], word), 0)
                           + (TRIGRAM_WEIGHT * counts.get((prefix, word), 0) if len(prefix) == 2 else 0)),
    )
    for length in range(len(prefix), 0, -1):
        preferred = [word for word in PREFERENCES.get(prefix[-length:], ()) if word in candidates]
        if preferred:
            return preferred + [word for word in ranked if word not in preferred]
    return ranked


def evaluate(events, groups, counts):
    results = Counter()
    for (prefix, word, key), weight in events.items():
        results["trials"] += weight
        crowded = len(groups[key]) > 3
        results["crowded_trials"] += weight * crowded
        for name, candidates in (
            ("frequency", groups[key]),
            ("rules", rank(prefix, groups[key], {})),
            ("trained", rank(prefix, groups[key], counts)),
        ):
            results[f"{name}_top1"] += weight * (candidates[0] == word)
            in_top3 = word in candidates[:3]
            results[f"{name}_top3"] += weight * in_top3
            results[f"{name}_crowded_top3"] += weight * (crowded and in_top3)
    return results


def combine(documents):
    events = Counter()
    for document, _ in documents:
        events.update(document)
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("brown_zip", type=Path)
    parser.add_argument("--model", type=Path, help="write deployment counts trained on all documents")
    args = parser.parse_args()

    encoding, groups = read_lexicon()
    documents = list(read_documents(args.brown_zip, encoding, groups))
    if len(documents) != 500:
        parser.error(f"expected 500 Brown documents, got {len(documents)}")
    random.Random(2026).shuffle(documents)
    total = Counter()
    fields = ["trials", "frequency_top1", "frequency_top3", "rules_top1", "rules_top3",
              "trained_top1", "trained_top3", "crowded_trials", "frequency_crowded_top3",
              "rules_crowded_top3", "trained_crowded_top3"]
    print("fold," + ",".join(fields))
    for fold in range(FOLDS):
        train = combine(doc for i, doc in enumerate(documents) if i % FOLDS != fold)
        test = combine(doc for i, doc in enumerate(documents) if i % FOLDS == fold)
        result = evaluate(test, groups, fit(train))
        total.update(result)
        print(f"{fold + 1}," + ",".join(str(result[field]) for field in fields))
    print("pooled," + ",".join(str(total[field]) for field in fields))
    print(f"Corpus words: {sum(words for _, words in documents)}")
    if args.model:
        counts = fit(combine(documents))
        with args.model.open("w", encoding="utf-8", newline="") as handle:
            for (prefix, word), count in sorted(counts.items()):
                handle.write(f"{' '.join(prefix)}\t{word}\t{count}\n")
        print(f"Exported {len(counts)} counts to {args.model}")


if __name__ == "__main__":
    main()
