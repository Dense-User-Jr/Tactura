# Listen mode sentence context

Listen mode uses the previous one or two selected words to rank exact gesture matches, then shows up to three choices. Tapping **Done** teaches the ranker the word transitions in the finished sentence. Later in that Listen session, previously confirmed matches rise in the list. The ranker prefers a matching two-word history, then tries a one-word history. It uses the bundled corpus ranker to break ties. A new Listen session starts with fresh memory, and the gesture classifier and prefix hints use their existing behavior.

## Ten held-out replays

Run from the repository root with Python 3 and the [Brown Corpus archive](https://www.nltk.org/data) used by the original trainer. Its checksum is recorded in [train.py](../ML/sentence_context/train.py).

```sh
python3 ML/sentence_context/replay_session_learning.py /path/to/brown.zip
```

Each fold fits the corpus ranker on 450 documents and compares it with session learning on the other 50 documents. The replay processes each held-out document in order as one simulated Listen session. It scores exact matches with at least one preceding spellable word and at least two candidate words. Words outside the watch lexicon break the context. After scoring a sentence, the replay treats its spellable spans as confirmed. No word transition from that sentence is available to the learner while it is being scored.

| Fold | Choices | First choice, corpus | First choice, session learning | Top three, corpus | Top three, session learning |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3,817 | 3,424 | 3,463 | 3,782 | 3,782 |
| 2 | 3,947 | 3,566 | 3,571 | 3,921 | 3,923 |
| 3 | 3,564 | 3,195 | 3,216 | 3,538 | 3,541 |
| 4 | 3,736 | 3,343 | 3,393 | 3,694 | 3,698 |
| 5 | 3,945 | 3,554 | 3,579 | 3,904 | 3,908 |
| 6 | 3,876 | 3,528 | 3,546 | 3,833 | 3,839 |
| 7 | 3,888 | 3,492 | 3,515 | 3,850 | 3,852 |
| 8 | 3,748 | 3,437 | 3,451 | 3,718 | 3,720 |
| 9 | 3,859 | 3,505 | 3,534 | 3,828 | 3,830 |
| 10 | 3,826 | 3,440 | 3,451 | 3,794 | 3,794 |
| **Pooled** | **38,206** | **34,484** | **34,719** | **37,862** | **37,887** |

First-choice accuracy rose from **90.26% to 90.87%**, a gain of **0.62 percentage points** from unrounded counts. Session learning corrected 339 first choices and displaced 104 previously correct ones. Top-three accuracy rose from **99.10% to 99.17%**, gaining 25 visible matches with zero losses in this replay. The order changed for 471 choices.

These runs reuse the prior Brown document splits to isolate session learning. They assume correct gestures and that the intended words in earlier sentences were successfully selected and confirmed. The results measure ranking with that feedback available. Free-form on-watch use may provide less feedback and can teach the model phrases containing selection errors.

## Test coverage

The Kotlin unit test checks that a finished sentence can promote an exact match, that a two-word history takes precedence over a one-word history, and that a fresh ranker starts without that session's memory. The Python replay records gains and losses for each fold. A full Wear build and device trial remain pending.
