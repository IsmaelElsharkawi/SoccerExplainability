"""Sample a class-balanced batch of new clips to annotate from the UniSoccer
MatchTime test split (train_data/json/MatchTime/classification_test.json in
https://github.com/jyrao/UniSoccer).

Clips already listed in --exclude are skipped. Each class gets an equal quota
(water-filling): classes with fewer remaining clips than the quota are taken in
full, and the leftover budget is redistributed across the larger classes, so
rare classes are over-sampled relative to their natural frequency.

    python sample_annotation_clips.py \
        --upstream path/to/classification_test.json \
        --exclude train_data/json/selected_videos_for_annotations.json \
        --out train_data/json/selected_videos_for_annotations_800.json \
        --n 800 --seed 0
"""
import argparse
import collections
import json
import random

# The 13 classes used in selected_videos_for_annotations.json, minus our own
# "foul lead to penalty" split (upstream only has "penalty").
DEFAULT_CLASSES = [
    "corner", "foul with no card", "lead to corner", "substitution",
    "yellow card", "goal", "injury", "free kick", "penalty", "red card",
    "throw in", "second yellow card",
]


def water_fill(avail, n):
    """Equal per-class quota capped by availability; returns {class: k}."""
    quota = {c: 0 for c in avail}
    open_ = {c for c in avail if avail[c] > 0}
    left = min(n, sum(avail.values()))
    while left > 0 and open_:
        share = max(1, left // len(open_))
        for c in sorted(open_, key=lambda c: avail[c] - quota[c]):
            take = min(share, avail[c] - quota[c], left)
            quota[c] += take
            left -= take
            if quota[c] == avail[c]:
                open_.discard(c)
            if left == 0:
                break
    return quota


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--exclude", nargs="*",
                    default=["train_data/json/selected_videos_for_annotations.json"])
    ap.add_argument("--out", default="train_data/json/selected_videos_for_annotations_800.json")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--classes", nargs="*", default=DEFAULT_CLASSES,
                    help="classes to sample from; pass 'all' for every upstream class")
    args = ap.parse_args()

    up = json.load(open(args.upstream, encoding="utf-8"))
    done = set()
    for path in args.exclude:
        done |= {d["video"] for d in json.load(open(path, encoding="utf-8"))}

    # Drop videos that carry more than one entry upstream (ambiguous label).
    per_video = collections.Counter(d["video"] for d in up)
    pool = [d for d in up if d["video"] not in done and per_video[d["video"]] == 1]
    if args.classes != ["all"]:
        pool = [d for d in pool if d["caption"] in set(args.classes)]

    by_class = collections.defaultdict(list)
    for d in pool:
        by_class[d["caption"]].append(d)
    avail = {c: len(v) for c, v in by_class.items()}
    quota = water_fill(avail, args.n)

    rng = random.Random(args.seed)
    picked = []
    for c in sorted(by_class):
        items = sorted(by_class[c], key=lambda d: d["video"])
        picked += rng.sample(items, quota[c])

    out = []
    for d in sorted(picked, key=lambda d: (d["caption"], d["video"])):
        league, match, _ = d["video"].split("/", 2)
        out.append({**d, "league": league, "match": match})
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4, ensure_ascii=False)

    print(f"pool {len(pool)} videos -> picked {len(out)} -> {args.out}")
    for c in sorted(quota, key=lambda c: -quota[c]):
        print(f"  {c:22s} {quota[c]:4d} / {avail[c]:4d} available")


if __name__ == "__main__":
    main()
