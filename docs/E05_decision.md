# E05 / submission 5: decision record (Sep 26, 19:30 IST)

Leaderboard so far: E01 0.953 · E02 0.964 · S3 0.956 · S4 (E02 ∩ S3) 0.969.

## Built
- **Union candidate set** (Decision 2): E04 stage-0 top-20 ∪ legacy E02/S3 stage-0 top-20 (same archived model) ∪ S3's
  actual candidate_pairs.tsv. Train: 48.7M pairs (22.1/S1, recall 97.63%). Test: 37.9M pairs (21.9/S1).
  Legacy pool (34.66M) ⊇ S3's file (34.64M). The pools each model scored are documented here (C5).
- **E04**: new blocking keys (k_sorted_num, k_nosp_num), negatives weighted ×2, density/blocking-source features
  removed (cvc_*, g_same_*, raw counts, pool flags), train re-normalized with the current code.
- **E02/S3 re-scored** on the legacy pool with legacy group statistics (their training conditions).
  Verified: re-scored OOF F0.5 E02 0.9777 (orig 0.9776), S3 0.9808 (orig 0.9806). Their vote on E04-only pairs = "no".

## Train OOF (2.2M S1)
| rule | F0.5 | P | R | FP |
|---|---|---|---|---|
| E02 ∩ S3 (S4 rule) | 0.9791 | 99.74 | 94.32 | 19,125 |
| E04 alone | 0.9803 | 99.63 | 94.94 | 27,031 |
| majority of 3 | 0.9809 | 99.65 | 95.08 | 25,847 |
| E04 ∩ (E02 ∪ S3) | 0.9805 | 99.71 | 94.77 | 21,096 |
| E02 ∩ E04 | 0.9787 | 99.75 | 94.14 | 18,150 |
| E02 ∩ S3 ∩ E04 | 0.9786 | 99.80 | 94.01 | 14,374 |
E04 agrees 99.55% with S3 on train: **not decorrelated on train** (spec stop condition, reported).
Train OOF ranks rules opposite to the leaderboard (S4 rule < S3 alone on train, but 0.969 vs 0.956 on LB).

## Test (label-free) evidence
- Agreement S3→E04: train 99.6% → **test 96.4%**. E04 rejects 215k S3 test pairs with 83%/77%/59% house-number
  mismatch (US/France/India) vs the true-pair rate 11%/3%/20%. E04 rejects test-density distractors.
- Rules that add pairs vs S4 (majority, E04 alone, E04 ∩ (E02 ∪ S3), E02 ∩ E04) add pairs with 32–73% mismatch → rejected.

## Selected: S5 = S4 ∩ E04
Removes 45,844 S4 pairs (US 42% / India 48% / France 53% mismatch vs kept 10.4% / 20.1% / 2.0%), adds none.
**Yellow flag:** train OOF alone rates this about −0.0005. Justification: 3-way disagreements are 0.4% of pairs on train
but 2.8% on test, and their house-number profile matches the pairs S4 removed (~81% false). Expected LB 0.971–0.973.
All go/no-go checks passed (matches ⊆ candidates per S1, official validator --check-ids, independent check).
