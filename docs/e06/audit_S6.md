# Audit: disputed-pair band and S6 (Sep 26, ~22:40 IST)

Leaderboard: E01 0.953 · E02 0.964 · S3 0.956 · S4 (E02 ∩ S3) 0.969 · **SAMPLE05 0.971** · S5 (S4 ∩ E04) not submitted.

## 1. Inputs inspected (before any processing)
| file | columns | rows | notes |
|---|---|---|---|
| sampleoutput05/matching_results.tsv | source1_entity_id, matched_entity_ids | 1,732,544 | ID lists only: **no raw fields, no per-model scores** |
| sampleoutput05/candidate_pairs.tsv | source1_entity_id, candidate_entity_ids | 1,732,544 | ID lists only (22.4 candidates/S1) |
Join key: `source1_entity_id` (+ exploded S2/S3 IDs). Raw fields come from the competition test files. **"E06" votes can't be recovered
from these files.** The disputed set was therefore rebuilt as S4 minus SAMPLE (valid because SAMPLE ⊂ S4, verified).

## 2. Claimed vs recomputed
| item | claimed | recomputed from files | status |
|---|---|---|---|
| pairs removed S4 → SAMPLE | 66,330 | **66,330** | ✅ |
| pairs added | 0 | **0** (SAMPLE is 100% inside S4) | ✅ |
| per-country split of removed | not given | France 24,154 · India 21,390 · US 20,786 | measured |
| false-merge share f of removed set | ≈0.43 | **0.437** (LB 0.969 → 0.971, gain +0.20/FP, cost −0.0625/TP) | ✅ |
| true matches wrongly removed | ~57% (~38k) | ~56% (~37,300) | ✅ approx |
| "train true-pair mismatch US 0.112 / India 0.206" | (assumed) | **US 10.8% / India 20.6%**, recomputed from train GT | minor discrepancy (US) |

## 3. Method (no hand-labelled test data)
Hand-labelling test pairs conflicts with the fair-play rule ("only the provided training data"). The same analysis was run on
**train with exact labels**: universe = pairs kept by E02 ∩ S3 (OOF) = 7,223,702 pairs, 19,125 false (0.265%). Disputed band =
kept by E02 ∩ S3 but rejected by E04 = 28,737 pairs, 16.5% false. The review CSV of 300 test disputed pairs (stratified by
country, raw fields side by side, blank `my_label`) was still produced for inspection: `disputed_review_sample.csv`.

## 4. What separates true matches from false merges (chart: `feature_separation.png`)
| feature | AUC, all kept pairs | AUC, disputed band |
|---|---|---|
| E04 score | **0.972** | 0.612 |
| E02 score | **0.960** | 0.593 |
| address token-set sim | 0.690 | 0.548 |
| house number exact | 0.673 | 0.514 |
| house-number edit distance | 0.543 | 0.551 |
| candidate address missing | 0.576 | 0.544 |
| legal-form-only difference | 0.538 | 0.533 |
| name similarity / exact core name | 0.58 / 0.56 | 0.50 / 0.51 |
**Finding:** inside the disputed band no label-free feature separates true from false (AUC ≈ 0.50–0.55), so **a rescue rule can't be
built from these fields**. Postal code: **not available**. The data has no separate postcode field and addresses rarely contain one (open item).
Legal-form tokens came from the pipeline's own normalization code (repo file), not an external list.

## 5. Rules and their estimated impact (chart: `policy_comparison.png`)
Logistic regression P(false) on train (AUC 0.90; coefficients: E04 −7.0, E02 −1.9, candidate address missing +2.35,
legal-form-only diff +1.25, exact core name −1.08, exact house number −0.58). Test calibration: model mean P(false) on the
66,330 SAMPLE-removed pairs vs their LB-implied 0.437 → multiplier **m = 2.80**. Remove when m·P(false) > 0.238 (break-even).
| policy | pairs changed | est. LB | decision |
|---|---|---|---|
| SAMPLE05 (discard all disputed) | – | 0.971 (actual) | baseline |
| **remove-only rule** | −7,657 (mismatch FR 3.6% / IN 38.4% / US 14.7%) | **≈0.9714** | **submitted as S6** |
| rescue rule (m·P(false) < 0.05) | +25,421 | 0.9718 on paper | **rejected**: rescued pairs have 57.6% / 40.3% / 35.6% house-number mismatch (true-like 1.8% / 20.1% / 10.4%); fails the "added pairs look like true pairs" go/no-go; the uniform multiplier assumption is violated |
**Ceiling note:** even a perfect rescue of all ~37,300 true pairs in the disputed set adds at most +0.0013 under F0.5.

## 6. S6 = SAMPLE05 − 7,657 pairs
Checks: matches ⊆ candidates per S1 (hard fail) ✅; official validator `--check-ids` ✅; independent check ✅ (1,732,544 rows,
5,662,486 pairs, 5.94% empty, no S2/S3 given to two S1). **Expected LB ≈ 0.971–0.972** (gain within noise).

## 7. Open items / not verifiable from the files
1. E06's votes/scores (the sample's third model) aren't in the files, so its decision logic can't be audited.
2. The calibration multiplier comes from **one** leaderboard point and is assumed uniform. The rescue analysis shows it's not uniform (yellow flag).
3. No postcode field exists; the postcode-match diagnostic couldn't be computed.
4. House-number mismatch is a strong proxy, not ground truth. Only the leaderboard can confirm the S6 delta.
5. Reaching 0.98 isn't supported by any evidence available tonight. The largest remaining lever (test-density precision) needs a new model generation.
