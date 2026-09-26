"""Apply the train-learned precision rule to test on top of SAMPLE05 (LB 0.971), with calibration, charts,
review CSV, output files and go/no-go checks.  Usage: python e06_apply.py <out_dir>"""
import os, sys, json, pickle
import numpy as np
import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pipeline as P
from e06_rule import diag, Xmat, FEATS, VD, OUT
from run_test import write_lists
from common import load_norm, DATA

out_dir = sys.argv[1]
N = 1732544
F_REMOVED_SAMPLE = 0.437          # LB-implied false share of S4 - SAMPLE (0.969 -> 0.971), recomputed from files
BREAK_EVEN = 0.0625 / 0.2625      # removal pays when false share > 0.238
GAIN_FP, COST_TP = 0.20, 0.0625
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..')
rd = lambda f: pl.read_csv(f, separator='\t', quote_char=None, infer_schema=False).fill_null('')
q = load_norm('test', 1).select(pl.int_range(pl.len()).cast(pl.UInt32).alias('qi'), pl.col('entity_id').alias('source1_entity_id'), 'country_n')
p = pl.concat([load_norm('test', 2), load_norm('test', 3)]).select(pl.int_range(pl.len()).cast(pl.UInt32).alias('pi'), pl.col('entity_id').alias('cand_id'))
def pairs(f):
    return (rd(f).rename({'matched_entity_ids': 'cand_id'}).with_columns(pl.col('cand_id').str.split(',')).explode('cand_id')
              .filter(pl.col('cand_id') != '').join(q.select('qi', 'source1_entity_id'), on='source1_entity_id').join(p, on='cand_id').select('qi', 'pi'))
k = ['qi', 'pi']
sample = pairs(os.path.join(root, 'sampleoutput05', 'matching_results.tsv'))
s4 = pairs(os.path.join(root, 'output_S4_consensus', 'matching_results.tsv'))
R = s4.join(sample, on=k, how='anti')                    # removed by SAMPLE (66,330)
print('sample', sample.height, 's4', s4.height, 'removed-by-sample', R.height, flush=True)

sc04 = pl.read_parquet(os.path.join(VD, 'test_scores_E04.parquet')).select(pl.col('qi').cast(pl.UInt32), pl.col('pi').cast(pl.UInt32), pl.col('stage2_e04').alias('p04'))
sc02 = pl.read_parquet(os.path.join(VD, 'test_scores_E02.parquet')).select(pl.col('qi').cast(pl.UInt32), pl.col('pi').cast(pl.UInt32), pl.col('p').alias('p02'))
uni = pl.concat([sample.with_columns(pl.lit(1, pl.Int8).alias('in_sample')), R.with_columns(pl.lit(0, pl.Int8).alias('in_sample'))])
d = uni.join(sc04, on=k, how='left').join(sc02, on=k, how='left')
print('missing scores: p04', d['p04'].null_count(), 'p02', d['p02'].null_count(), flush=True)
d = diag(d.with_columns(pl.col('p04').fill_null(0.0), pl.col('p02').fill_null(0.0)), 'test')
lr, tree = pickle.load(open(os.path.join(P.MODELS, 'e06_rule.pkl'), 'rb'))
d = d.with_columns(pl.Series('pf', lr.predict_proba(Xmat(d))[:, 1]))

# ---- calibration on the one leaderboard-verified removal set
m = F_REMOVED_SAMPLE / d.filter(pl.col('in_sample') == 0)['pf'].mean()
d = d.with_columns((pl.col('pf') * m).clip(0, 1).alias('f_est'))
print(f'calibration multiplier m = {m:.2f} (mean P(false) on SAMPLE-removed set -> 0.437)', flush=True)
print('mean est. false share inside SAMPLE:', round(d.filter(pl.col('in_sample') == 1)['f_est'].mean(), 4), flush=True)

gain = lambda f: (f * GAIN_FP - (1 - f) * COST_TP)            # per-entity effect of REMOVING a pair
rem = d.filter((pl.col('in_sample') == 1) & (pl.col('f_est') > BREAK_EVEN))
res = d.filter((pl.col('in_sample') == 0) & (pl.col('f_est') < 0.05))   # rescue only near-certain true pairs
g_rem = float(gain(rem['f_est'].to_numpy()).sum() / N)
g_res = float((-gain(res['f_est'].to_numpy())).sum() / N)
print(f'rule: remove {rem.height} pairs from SAMPLE (est. gain {g_rem:+.4f}); rescue {res.height} (est. gain {g_res:+.4f})', flush=True)
final = sample.join(rem.select(k), on=k, how='anti')
final = pl.concat([final, res.select(k)]).unique()

prof = lambda df: df.join(d.select(k + ['num_exact']), on=k, how='left').join(q.select('qi', 'country_n'), on='qi') \
    .group_by('country_n').agg(pl.len(), ((pl.col('num_exact') == 0).mean() * 100).round(1).alias('num_mismatch%')).sort('country_n').rows()
print('REMOVED by rule profile:', prof(rem.select(k)), flush=True)
print('RESCUED profile       :', prof(res.select(k)), flush=True)
print('FINAL profile         :', prof(final), flush=True)

# ---- go/no-go 1: matches subset of candidates per S1
cand = pl.concat([pl.read_parquet(f, columns=k) for f in P.parts('test', 'pr')]).select(pl.col('qi').cast(pl.UInt32), pl.col('pi').cast(pl.UInt32)).unique()
bad = final.join(cand, on=k, how='anti').height
assert bad == 0, f'NO-GO: {bad} matches not in candidates'
print('GO/NO-GO 1 PASS: matches_not_in_candidates == 0', flush=True)
os.makedirs(out_dir, exist_ok=True)
write_lists(q.select('qi', 'source1_entity_id'), cand, p, 'candidate_entity_ids', os.path.join(out_dir, 'candidate_pairs.tsv'))
write_lists(q.select('qi', 'source1_entity_id'), final, p, 'matched_entity_ids', os.path.join(out_dir, 'matching_results.tsv'))

# ---- charts
sep = pl.read_csv(os.path.join(OUT, 'feature_separation.csv'))
fig, ax = plt.subplots(figsize=(9, 4.8))
y = np.arange(sep.height)
ax.barh(y - 0.2, sep['auc_all'], 0.4, label='all E02∩S3 pairs (train, labelled)', color='#4C72B0')
ax.barh(y + 0.2, sep['auc_band'], 0.4, label='disputed band only (E04 = no)', color='#DD8452')
ax.axvline(0.5, color='grey', lw=1, ls='--'); ax.set_yticks(y); ax.set_yticklabels(sep['feature']); ax.invert_yaxis()
ax.set_xlabel('separation of true match vs false merge (AUC; 0.5 = none)'); ax.set_xlim(0.45, 1.0)
ax.set_title('Which diagnostic features separate true matches from false merges'); ax.legend(loc='lower right', fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'feature_separation.png'), dpi=130); plt.close()

pol = ['SAMPLE05 (LB 0.971)', 'rescue only', 'remove only', 'remove + rescue']
est = [0.971, 0.971 + g_res, 0.971 + g_rem, 0.971 + g_rem + g_res]
fig, ax = plt.subplots(figsize=(7.5, 4))
bars = ax.bar(pol, est, color=['#8C8C8C', '#55A868', '#4C72B0', '#C44E52'])
for b, v in zip(bars, est):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.0002, f'{v:.4f}', ha='center', fontsize=9)
ax.set_ylim(0.968, max(est) + 0.002); ax.set_ylabel('estimated leaderboard F0.5')
ax.set_title('Estimated F0.5: discard-all-disputed (SAMPLE05) vs new rule\n(train-learned rule, calibrated on the 0.969→0.971 move)')
plt.xticks(fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(OUT, 'policy_comparison.png'), dpi=130); plt.close()

# ---- review CSV: 300 disputed pairs, stratified by country share, raw fields side by side
raw1 = pl.read_csv(os.path.join(DATA, 'test', 'test_source1.tsv'), separator='\t', quote_char=None, infer_schema=False).fill_null('')
raw2 = pl.concat([pl.read_csv(os.path.join(DATA, 'test', f'test_source{i}.tsv'), separator='\t', quote_char=None, infer_schema=False).fill_null('') for i in (2, 3)])
dis = d.filter(pl.col('in_sample') == 0).join(q, on='qi', how='left', suffix='_q').join(p, on='pi', how='left')
share = dis.group_by('country_n').len()
parts = [dis.filter(pl.col('country_n') == c).sample(max(1, round(300 * n / dis.height)), seed=1) for c, n in share.rows()]
rv = pl.concat(parts).join(raw1.rename({c: f's1_{c}' for c in raw1.columns}), left_on='source1_entity_id', right_on='s1_entity_id') \
       .join(raw2.rename({c: f'cand_{c}' for c in raw2.columns}), left_on='cand_id', right_on='cand_entity_id')
rv.select(['source1_entity_id', 'cand_id', 'country_n', 's1_business_name', 'cand_business_name', 's1_business_address', 'cand_business_address']
          + FEATS + ['f_est']).with_columns(pl.lit('').alias('my_label')).write_csv(os.path.join(OUT, 'disputed_review_sample.csv'))
json.dump(dict(m=m, removed=rem.height, rescued=res.height, est_gain_remove=g_rem, est_gain_rescue=g_res,
               final_pairs=final.height, removed_by_sample=R.height, sample_pairs=sample.height),
          open(os.path.join(OUT, 'e06_summary.json'), 'w'), indent=1)
print('done', flush=True)
