"""Label-free test diagnostic for choosing the E05 rule (audit_consensus: the house-number profile of
changed pairs tracks truth; removed pairs should skew to mismatch, added pairs should look like true pairs:
true-pair mismatch rate in train = US 10.8%, India 20.6%)."""
import os
import polars as pl
import pipeline as P, e05
from common import load_norm

VD = os.path.join(P.CACHE, 'e05')
v04 = pl.read_parquet(os.path.join(VD, 'test_votes_E04.parquet'))
v02 = pl.read_parquet(os.path.join(VD, 'test_votes_E02.parquet'))
v3 = pl.read_parquet(os.path.join(VD, 'test_votes_S3.parquet'))
k = ['qi', 'pi']
ctry = load_norm('test', 1).select(pl.int_range(pl.len()).cast(pl.UInt32).alias('qi'), 'country_n')
feat = pl.concat([pl.read_parquet(f, columns=['qi', 'pi', 'num_first_eq']) for f in P.parts('test', 'feat')]).with_columns(pl.col('qi').cast(pl.UInt32), pl.col('pi').cast(pl.UInt32))
cast = lambda d: d.select(pl.col('qi').cast(pl.UInt32), pl.col('pi').cast(pl.UInt32))
v04, v02, v3 = cast(v04), cast(v02), cast(v3)
s4 = e05.ensemble(v04, v02, v3, 'e02_and_s3')


def prof(d):
    x = d.join(feat, on=k, how='left').join(ctry, on='qi')
    return {r[0]: (r[1], r[2]) for r in x.group_by('country_n').agg(pl.len(), ((pl.col('num_first_eq') == 0).mean() * 100).round(1)).rows()}


agree = lambda a, b: round(a.join(b, on=k, how='semi').height / max(a.height, 1), 4)
print('TEST agreement  E04->S3', agree(v04, v3), ' S3->E04', agree(v3, v04), ' E04->E02', agree(v04, v02), ' E02->S3', agree(v02, v3))
print('(train: E04->S3 0.9955, E04->E02 0.9903, E02->S3 0.9942)')
print('S4-rule reconstructed pairs:', s4.height, prof(s4))
for rule in ('e02_and_e04', 'all_three', 'e04_and_any', 'majority', 'e04'):
    r = e05.ensemble(v04, v02, v3, rule)
    add, rem = r.join(s4, on=k, how='anti'), s4.join(r, on=k, how='anti')
    print(f'{rule:12s} pairs={r.height}  ADDED {add.height} {prof(add)}  REMOVED {rem.height} {prof(rem)}', flush=True)
print('country share (pairs, mismatch%) of E04-only-vs-S3 disagreements:')
print('  S3 yes, E04 no :', prof(v3.join(v04, on=k, how='anti')))
print('  E04 yes, S3 no :', prof(v04.join(v3, on=k, how='anti')))
