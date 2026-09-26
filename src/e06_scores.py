"""Save continuous test scores of E04 (stage-2) and E02 (stage-1) for the final precision rule."""
import os
import polars as pl
import pipeline as P, e05
VD = os.path.join(P.CACHE, 'e05')
f04, f02 = os.path.join(VD, 'test_scores_E04.parquet'), os.path.join(VD, 'test_scores_E02.parquet')
if not os.path.exists(f04):
    s1 = e05.predict_e04('test', 'stage1_e04')
    ctx1 = P.group_stats(s1.select('qi', 'pi', 'stage1_e04'), 'stage1_e04', 'p1')
    e05.predict_e04('test', 'stage2_e04', extra=ctx1).select('qi', 'pi', 'stage2_e04').write_parquet(f04)
    print('E04 scores saved', flush=True)
if not os.path.exists(f02):
    e05.score_legacy('test', 'E02').select('qi', 'pi', 'p').write_parquet(f02)
    print('E02 scores saved', flush=True)
print('done', flush=True)
