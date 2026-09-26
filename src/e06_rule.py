"""Precision rule on top of the current best submission, learned from TRAIN labels (no hand-labelled test).

Universe = pairs predicted by E02 ∩ S3 (the S4 rule) - on train via OOF votes, on test via the S4 file.
Label-free pair diagnostics + E04/E02 scores -> logistic regression (inspectable) -> P(false merge).
Test-density calibration: the leaderboard move S4 (0.969) -> SAMPLE (0.971) implies the 66,330 pairs SAMPLE removed
were f = 0.437 false; the train-fitted model's mean P(false) on exactly those pairs gives the multiplier m.
A pair is removed when m * P(false) > 0.238 (F0.5 break-even: removing an FP gains ~0.20, a TP costs ~0.0625).
"""
import os, sys, json
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cpdist
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import roc_auc_score
import pipeline as P, e05
from run_train import apply_rule
from common import load_norm, WORK

VD = os.path.join(P.CACHE, 'e05')
OUT = os.path.join(WORK, 'reports', 'e06')
os.makedirs(OUT, exist_ok=True)
COLS = ['entity_id', 'country_n', 'name_norm', 'name_core', 'addr_norm', 'addr_nums']
FEATS = ['p04', 'p02', 'num_exact', 'num_lev', 'num_missing', 'name_tsort', 'name_core_eq', 'legal_only_diff',
         'cand_addr_missing', 'addr_tset']


def diag(pairs: pl.DataFrame, split: str) -> pl.DataFrame:
    """pairs: qi, pi (+ p04, p02, label?) -> adds label-free diagnostics from the normalized records."""
    q = load_norm(split, 1).select(COLS).with_row_index('qi')
    p = pl.concat([load_norm(split, 2), load_norm(split, 3)]).select(COLS).with_row_index('pi')
    d = pairs.with_columns(pl.col('qi').cast(pl.UInt32), pl.col('pi').cast(pl.UInt32)) \
             .join(q, on='qi', how='left').join(p, on='pi', how='left', suffix='_c')
    n1 = lambda c: pl.col(c).str.split(' ').list.first().fill_null('')
    d = d.with_columns(n1('addr_nums').alias('_a'), n1('addr_nums_c').alias('_b'))
    kw = dict(workers=-1, dtype=np.float32)
    both = ((d['_a'] != '') & (d['_b'] != '')).to_numpy()
    lev = cpdist(d['_a'].to_list(), d['_b'].to_list(), scorer=Levenshtein.distance, **kw)
    return d.with_columns(
        pl.Series('num_exact', np.where(both, (d['_a'] == d['_b']).to_numpy(), np.nan).astype(np.float32)),
        pl.Series('num_lev', np.where(both, lev, np.nan).astype(np.float32)),
        pl.Series('num_missing', (~both).astype(np.float32)),
        pl.Series('name_tsort', cpdist(d['name_core'].to_list(), d['name_core_c'].to_list(), scorer=fuzz.token_sort_ratio, **kw)),
        (pl.col('name_core') == pl.col('name_core_c')).cast(pl.Float32).alias('name_core_eq'),
        ((pl.col('name_core') == pl.col('name_core_c')) & (pl.col('name_norm') != pl.col('name_norm_c'))).cast(pl.Float32).alias('legal_only_diff'),
        (pl.col('addr_norm_c') == '').cast(pl.Float32).alias('cand_addr_missing'),
        pl.Series('addr_tset', cpdist(d['addr_norm'].to_list(), d['addr_norm_c'].to_list(), scorer=fuzz.token_set_ratio, **kw)),
    )


def Xmat(d):
    X = d.select(FEATS).to_numpy().astype(np.float64)
    return np.nan_to_num(X, nan=-1.0)


if __name__ == '__main__' and sys.argv[1] == 'train':
    s04 = pl.read_parquet(os.path.join(VD, 'oof_e04.parquet')).select('qi', 'pi', 'label', pl.col('stage2_e04').alias('p04'))
    s02 = pl.read_parquet(os.path.join(VD, 'oof_E02.parquet')).select('qi', 'pi', pl.col('p').alias('p02'))
    s3 = pl.read_parquet(os.path.join(VD, 'oof_S3.parquet'))
    v02 = e05.legacy_votes(pl.read_parquet(os.path.join(VD, 'oof_E02.parquet')), 'E02')
    v3 = e05.legacy_votes(s3, 'S3')
    uni = v02.join(v3, on=['qi', 'pi'], how='semi')                        # S4 rule on train
    v04 = apply_rule(s04.rename({'p04': 'p'}), 'p', json.load(open(os.path.join(P.MODELS, 'decision_e04.json')))).select('qi', 'pi')
    d = uni.join(s04, on=['qi', 'pi'], how='left').join(s02, on=['qi', 'pi'], how='left') \
           .with_columns(pl.col('label').fill_null(0), pl.col('p04').fill_null(0.0), pl.col('p02').fill_null(0.0))
    d = d.join(v04.with_columns(pl.lit(1, pl.Int8).alias('e04_yes')), on=['qi', 'pi'], how='left').with_columns(pl.col('e04_yes').fill_null(0))
    d = diag(d, 'train')
    y_false = (d['label'] == 0).to_numpy().astype(int)
    print(f'train universe (E02∩S3 OOF): {d.height} pairs, false merges {y_false.sum()} ({y_false.mean() * 100:.3f}%)')
    band = d.filter(pl.col('e04_yes') == 0)
    yb = (band['label'] == 0).to_numpy().astype(int)
    print(f'disputed band (E02∩S3 but E04=no): {band.height} pairs, false share {yb.mean():.3f}')
    # per-feature separation (AUC of feature alone for false-vs-true), overall and in the disputed band
    sep = []
    for f in FEATS:
        x_all, x_b = np.nan_to_num(d[f].to_numpy(), nan=-1), np.nan_to_num(band[f].to_numpy(), nan=-1)
        a_all, a_b = roc_auc_score(y_false, x_all), roc_auc_score(yb, x_b) if 0 < yb.sum() < len(yb) else np.nan
        sep.append(dict(feature=f, auc_all=max(a_all, 1 - a_all), auc_band=max(a_b, 1 - a_b),
                        direction='higher=false' if a_all > 0.5 else 'higher=true'))
    sep = pl.DataFrame(sep).sort('auc_band', descending=True)
    print(sep)
    sep.write_csv(os.path.join(OUT, 'feature_separation.csv'))
    # inspectable models
    X = Xmat(d)
    lr = LogisticRegression(max_iter=300, class_weight=None, C=1.0).fit(X, y_false)
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=500).fit(X, y_false)
    print('logistic regression coefficients:', dict(zip(FEATS, np.round(lr.coef_[0], 3))), 'intercept', round(lr.intercept_[0], 3))
    print(export_text(tree, feature_names=FEATS, show_weights=True))
    pf = lr.predict_proba(X)[:, 1]
    print('train AUC of LR P(false):', round(roc_auc_score(y_false, pf), 4), ' disputed band AUC:',
          round(roc_auc_score(yb, lr.predict_proba(Xmat(band))[:, 1]), 4))
    import pickle
    pickle.dump((lr, tree), open(os.path.join(P.MODELS, 'e06_rule.pkl'), 'wb'))
    d.select(['qi', 'pi', 'label', 'e04_yes'] + FEATS).with_columns(pl.Series('pf', pf)).write_parquet(os.path.join(VD, 'train_rule_table.parquet'))
