"""Synthetic educational dataset for the Food Preservation Safety project.

Run: .venv313/bin/python generate_dataset.py (requires numpy and pandas).
Writes only food_preservation_dataset.csv beside this script; prints the report.
Seed: 20261004. Each row is an independent batch assessed after storage_days.

Data dictionary (CSV column order):
food_type: broad food category; milk means pasteurized milk, meat/fish raw,
    produce whole, bread baked, cooked food ready-to-eat, dairy non-cultured
    fresh/soft dairy. Fermented dairy/starter organisms are outside this scope.
temperature_c: representative constant storage temperature, degrees Celsius.
humidity_percent: surrounding relative humidity, percent (not water activity).
storage_days: elapsed storage in days, including fractional days.
ph_level: simulated batch pH at assessment, dimensionless.
packaging_type: packaging category.
storage_condition: simulated environment; refrigerated includes suboptimal 4–8 C.
microbial_count_cfu: aggregate viable spoilage-organism concentration, CFU/g
    (CFU/mL for milk), rounded integer.
preservative_used: Yes/No for a modest suitable antimicrobial intervention.
expiration_days_remaining: stated shelf life minus elapsed storage, whole days;
    positive=future, zero=today, negative=past. Rounding is to nearest day.
preservation_score: continuous 0–100 analytical index; higher is more favorable.
preserved: sampled Yes/No label for preservation at assessment.

Simulation assumptions:
All batches begin storage fresh, with variable initial contamination. Frozen
products are frozen promptly. Stated shelf life assumes correct storage
(freezer for frozen rows, room for bread, refrigeration otherwise) and is an
illustrative synthetic label. Broad categories hide subtype variation; all pH,
growth, packaging, score and probability coefficients are educational
simulation assumptions. Ambient humidity has most influence on exposed
bread/produce; sealed-food effects are attenuated.

Reference anchors for storage-condition assumptions:
https://www.fda.gov/consumers/consumer-updates/are-you-storing-food-safely
https://www.foodsafety.gov/food-safety-charts/cold-food-storage-charts
These describe cold-storage distinctions and differing food storage durations.

The score is computed before labels and shares their simulated drivers.
The primary model excludes preservation_score. Microbial count is included as
an input feature. Model metrics summarize performance on held-out simulated
records.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261004
ROWS = 200_000
FOODS = np.array(['Milk', 'Meat', 'Fish', 'Vegetables', 'Fruits', 'Bread',
                  'Cooked Food', 'Dairy Products'])
CONDITIONS = np.array(['Refrigerated', 'Frozen', 'Room Temperature', 'Poor Storage'])
PACKAGES = np.array(['Sealed', 'Vacuum Packed', 'Open', 'Plastic Container',
                     'Glass Container'])
# Food-specific baseline shelf life, pH bounds, initial log10 CFU concentration.
LIFE = np.array([7., 3., 2., 10., 12., 6., 3., 14.])
FROZEN_LIFE = np.array([60., 120., 90., 180., 180., 90., 90., 60.])
PH_LOW = np.array([6.2, 5.3, 5.8, 4.6, 2.8, 4.8, 4.8, 4.7])
PH_HIGH = np.array([6.9, 6.4, 6.8, 6.8, 4.8, 6.2, 6.8, 6.5])
INITIAL = np.array([2.1, 3.0, 3.2, 3.1, 2.8, 1.8, 2.0, 2.2])
CONDITION_PROBS = np.array([
    [.73, .08, .08, .11], [.61, .22, .06, .11], [.62, .23, .04, .11],
    [.52, .12, .25, .11], [.40, .10, .39, .11], [.12, .13, .64, .11],
    [.65, .18, .06, .11], [.73, .09, .07, .11]])
PACKAGE_PROBS = np.array([
    [.52, .01, .07, .20, .20], [.26, .37, .10, .24, .03],
    [.24, .36, .12, .25, .03], [.23, .09, .36, .28, .04],
    [.19, .04, .43, .28, .06], [.54, .04, .29, .12, .01],
    [.20, .10, .13, .43, .14], [.40, .12, .08, .24, .16]])


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


def generate():
    rng = np.random.default_rng(SEED)
    f = rng.integers(0, len(FOODS), ROWS)
    def categorical(prob):
        return (rng.random(ROWS)[:, None] > prob[f].cumsum(axis=1)).sum(axis=1)
    c, p = categorical(CONDITION_PROBS), categorical(PACKAGE_PROBS)
    frozen, poor = c == 1, c == 3
    temp = np.round(np.clip(
        rng.normal(np.array([3.8, -18., 24., 30.])[c]
                   + np.array([0, -.3, -.6, .5, .7, 1., 0, 0])[f],
                   np.array([1.6, 3., 3., 5.])[c]),
        np.array([0., -26., 18., 12.])[c], np.array([8., -8., 30., 42.])[c]), 2)
    humidity = np.round(np.clip(rng.normal(
        np.array([66., 70., 73., 83., 77., 56., 66., 65.])[f]
        + np.array([3., -12., -4., 8.])[c], 8.), 20, 95), 2)
    ph = np.round(PH_LOW[f] + (PH_HIGH[f] - PH_LOW[f]) * rng.beta(3, 3, ROWS), 3)
    preservative = rng.random(ROWS) < np.array([.03, .12, .06, .03, .05, .48, .15, .20])[f]
    # Labels depend on intended storage, not the actual abusive temperature.
    shelf = np.maximum(1, np.rint(np.where(frozen, FROZEN_LIFE[f], LIFE[f])
                                * rng.lognormal(0, .18, ROWS))).astype(int)
    days = np.round(np.maximum(.02, shelf * np.clip(rng.gamma(1.7, .50, ROWS), .003, 3.)), 2)
    expiration = np.rint(shelf - days).astype(int)
    age = days / shelf
    exposure = np.array([.22, .12, 1., .50, .40])[p]
    susceptible = np.isin(f, [3, 4, 5])
    damp = np.maximum(humidity - 70, 0) / 25 * exposure * np.where(susceptible, 1., .15)
    # Acidity inhibits generic bacterial growth; an acid-tolerant spoilage floor
    # retains yeast/mold spoilage for fruit and bread.
    acidity = .35 + .65 * sigmoid((ph - 4.8) * 1.5)
    thermal = np.clip(2.3 ** ((temp - 4) / 10), .1, 15)
    rate = 2.4 / LIFE[f] * thermal * acidity
    rate *= np.array([.85, .65, 1.2, 1., .95])[p] * np.where(preservative, .86, 1.)
    rate *= (1 + .45 * damp + .12 * poor)
    rate = np.where(frozen, 0., rate)
    initial = INITIAL[f] + rng.normal(0, .50, ROWS) + .2 * (p == 2)
    lag = rng.lognormal(-.6, .35, ROWS) / np.maximum(thermal, .2)
    growth = rate * np.maximum(days - lag, 0) * rng.lognormal(0, .24, ROWS)
    capacity = rng.uniform(8.0, 9.2, ROWS)
    log_count = np.clip(capacity - (capacity - initial) * np.exp(-growth / 5.)
                        + rng.normal(0, .16, ROWS), 0, 9.5)
    count = np.maximum(1, np.rint(10 ** log_count)).astype(np.int64)
    warm = np.maximum(temp - np.where(susceptible, 12., 4.), 0) / 12
    warm = np.where(frozen, 0., warm)
    overdue = np.maximum(-expiration, 0) / shelf
    ph_deviation = np.abs(ph - (PH_LOW[f] + PH_HIGH[f]) / 2) / (PH_HIGH[f] - PH_LOW[f])
    # Score precedes labels and is a descriptive index, not a stored probability.
    burden = (1.05 * age + .85 * np.maximum(log_count - 3., 0)
              + .65 * warm + .55 * damp + .45 * poor + .60 * overdue
              + .25 * ph_deviation + .25 * exposure - .12 * preservative)
    score = np.round(100 * sigmoid((3.0 - burden) / 1.3), 2)
    # Separate probabilistic outcome with latent batch resilience, nonlinear
    # temperature-duration interaction, and no label-based row selection.
    logit = (3.4 - 1.15 * age - 1.05 * np.maximum(log_count - 3.2, 0)
             - .65 * warm * np.minimum(days, 2.) - .65 * damp - .45 * poor
             - .65 * overdue - .20 * ph_deviation - .35 * exposure
             + .15 * preservative + rng.normal(0, .45, ROWS))
    target = np.where(rng.random(ROWS) < sigmoid(logit), 'Yes', 'No')
    return pd.DataFrame(dict(food_type=FOODS[f], temperature_c=temp,
        humidity_percent=humidity, storage_days=days, ph_level=ph,
        packaging_type=PACKAGES[p], storage_condition=CONDITIONS[c],
        microbial_count_cfu=count, preservative_used=np.where(preservative, 'Yes', 'No'),
        expiration_days_remaining=expiration, preservation_score=score, preserved=target))


def validate_and_report(df):
    assert df.shape == (ROWS, 12)
    assert not df.isna().any().any()
    assert df.duplicated().sum() == 0
    numeric = df.select_dtypes(include='number')
    assert np.isfinite(numeric.to_numpy()).all()
    for col, (low, high) in dict(temperature_c=(-26, 42), humidity_percent=(20, 95),
            storage_days=(.02, 2000), ph_level=(0, 14), microbial_count_cfu=(1, 10**10),
            expiration_days_remaining=(-2000, 2000), preservation_score=(0, 100)).items():
        assert df[col].between(low, high).all(), col
    for col, allowed in dict(food_type=FOODS, packaging_type=PACKAGES,
            storage_condition=CONDITIONS, preservative_used=['Yes', 'No'],
            preserved=['Yes', 'No']).items():
        assert set(df[col]) == set(allowed), col
    for condition, low, high in zip(CONDITIONS, [0, -26, 18, 12], [8, -8, 30, 42]):
        assert df.loc[df.storage_condition == condition, 'temperature_c'].between(low, high).all()
    for i, food in enumerate(FOODS):
        assert df.loc[df.food_type == food, 'ph_level'].between(PH_LOW[i], PH_HIGH[i]).all()
    assert (df.expiration_days_remaining + df.storage_days >= .5).all()
    share = df.preserved.eq('Yes').mean()
    assert .40 <= share <= .60, f'Class imbalance: {share:.3%}'
    assert df.microbial_count_cfu.skew() > 1
    audit = df.assign(yes=df.preserved.eq('Yes').astype(int),
                      log10_cfu=np.log10(df.microbial_count_cfu))
    assert audit.log10_cfu.corr(audit.preservation_score) < -.3
    rates = audit.groupby('storage_condition').yes.mean()
    assert rates['Frozen'] > rates['Poor Storage']
    assert rates['Refrigerated'] > rates['Poor Storage']
    print(__doc__)
    print('Dataset Shape:', df.shape, '\nColumn Names:', list(df.columns))
    print('\nData Types\n', df.dtypes, '\nMissing Values\n', df.isna().sum())
    print('\nDuplicate Count:', df.duplicated().sum())
    print('\nTarget Distribution\n', df.preserved.value_counts().to_frame('count')
          .assign(percent=df.preserved.value_counts(normalize=True) * 100))
    print('\nFood Type Distribution\n', df.food_type.value_counts())
    print('\nNumerical Statistics (including expiration)\n', numeric.describe().to_string())
    print('\nPearson Correlations\n', numeric.corr().round(3).to_string())
    print('\nSpearman Correlations\n', numeric.corr(method='spearman').round(3).to_string())
    print('\nStorage-condition sanity check\n', audit.groupby('storage_condition')
          [['yes', 'preservation_score', 'log10_cfu', 'storage_days']].mean().round(3))
    print('\nWithin food/environment duration correlations (frozen quality can age\n'
          'without microbial growth; pooled duration is confounded by long freezing):')
    print(audit.groupby(['food_type', 'storage_condition'])[
        ['storage_days', 'preservation_score']].corr().iloc[0::2, 1].round(3).to_string())
    print('\nSample Rows\n', df.sample(6, random_state=SEED).to_string(index=False))
    print('\nAll quality checks passed.')


if __name__ == '__main__':
    dataset = generate()
    validate_and_report(dataset)
    output = Path(__file__).resolve().with_name('food_preservation_dataset.csv')
    dataset.to_csv(output, index=False, lineterminator='\n')
    reloaded = pd.read_csv(output)
    pd.testing.assert_frame_equal(dataset, reloaded, check_exact=False, rtol=1e-12)
    print(f'\nSaved and verified: {output}')
