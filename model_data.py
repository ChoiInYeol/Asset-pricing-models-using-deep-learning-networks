from pathlib import Path
import pandas as pd

from sklearn.preprocessing import quantile_transform
idx = pd.IndexSlice

results_path = Path('KR2_results', 'asset_pricing')
if not results_path.exists():
    results_path.mkdir(parents=True)

characteristics = ['beta', 'betasq', 'chmom', 'krwvol', 'idiovol', 'ill', 'indmom',
                   'maxret', 'mom12m', 'mom1m', 'mom36m', 'mvel', 'retvol', 'turn', 'turn_std']

with pd.HDFStore(results_path / 'autoencoder.h5') as store:
    print(store.info())

data = (pd.read_hdf(results_path / 'autoencoder.h5', 'returns')
        .stack(dropna=False)
        .to_frame('returns') # type: ignore
        .loc[idx['2006':, :], :])

with pd.HDFStore(results_path / 'autoencoder.h5') as store:
    keys = [k[1:] for k in store.keys() if k[1:].startswith('factor')]
    for key in keys:
        data[key.split('/')[-1]] = store[key].squeeze()

characteristics = data.drop('returns', axis=1).columns.tolist()
data['returns_fwd'] = data.returns.unstack('ticker').shift(-1).stack()

data.loc[:, characteristics] = (data.loc[:, characteristics]
                                .groupby(level='date')
                                .apply(lambda x: pd.DataFrame(quantile_transform(x, 
                                                                                 copy=True, 
                                                                                 n_quantiles=x.shape[0]),
                                                              columns=characteristics,
                                                              index=x.index.get_level_values('ticker')))
                               .mul(2).sub(1))
data = data.loc[idx[:'2023', :], :]
data.loc[:, ['returns', 'returns_fwd']] = data.loc[:, ['returns', 'returns_fwd']].clip(lower=-1, upper=1.0)
data = data.fillna(-2)
data.to_hdf(results_path / 'autoencoder.h5', 'model_data')