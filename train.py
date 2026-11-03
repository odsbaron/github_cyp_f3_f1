"""Train fixed F1/F3 models on supplied labelled data into a new directory."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from predict import ROOT, CYPS, TARGETS, features

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('input',type=Path); p.add_argument('output_dir',type=Path); a=p.parse_args()
    df=pd.read_csv(a.input)
    if not set(['SMILES']+TARGETS).issubset(df.columns) or df.empty: raise ValueError('Missing SMILES or target columns')
    a.output_dir.mkdir(parents=True,exist_ok=False)
    x1,x3=features(df.SMILES.tolist()); params=json.loads((ROOT/'config.json').read_text())['params']
    for name,x in [('F1',x1),('F3',x3)]:
        for c,t in zip(CYPS,TARGETS):
            y=pd.to_numeric(df[t],errors='raise').to_numpy(float); mask=np.isfinite(y)
            if not mask.any(): raise ValueError(f'No labels for {t}')
            model=LGBMRegressor(**params[name]).fit(x[mask],y[mask]); model.booster_.save_model(str(a.output_dir/f'{name}_final_{c}.txt'))
    (a.output_dir/'descriptor_names.json').write_text((ROOT/'models/descriptor_names.json').read_text())
if __name__=='__main__': main()
