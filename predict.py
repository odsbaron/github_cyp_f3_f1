"""Predict CYP direct-inhibition pIC50 from a CSV of SMILES."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator, rdReducedGraphs
from rdkit.Avalon import pyAvalonTools

ROOT = Path(__file__).resolve().parent
CYPS = ('CYP1A2', 'CYP2C9', 'CYP2D6', 'CYP3A4')
TARGETS = [c + '_pIC50_direct_inhibition' for c in CYPS]

def features(smiles):
    expected = json.loads((ROOT / 'models/descriptor_names.json').read_text())
    if [name for name, _ in Descriptors._descList] != expected:
        raise ValueError('RDKit descriptor schema differs from the training environment')
    mg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    ap = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)
    def bits(fp):
        a = np.zeros(fp.GetNumBits(), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, a)
        return a
    f1, f3 = [], []
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if mol is None: raise ValueError(f'Invalid SMILES at input row {i+2}: {s!r}')
        desc = []
        for _, fn in Descriptors._descList:
            try: desc.append(float(fn(mol)))
            except Exception: desc.append(0.)
        desc = np.asarray(desc, dtype=np.float32)
        count = np.zeros(2048, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(mg.GetCountFingerprint(mol), count)
        # Match upstream feature implementation order, not the requested list order.
        a = np.concatenate([bits(mg.GetFingerprint(mol)), bits(pyAvalonTools.GetAvalonFP(mol,nBits=1024)), np.asarray(rdReducedGraphs.GetErGFingerprint(mol),dtype=np.float32), desc])
        b = np.concatenate([count, bits(MACCSkeys.GenMACCSKeys(mol)), bits(ap.GetFingerprint(mol)), desc])
        f1.append(np.nan_to_num(a, nan=0., posinf=0., neginf=0.)); f3.append(np.nan_to_num(b, nan=0., posinf=0., neginf=0.))
    return np.asarray(f1,dtype=np.float32), np.asarray(f3,dtype=np.float32)

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('input',type=Path); parser.add_argument('output',type=Path); args=parser.parse_args()
    df=pd.read_csv(args.input)
    if not {'SMILES','Molecule_Name'}.issubset(df.columns) or df.empty: raise ValueError('Input must have nonempty SMILES and Molecule_Name columns')
    if df.Molecule_Name.isna().any() or df.Molecule_Name.duplicated().any(): raise ValueError('Molecule_Name must be nonempty and unique')
    x1,x3=features(df.SMILES.tolist()); out=df[['SMILES','Molecule_Name']].copy()
    for c,t in zip(CYPS,TARGETS):
        p1=lgb.Booster(model_file=str(ROOT/f'models/F1_final_{c}.txt')).predict(x1, num_threads=4)
        p3=lgb.Booster(model_file=str(ROOT/f'models/F3_final_{c}.txt')).predict(x3, num_threads=4)
        out[t]=.5*(p1+p3)
    if not np.isfinite(out[TARGETS].to_numpy()).all(): raise ValueError('Nonfinite prediction')
    out.to_csv(args.output,index=False)
if __name__=='__main__': main()
