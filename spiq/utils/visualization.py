"""Simple script to create a simple TMAP from a list of SMILES strings
"""
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from faerun import Faerun
from multiprocessing import Pool
import tmap as tm 
import time
import argparse
import pandas as pd
from pandarallel import pandarallel
from spiq.utils.fingerprints import FingerprintCalculator
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process SMILES to TMAP")
    parser.add_argument('--smiles', type=str, help="The path to a file or directory containing SMILES for the TMAP")  
    parser.add_argument('--fingerprint', type=str, default='morgan', help="The fingerprint to be used in the TMAP")
    parser.add_argument('--dataframe', type=str, help="The path to the .csv o .parquet containing `SMILES` and `cluster_id columns. One TMAP per cluster_id value`")

    return parser.parse_args()

def _calculate_fingerprint(smiles:str, fp:str):
    fp_calc = FingerprintCalculator()
    
    fp_arr = fp_calc.FingerprintFromSmiles(smiles,fp=fp, fpSize=1024, radius=3)
    return fp_arr


def _mol_properties_from_smiles(smiles: str) -> tuple:
    """ Get molecular properties from a single SMILES string"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    hac = mol.GetNumHeavyAtoms()
    num_aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    fraction_aromatic_atoms = num_aromatic_atoms / hac if hac > 0 else 0
    number_of_rings = rdMolDescriptors.CalcNumRings(mol)
    molecular_weight = Descriptors.ExactMolWt(mol)
    clogP = Descriptors.MolLogP(mol)
    fraction_Csp3 = Descriptors.FractionCSP3(mol)

    return (hac,num_aromatic_atoms, fraction_aromatic_atoms, number_of_rings, molecular_weight, clogP, fraction_Csp3 )


def calc_properties(smiles:list):

    pandarallel.initialize(progress_bar=False)

    dataframe = pd.DataFrame({
        'smiles': smiles
    })

    dataframe[['HAC', 'Number Aromatic Atoms', 'Fraction Aromatic_atoms', 'Number of Rings', 'MW', 'clogP', 'Fraction Csp3']]  = dataframe['smiles'].apply(
    _mol_properties_from_smiles
    ).parallel_apply(pd.Series)
    
    return dataframe



def create_tmap(smiles:list,
                fingerprint:str,
                descriptors:bool=True, 
                tmap_name:str='my_tmap'):
   

   fp_arr = _calculate_fingerprint(smiles, fingerprint)

   descriptors = calc_properties(smiles) # This should be a np.array

   tm_fingerprint = [tm.VectorUint(fp) for fp in fp_arr]

   lf = tm.LSHForest(1024)
   lf.batch_add(tm_fingerprint)
   lf.index()

   # Get the coordinates and Layout Configuration
   cfg = tm.LayoutConfiguration()
   cfg.node_size = 1/30 
   cfg.mmm_repeats = 2
   cfg.sl_extra_scaling_steps = 10
   cfg.k = 20 
   cfg.sl_scaling_type = tm.RelativeToAvgLength
   x, y, s, t, _ = tm.layout_from_lsh_forest(lf, cfg)

   labels = []

   for inx, row in descriptors.iterrows():
       labels.append(row['smiles'])

   c_columns = ["HAC", "Number Aromatic Atoms", "Fraction Aromatic_atoms",
                "Number of Rings", "MW", "clogP", "Fraction Csp3"]
   c = [descriptors[col].to_numpy() for col in c_columns]

   # Plotting
   f = Faerun(
       view="front",
       coords=False,
       title="",
       clear_color="#FFFFFF",
   )

   f.add_scatter(
       tmap_name+"_TMAP",
       {
           "x": x,
           "y": y,
           "c": np.array(c),
           "labels":labels,
       },
       shader="smoothCircle",
       point_scale= 2.5 ,
       max_point_size= 20,
       interactive=True,
       series_title= ['HAC', 'Number Aromatic Atoms', 'Fraction Aromatic_atoms', 'Number of Rings', 'MW', 'clogP', 'Fraction Csp3'], 
       has_legend=True,           
       colormap=['rainbow', 'rainbow', 'rainbow', 'rainbow', 'rainbow', 'rainbow', 'rainbow'],
       categorical=[False, False, False, False, False, False, False],
   )

   f.add_tree(tmap_name+"_TMAP_tree", {"from": s, "to": t}, point_helper=tmap_name+"_TMAP")
   f.plot(tmap_name+"_TMAP", template='smiles')


if __name__=="__main__":
    args = parse_arguments()
    if args.dataframe is not None:
        import pandas as pd
        if args.dataframe.endswith('.parquet'):
            df = pd.read_parquet(args.dataframe)
        elif args.dataframe.endswith('.csv'):
            df = pd.read_csv(args.dataframe)
        else:
            raise ValueError('Format for dataframe not supported. Only `.csv` and `.parquet` files')

        for cluster in df['cluster_id'].unique():
            # print(df[df['cluster_id']==cluster]['smiles'])
            create_tmap(df[df['cluster_id']==cluster]['smiles'], fingerprint=args.fingerprint, tmap_name=f'tmap_{cluster}')
            print('TMAP generated for cluster_id ', cluster)

    elif args.smiles is not None:
        with open(args.smiles, 'r') as file:
            smiles = file.split(f'/n')
        create_tmap(smiles=args.smiles, fingerprint=args.fingerprint, tmap_name='my_tmap')

    else:
        raise ValueError("Either a valid path for the Dataframe or the SMILES must be provided")