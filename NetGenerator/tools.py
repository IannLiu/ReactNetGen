"""
this script is the tools for rank reactions
"""
from itertools import combinations

import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

from rdkit.Chem.Descriptors import NumRadicalElectrons


RDKIT_SMILES_PARSER_PARAMS = Chem.SmilesParserParams()

def str_to_mol(string: str, explicit_hydrogens: bool = True) -> Chem.Mol:

    if string.startswith('InChI'):
        mol = Chem.MolFromInchi(string, removeHs=not explicit_hydrogens)
    else:
        # Set params here so we don't remove hydrogens with atom mapping
        RDKIT_SMILES_PARSER_PARAMS.removeHs = not explicit_hydrogens
        mol = Chem.MolFromSmiles(string, RDKIT_SMILES_PARSER_PARAMS)

    if explicit_hydrogens:
        return Chem.AddHs(mol)
    else:
        return Chem.RemoveHs(mol)
        

def get_radical_info(mol):
    radical_list = {} 
    for atom in mol.GetAtoms():
        num_radical = atom.GetNumRadicalElectrons()
        atom_map_num = atom.GetAtomMapNum()
        if num_radical > 0:
            for na in atom.GetNeighbors():
                na_rad = na.GetNumRadicalElectrons()
                if na_rad > 0:
                    na_map_num = na.GetAtomMapNum()
                    if na_map_num > atom_map_num:
                        key = (atom_map_num, na_map_num)
                        value = [atom.GetIdx(), na.GetIdx()]
                    else:
                        key = (na_map_num, atom_map_num)
                        value = [na.GetIdx(), atom.GetIdx()]
                    if key not in radical_list.keys():
                        radical_list[key] = value
    return radical_list


def get_new_mols(mol, radical_dic):
    
    BOND_TYPE = [0, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE]
    mol_list = []
    for key, value in radical_dic.items():
        a1 = mol.GetAtomWithIdx(value[0])
        a2 = mol.GetAtomWithIdx(value[1])
        rd_num1 = a1.GetNumRadicalElectrons()
        rd_num2 = a2.GetNumRadicalElectrons()
        bond = mol.GetBondBetweenAtoms(value[0],value[1])
        if bond is not None:
            bd_type = bond.GetBondType()
        else:
            bd_type = None
        if bd_type is not None and bd_type in BOND_TYPE:
            bond_idx = BOND_TYPE.index(bd_type)
            if bond_idx < 3:
                new_mol = Chem.RWMol(mol)
                new_mol.RemoveBond(value[0],value[1])
                new_mol.AddBond(value[0],value[1],BOND_TYPE[bond_idx+1])
                atom1 = new_mol.GetAtomWithIdx(value[0])
                atom1.SetNumRadicalElectrons(rd_num1-1)
                atom2 = new_mol.GetAtomWithIdx(value[1])
                atom2.SetNumRadicalElectrons(rd_num2-1)
                edited_mol = new_mol.GetMol()
                Chem.SanitizeMol(edited_mol)
            mol_list.append(edited_mol)
    
    return mol_list

    
def filter_reactions(reactants, outcomes):
    new_outcomes = []
    for outcome in outcomes:
        mol = str_to_mol(outcome)
        if NumRadicalElectrons(mol) >= 2:
            radical_dic = get_radical_info(mol)
            if radical_dic in [None, {}]:
                if outcome not in new_outcomes:
                    new_outcomes.append(outcome)
            else:
                mol_list = get_new_mols(mol, radical_dic)
                for new_mol in mol_list:
                    new_smi = Chem.MolToSmiles(new_mol)
                    if new_smi != reactants:
                        if new_smi not in new_outcomes:
                            new_outcomes.append(new_smi)
        else:
            if outcome not in new_outcomes:
                new_outcomes.append(outcome)
    return new_outcomes