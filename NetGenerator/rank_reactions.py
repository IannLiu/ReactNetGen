"""
this script is used for rank reactions
Note: Both pairwise and listwise method generate scores
      Sort all reactions via scores
"""
import torch
import os
import logging
import torch
import numpy as np
import csv
import pandas as pd
import rdkit
import math

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from reactranker.data.load_reactions import get_data, Parsing_features
from reactranker.models.base_model import build_model
from reactranker.train.predict import test

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


def form_data(reactants, outcomes, temperatures):
    """
    This function is to form DataFrame for rank reactions
    
    param reactants: the reactants smiles, list
    param outcomes: the outcomes smiles, list
    """
    # if the number of reactants equal to outcomes, merging the outcomes and reactant
    # else, expanding the reactants
    if len(reactants) != len(outcomes):
        rsmi = reactants * len(outcomes)
    else:
        rsmi = reactants
    if len(temperatures) != len(outcomes):
        temp = temperatures * len(outcomes)
    else:
        temp = temperatures
    reaction_list = [[r, p, t/1000] for r, p, t in zip(rsmi, outcomes, temp)]
    reaction_df = pd.DataFrame(reaction_list, columns = ['rsmi', 'psmi', 'temperatures'])
    
    return reaction_df
    

def rank_reactions(model_path, reaction_data, train_strategy='listnet', 
                   gpu=None, batch_size=1, fold_number=10, show_info=False, model_dic=None):
    """
    Rank reactions by running ranking models
    
    param model_path: the model results that should be loaded
    param reaction_data: the reactants and products smiles dataframe
    param train_strategy: the traing strategy
    param model_dic: the model information
    """
    path = model_path
    if not os.path.exists(path):
        os.makedirs(path)
    log_path = './test.log'
    logging.basicConfig(filename=log_path,
                        level=logging.INFO,
                        format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger()

    data = reaction_data
    k_fold = fold_number
    gpu = gpu  # cuda
    scores = []
    std_scores = []
    batch_size = batch_size
    train_strategy = train_strategy
    smiles2graph_dic = Parsing_features()
    for ii in range(k_fold):
        logging.info(
            '\n\n**********************************\n**    This is the fold [{}/{}]   **\n**********************************'.format(
                ii + 1, k_fold))
        if show_info:
            print('**********************************')
            print('**   This is the fold [{}/{}]   **'.format(ii + 1, k_fold))
            print('**********************************')
        path_checkpoints = path
        seed = ii
        k_fold_str = str(ii) + '.pt'
        # shuffle data randomly
        path_checkpoints = os.path.join(path_checkpoints, k_fold_str)
        if model_dic is None:
            model = build_model(hidden_size=300,
                                mpnn_depth=3,
                                mpnn_diff_depth=3,
                                ffn_depth=3,
                                use_bias=True,
                                dropout=0.1,
                                task_num=2,
                                ffn_last_layer='with_softplus',
                                add_features_dim=1)
        else:
            model = build_model(hidden_size=model_dic['hidden_size'],
                                mpnn_depth=model_dic['mpnn_depth'],
                                mpnn_diff_depth=model_dic['mpnn_diff_depth'],
                                ffn_depth=model_dic['ffn_depth'],
                                use_bias=model_dic['use_bias'],
                                dropout=model_dic['dropout'],
                                task_num=model_dic['task_num'],
                                ffn_last_layer=model_dic['ffn_last_layer'])

        logger.info('Model Sturture')
        logger.info(model)
        if gpu is not None:
            torch.cuda.set_device(gpu)
            model = model.cuda(gpu)
        if show_info:
            print(path_checkpoints)
        smiles_without_map, smiles_with_map, results = test(model, data, path_checkpoints,
                                                            smiles2graph_dic, gpu=gpu, logger=logger, smiles_name=None)


        """
        if train_strategy != 'evidential':
            score = [a[0] for a in results]
            log_var = np.array([a[1] for a in results])
            var_score = np.exp(log_var).tolist()
            var_scores.append(var_score)
            var_mean = np.mean(var_scores, axis=0)
            if show_info:
                print("var_scores is: ", var_mean)
        else:
            score = results
        """
        score = [a[0] for a in results]
        std_list = np.array([a[1] for a in results])
        # print(results)
        # var_score = np.exp(log_var).tolist()
        std_score = std_list.tolist()
        std_scores.append(std_score)
        std_mean = np.mean(std_scores, axis=0)
        if show_info:
            print("std_scores is: ", std_mean)
        scores.append(score)
        score_mean = np.mean(scores, axis=0)

    if show_info:
        print("scores is: ", score_mean)
        print(smiles_without_map)
    
    # sort smiles in terms of score
    sort_item = sorted(enumerate(score_mean), key=lambda x: x[1], reverse=True)
    sort_num = [a[0] for a in sort_item]
    rsmi = [a[0] for a in smiles_with_map]
    psmi = [a[1] for a in smiles_with_map]

    sort_smile_without_map = [smiles_without_map[i] for i in sort_num]
    sort_smile_with_map = [smiles_with_map[i] for i in sort_num]
    if train_strategy != 'evidential':
        sort_result = [[score_mean[i], std_mean[i]] for i in sort_num]
        sort_smile_with_map_and_scores = [[score_mean[i], std_mean[i], rsmi[i], psmi[i]] for i in sort_num]
        sort_smile = [[rsmi[i], psmi[i]] for i in sort_num]
    else:
        sort_smile_with_map_and_scores = [[score_mean[i], rsmi[i], psmi[i]] for i in sort_num]
        sort_smile = [[rsmi[i], psmi[i]] for i in sort_num]
    
    return sort_smile_with_map_and_scores, sort_smile
    
    
def show_results(all_results):
    """
    Show the final reuslts of ranked reaction lists
    
    param all_results: the final results including ranked outcomes and scores 
    """
    reactant = all_results[0][-2]
    print('==============Following is the ranked reactions==============')
    for idx, result in enumerate(all_results):
        print('*'*40)
        print('{} pathways {}'.format(idx+1, result[:2]))
        psmiles = result[-1]
        rsmiles = result[-2]
        reaction_str = rsmiles+">>"+psmiles
        print('reactants smiles: ', rsmiles)
        rmol = str_to_mol(rsmiles)
        display(rmol)
        print('products smiles: ', psmiles)
        pmol = str_to_mol(psmiles)
        display(pmol)
        
        
def select_reactions(reaction_batch, cut_off, use_uncertainty=True, show_info=True):
    """
    selecting reactions by their scores with or without uncertainty
    """
    if cut_off < 1:
        length = round(len(reaction_batch)*cut_off)
    else:
        length = cut_off
    if show_info:
        print('selected reactions are top: ', length)
    expanded = []
    ith = length
    if use_uncertainty:
        score = np.array([s[0] for s in reaction_batch])
        std = np.array([s[1] for s in reaction_batch])
        threshold = score[length-1]-std[length-1]
        for s, std in zip(score[length:], std[length:]):
            ith += 1
            if s + std > threshold:
                expanded.append(ith)
        if show_info:
            print('expanded reactions are: ', expanded)
        
    return length, expanded
    