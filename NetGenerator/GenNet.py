import json
import copy
import math
from itertools import combinations

import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem


from rdradical.main import rdradicalRun, rdradicalRunText, rdradicalReaction, rdradicalReactants
from rdradical import extractor_radical_invert
from NetGenerator.rank_reactions import form_data, rank_reactions, show_results
from NetGenerator.tools import filter_reactions

from rdkit.Chem.Descriptors import NumRadicalElectrons


rank_by_rmg = True
try:
     from NetGenerator.rank_reactions_rmg import rank_reactions_by_rmg
except:
    print('Please install RMG-database!')
    rank_by_rmg =False


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

        
def read_temp(temp_path):
    dic_path = temp_path
    file = open(dic_path, 'r') 
    js = file.read()
    react_dic = json.loads(js)   
    file.close()
    react_dic_reverse = copy.deepcopy(react_dic)
    for key, value in react_dic.items():
        reac_temp = value['reaction_smarts']
        if reac_temp in [None, {}, '']:
            continue
        rad_rt = value['react_temp_radical']
        rad_pt = value['prod_temp_radical']
        reactant_smarts = reac_temp.split('>')[0]
        react_dic_reverse[key]['reaction_smarts'] = value['reaction_smarts'].split('>')[2] + '>>' + value['reaction_smarts'].split('>')[0]
        react_dic_reverse[key]['react_temp_radical'] = value['prod_temp_radical']
        react_dic_reverse[key]['prod_temp_radical'] = value['react_temp_radical']
    
    return react_dic, react_dic_reverse

    
def assign_map_num(smi):
    mol = str_to_mol(smi)
    #Check if smi has atom map numbers
    Has_map_num = True
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            Has_map_num = False
            break
    # if any atom without map_num
    if Has_map_num is False:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx()+1)
    else:
        return smi
    smi1 = Chem.MolToSmiles(mol)
    
    return smi1


def regular_smi(smi_list):
    """
    This function is to regularize smiles list
    """
    reg_smi = []
    for smi in smi_list:
        reg_smi.append(Chem.MolToSmiles(Chem.MolFromSmiles(smi)))
        
    return reg_smi
    
    

def drop_map_num(smi):
    mol = Chem.MolFromSmiles(smi)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)
     

def drop_Hs_map_num(smi):
    mol = Chem.MolFromSmiles(smi)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'H':
            atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)
    
def drop_same_reactions(reactants, mapped_outcome_dup):
    drop_dup_outcomes = {}
    outcomes = []
    for smi in mapped_outcome_dup:
        mol = str_to_mol(smi)
        smi_nomap = drop_Hs_map_num(smi)
        if smi_nomap not in drop_dup_outcomes.keys():
            drop_dup_outcomes[smi_nomap] = smi
            outcomes.append(smi)
        else:
            continue
    # drop outcome by compare the template and smiles
    # if the smiles of outcomes is same, the reaction smarts is same as well
    # the outcomes are same
    print(outcomes)
    psmis_nomap = []
    reaction_smarts = []
    dropped_outcomes = []
    for i, psmi in enumerate(outcomes):
        reaction = {}
        reaction['reactants'] = reactants
        reaction['products'] = psmi
        reaction['_id'] = i
        template = extractor_radical_invert.extract_from_reaction(reaction)
        p_nomap = drop_map_num(psmi)
        if p_nomap not in psmis_nomap:  # if the no_map_smi not in the list, append!
            psmis_nomap.append(p_nomap)
            reaction_smarts.append(str(template['reaction_smarts']))
            dropped_outcomes.append(psmi)
        else:  # compare the reaction template
            idx_list = []
            for k, smi in enumerate(psmis_nomap):
                if smi == p_nomap:
                    idx_list.append(k)
            
            reaction_exits = False
            for idx in idx_list:
                if reaction_smarts[idx] == template['reaction_smarts']:
                    reaction_exits = True
                    continue
            if not reaction_exits:
                psmis_nomap.append(p_nomap)
                reaction_smarts.append(template['reaction_smarts'])
                dropped_outcomes.append(psmi)
    return dropped_outcomes
    
                    
def run_one_step(reactants, react_dic, direction: str, dropped_temp = [], rsmi_psmi_del=False):
    outcomes = []
    for key, value in react_dic.items():
        # run reactant by rdkit. react_temp: reaction SMARTS; rad_rt: dictionary; rad_pt:dictionary
        if key in dropped_temp:
            continue
        reac_temp = value['reaction_smarts']
        rad_rt = value['react_temp_radical']
        rad_pt = value['prod_temp_radical']
        if reac_temp in [None, {}, '']:
            continue
        reactant_smarts = reac_temp.split('>')[0]
        if reactant_smarts[0] == '(' and reactant_smarts[-1] == ')':
            reactant_smarts_length = 1
        else:
            reactant_smarts_length = len(reactant_smarts.split('.'))
        if len(reactants.split('.')) == reactant_smarts_length:
            outcomes_rdchirl, mapped_outcome, mapped_outcome_dup  =  rdradicalRunText(reac_temp, rad_rt, rad_pt, reactants)
            reactants_no_map = drop_map_num(reactants)
            outcome_sing_temp = {}
            for outcome_smi in mapped_outcome_dup:
                smi = drop_Hs_map_num(outcome_smi)
                smi_without_map = drop_map_num(outcome_smi)
                if smi not in outcome_sing_temp.keys():
                    if  rsmi_psmi_del:
                        if smi_without_map != reactants_no_map:
                            outcome_sing_temp[smi] = [outcome_smi]
                            outcomes.append(outcome_smi)
                    else:
                        outcome_sing_temp[smi] = [outcome_smi]
                        outcomes.append(outcome_smi)
                    
                else:
                    continue
            """
            if key ==  'intra_H_migration':
                outcomes.extend(drop_same_reactions(reactants, mapped_outcome_dup))
            else:
                for outcome_smi in mapped_outcome_dup:
                    smi = drop_map_num(outcome_smi)
                    if smi not in outcome_sing_temp.keys():
                        outcome_sing_temp[smi] = [outcome_smi]
                    else:
                        continue
                for mapped_outcome_no_dup in outcome_sing_temp.values():
                    outcomes.extend(mapped_outcome_no_dup)
            """
    
    return outcomes

def run_one_rank(reactants, react_dic, react_dic_reverse=None, 
                 use_forward_temp=True, use_reverse_temp=False, dropped_temp=[], rsmi_psmi_del=False):
    outcomes = []
    # run forward reactions
    outcomes_dup = []
    outcomes_dic = {}
    # products = []
    # products_dic = {}
    if use_forward_temp:
        forward_outcomes = run_one_step(reactants, react_dic, 'forward_', dropped_temp, rsmi_psmi_del=rsmi_psmi_del)
        outcomes.extend(forward_outcomes)
    # run reverse reactions
    if use_reverse_temp:
        if react_dic_reverse is None:
            print('Warnning !! the reaction dict of recerse reaction is None')
        else:
            reverse_outcomes = run_one_step(reactants, react_dic_reverse, 'reverse_', dropped_temp, 
                                            rsmi_psmi_del=rsmi_psmi_del)
            for reverse_outcome in reverse_outcomes:
                if reverse_outcome not in outcomes:
                    outcomes.append(reverse_outcome)
    # drop the outcomes
    # for Hs migration reactions, drop_same_reactions by smiles and template. 
    # For other reactions, dropping by smiles
    products = []
    for smi in outcomes:
        outcome_drop_Hs = drop_Hs_map_num(smi)
        if outcome_drop_Hs not in outcomes_dic:
            outcomes_dic[outcome_drop_Hs] = [smi]
            products.append(smi)
        else:
            continue
    """
    for key, values in outcomes_dic.items():
        if key == 'forward_intra_H_migration' or key == 'reverse_intra_H_migration':
            products.extend(drop_same_reactions(reactants, values))
        else:
            for value in values:
                smi = drop_map_num(value)
                if smi not in products_dic.keys():
                    products_dic[smi] = [value]
                    products.append(value)
                else:
                    continue
    
    forward_outcomes.extend(reverse_outcomes)
    if use_reverse_temp:
        for outcome in forward_outcomes:
            smi = drop_map_num(outcome)
            if smi not in outcomes_dic.keys():
                outcomes_dic[smi] = [outcome]
                outcomes.append(outcome)
            else:
                continue
    """
    return products

    
def NetGen(react_list: list, ranks: int, cut_off, min_paths: int, bi_molecule, react_dic, react_dic_reverse, 
           use_reverse_temp, use_uncertainty=False, dropped_temp=None, model_path=None, show_info=True, 
           rank_pathways=True, using_rmg_database=False, rmg_estimator='group additivity', temperature=1000, pressure=1):
    """
    define the reaction_dic and species_dic
    if species in species_dic, this species should be removed 
    (new species will react with old species, old species react with old species already)
    if a reaction already happend, skip this reaction
    In summary, we should build a species dic and reaction dic. For every update if species_dic,
    the reaction_dic updated as well
    For every new reactant(s) (active_species), checking this reactant(s) has been used by the old species
    after reactions, the active_species moved to species_dic.
    
    param react_list: the reactant SMILES list
    param ranks: the rank of species
    param cut_off: the cutoff of reaction candidate list.
    param min_path: at least min_path reactions should be selected (every rank)
    param bi_molecule: Run bi_molecule reaction template
    param react_dic: the reaction template dictionary
    param react_dic_reverse: the reserse reaction template dictionary
    param use_reverse_temp: whether use reverse reaction template
    param dropped_temp: Forbident the usage of this template
    param model_path: the path of ranking models
    param show_info: print the information
    param rank_pathways: ranking reaction pathways or not.
    
    return: reaction list and species dic
    """
    
    if using_rmg_database:
        f = open('rmg_ranker_out.csv', 'a')
    else:
        f = open('ml_ranker_out.csv', 'a')
    f.write('\n rank_stage, ith_rank, rsmi, psmi, ranked_flag, selected_flag')
    smi_list = react_list
    ranks = ranks
    path = model_path
    cut_off = cut_off
    min_paths = min_paths
    reactant = regular_smi(smi_list)
    bi_molecule = bi_molecule
    species_dic = {}
    reaction_list = []
    for react in reactant:
        species_dic[react]=[[0, 0]]
    active_species = reactant
    use_uncertainty = use_uncertainty
    rank_pathways = rank_pathways
    rank_counter = 0
    if using_rmg_database:
        num_path_rmg_ranked = 0
        num_path_ml_ranked = 0
        print('rate estimator: ', rmg_estimator)
    for rank in range(ranks):
        old_species = []
        reaction_counter = 0
        # update the old species list and reactant list(from reaction list)
        for specie in species_dic.keys():
            old_species.append(specie)
        if reaction_list is not None:
            reactants = [r[0] for r in reaction_list]
        # run reactants for active species
        outcomes_in_one_rank = {}
        # combine the active species and old species.
        # Note that active species can react with old species and themselves. We just consider the bi-molecule reactions
        if bi_molecule:
            reaction_species = []
            for ii in combinations(active_species, 2):
                diff_comb = list(ii)
                diff_comb.sort()
                diff_comb_smi = '.'.join(diff_comb)
                if diff_comb_smi not in reaction_species:
                    reaction_species.append(diff_comb_smi)
            for ii in range(len(active_species)):
                for jj in range(len(old_species)):
                    r_list = [active_species[ii], old_species[jj]]
                    r_list.sort()
                    rsmi = '.'.join(r_list)
                    if rsmi not in reaction_species:
                        reaction_species.append(rsmi)
            for kk in range(len(active_species)):
                ssmi = active_species[kk] + '.' + active_species[kk]
                if ssmi not in reaction_species:
                    reaction_species.append(ssmi)
            reaction_species.extend(active_species)
        else:
            reaction_species = active_species
        for rsmi in reaction_species:
            # if reactants exit in reactant list, this reaction has been run. Skip!
            # rsmi.sort()
            if rsmi in reactants:
                continue
            rsmi_mapped = assign_map_num(rsmi)
            rsmi_no_Hs_map = drop_Hs_map_num(rsmi_mapped)
            outcomes = run_one_rank(rsmi_mapped, react_dic, react_dic_reverse, 
                                    use_reverse_temp=use_reverse_temp, dropped_temp=dropped_temp,rsmi_psmi_del=True)
            # no_Hs_outcomes = [drop_Hs_map_num(outcome) for outcome in outcomes]
            no_map_outcomes = [drop_map_num(outcome) for outcome in outcomes]
            if len(rsmi_mapped.split('.')) == 2:
                rsmi_mapped1 = rsmi_mapped.split('.')[1] + '.' + rsmi_mapped.split('.')[0]
                outcomes1 = run_one_rank(rsmi_mapped1, react_dic, react_dic_reverse, 
                                         use_reverse_temp=use_reverse_temp, dropped_temp=dropped_temp,rsmi_psmi_del=True)
                for outcome1 in outcomes1:
                    if drop_map_num(outcome1) not in no_map_outcomes:
                        outcomes.extend(outcomes1)
            if outcomes in [[], None]:
                continue
                
            outcomes = filter_reactions(rsmi_mapped, outcomes)
            if outcomes in [[], None]:
                continue
            if rank_pathways:
                # rank reactions
                reaction_df = form_data([rsmi_mapped], outcomes, [temperature])
                if using_rmg_database and rank_by_rmg:
                    sort_smile_with_map_and_scores, sort_smile = rank_reactions_by_rmg(reaction_df, temperature, pressure, rmg_estimator)
                    rank_counter += 1
                    # num_path_rmg_ranked += reaction_df.shape[0]
                    # some of reactants might not be selected
                else:
                    sort_smile_with_map_and_scores, sort_smile = rank_reactions(path, reaction_df, fold_number=5)
                    rank_counter += 1
                    # all reactants can be ranked
                    
                # selecting reactions
                scores = np.array([s[0] for s in sort_smile_with_map_and_scores])
                stds = np.array([s[1] for s in sort_smile_with_map_and_scores])
                products = [s[3] for s in sort_smile_with_map_and_scores]
                if cut_off < 1: # if cut_off smaller than 1, cut_off means the propotion of length
                    length = round(len(scores) * cut_off)
                    if length > min_paths:
                        selected_outcomes = products[:length]
                        if use_uncertainty:
                            other_outcomes = products[length:]
                            cand_scores = scores[length:]+stds[length:]
                            threshold = np.min(scores[:length]-stds[:length])
                            for i, score in enumerate(cand_scores):
                                if score > threshold:
                                    selected_outcomes.append(other_outcomes[i])
                    elif len(scores) > min_paths:
                        selected_outcomes = products[:min_paths]
                    else:
                        selected_outcomes = products
                else: # if cut_off larger than 1, cut_off euqal to the length
                    length = cut_off
                    if len(scores) < length:
                        selected_outcomes = products
                    else:
                        selected_outcomes = products[:length]
                        if use_uncertainty:
                            other_outcomes = products[length:]
                            cand_scores = scores[length:]+stds[length:]
                            threshold = np.min(scores[:length]-stds[:length])
                            for i, score in enumerate(cand_scores):
                                if score > threshold:
                                    selected_outcomes.append(other_outcomes[i])
                final_outcomes = selected_outcomes
            else:
                final_outcomes = outcomes
            
            # write the rank stage, generated reactions, ranked reactions, selected reactions
            
            w_reactant = rsmi_mapped
            w_product1 = final_outcomes  # selected outcomes
            if rank_pathways:
                for w_psmi in w_product1:
                    f.write('\n'+ str(rank) +','+ str(rank_counter) + ',' + w_reactant + ',' + w_psmi + ',' + '1,'+ '1')
                if len(final_outcomes) < len(products):
                    for w_psmi in products[len(final_outcomes):]:
                        f.write('\n'+ str(rank) + ',' + str(rank_counter) + ',' + w_reactant + ',' + w_psmi + ',' + '1,'+ '0')
                if len(products) < len(outcomes):
                    # some reactions can not be ranked by RMG dataset.
                    # Add these reactions
                    w_product3 = [prod for prod in outcomes if prod not in products]
                    for w_psmi in w_product3:
                        f.write('\n'+ str(rank) + ','+ str(rank_counter) + ',' + w_reactant + ',' + w_psmi + ',' + '0,'+ '0')
            

            for outcome in final_outcomes:
                reaction_counter += 1
                mapped_psmis = outcome.split('.')
                no_map_psmis = [drop_map_num(p) for p in mapped_psmis]
                no_Hs_psmis = [drop_Hs_map_num(p) for p in mapped_psmis]
                # add this rection to reaction list
                no_map_psmis.sort()
                product_smi = '.'.join(no_map_psmis)
                product_no_Hs_map = '.'.join(no_Hs_psmis)
                reaction_flag = [rsmi_no_Hs_map, product_no_Hs_map, rank+1, reaction_counter]
                reaction_list.append(reaction_flag)
                for no_map_psmi in no_map_psmis:
                    if no_map_psmi in old_species:
                        species_dic[no_map_psmi].append([rank+1, reaction_counter])
                    elif no_map_psmi in outcomes_in_one_rank.keys():
                        outcomes_in_one_rank[no_map_psmi].append([rank+1, reaction_counter])
                    else:
                        outcomes_in_one_rank[no_map_psmi] = [[rank+1, reaction_counter]]
        # update the activate species and reactant dic
        active_species = list(outcomes_in_one_rank.keys())
        if active_species in [[], None]:
            break
        species_dic.update(outcomes_in_one_rank)
        if show_info:
            print(active_species)
            print('*******************')
    if using_rmg_database:
        return reaction_list, species_dic, [num_path_rmg_ranked, num_path_ml_ranked]
    else:
        return reaction_list, species_dic, ['all of the reactions ranked by ML models']
    
    
def trans_to_graph(reaction_list):
    """
    tranform the reaction list to graph.
    The nodes represent species and edgs represnt reaction
    """
    link_data = []
    links = []
    for reaction in reaction_list:
        reactants = reaction[0].split('.')
        rsmi = [Chem.MolToSmiles(Chem.MolFromSmiles(drop_map_num(r))) for r in reactants]
        products = reaction[1].split('.')
        psmi = [Chem.MolToSmiles(Chem.MolFromSmiles(drop_map_num(p))) for p in products]
        if len(reactants) > 1:
            node_type = '0'
            rsmi.sort()
            source = '.'.join(rsmi)
            link1 = rsmi[0] + '>' + source
            if link1 not in links:
                links.append(link1)
                link_data.append([rsmi[0], source, node_type])
            link2 = rsmi[1] + '>' + source
            if link2 not in links:
                links.append(link2)
                link_data.append([rsmi[1], source, node_type])
        else:
            node_type = '1'
            source = rsmi[0]
            if len(products) > 1:
                link3 = source + '>' + psmi[0]
                if link3 not in links:
                    links.append(link3)
                    link_data.append([source, psmi[0], node_type])
                link4 = source + '>' + psmi[1]
                if link4 not in links:
                    links.append(link4)
                    link_data.append([source, psmi[1], node_type])
            else:
                link5 = rsmi[0] + '>' + psmi[0]
                if link5 not in links:
                    links.append(link5)
                    link_data.append([rsmi[0], psmi[0], node_type])
    edges = pd.DataFrame(np.array(link_data), columns = ['source','target','type'])
    
    return link_data, edges, links
    
    
def run_bi_mol(reactant, react_dic, react_dic_reverse, use_forward_temp=True, use_reverse_temp=True):
    """
    run bimolecular reactions
    """
    reactant1 = reactant.split('.')[1] + '.'+ reactant.split('.')[0]
    outcomes = run_one_rank(reactant, react_dic, react_dic_reverse, 
                            use_forward_temp=use_forward_temp, use_reverse_temp=use_reverse_temp)
    outcomes1 = run_one_rank(reactant1, react_dic, react_dic_reverse, 
                             use_forward_temp=use_forward_temp, use_reverse_temp=use_reverse_temp)
    outcomes_drop_H = [drop_Hs_map_num(out) for out in outcomes]
    outcomes1_drop_H = [drop_Hs_map_num(out1) for out1 in outcomes1]
    for i, s in enumerate(outcomes1_drop_H):
        if s not in outcomes_drop_H:
            outcomes.append(outcomes1[i])
    return(outcomes)
