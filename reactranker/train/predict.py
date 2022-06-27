from typing import Union
from logging import Logger
from rdkit import Chem
import numpy as np

import torch
from ..data.load_reactions import DataProcessor


def test(model: object, test_data: object, path_checkpoints: object, smiles2graph_dic: object,
         gpu: Union[int, str], logger: Logger = None, smiles_name: object = None, show_info=False) -> object:
    logger.info('\n==========================================\n'
                '   Now, the test section is beginning!!!   \n'
                '==========================================')
    if show_info:
        print('==========================================')
        print('  Now, the test section is beginning!!!   ')
        print('==========================================')
    logger.info('The path of checkpoints is:\n')
    logger.info(path_checkpoints)
    
    test_len = test_data.shape[0]
    if show_info:
        print('the length of test data is:', test_len)
    logger.info('the length of test data is: {}'.format(test_len))
    
    # build and load model
    state = torch.load(path_checkpoints, map_location=lambda storage, loc: storage)
    loaded_state_dict = state['state_dict']
    means = state['data_scaler']['means']
    stds = state['data_scaler']['stds']
    model = model
    if gpu is not None:
        model = model.cuda()
    model.load_state_dict(loaded_state_dict)
    
    # evaluate
    model.eval()
    test_data_processor = DataProcessor(test_data)
    smiles2graph_dic = smiles2graph_dic
    smiles_without_map_list = []
    smiles_with_map_list = []
    preds_list = []
    for smiles in test_data_processor.generate_pred_batch_per_query(smiles_name=smiles_name):
        smiles = smiles.tolist()
        rsmi = [s[0] for s in smiles]
        psmi = [s[1] for s in smiles]
        r_batch_graph = smiles2graph_dic.parsing_smiles(rsmi)
        p_batch_graph = smiles2graph_dic.parsing_smiles(psmi)
        preds = model(r_batch_graph, p_batch_graph, gpu=gpu)

        # transfer atom mapped smiles to none mapped
        rsmi_without_map = [Chem.MolToSmiles(Chem.MolFromSmiles(i)) for i in rsmi]
        psmi_without_map = [Chem.MolToSmiles(Chem.MolFromSmiles(i)) for i in psmi]
        smiles_without_map = [[rsmi_without_map[i], psmi_without_map[i]] for i in range(len(psmi_without_map))]
        smiles_with_map = [[rsmi[i], psmi[i]] for i in range(len(psmi))]
        preds = preds.tolist()
        smiles_without_map_list.extend(smiles_without_map)
        smiles_with_map_list.extend(smiles_with_map)
        preds_list.extend(preds)
        """
        sort_item = sorted(enumerate(preds), key=lambda x: x[1], reverse=True)
        sort_num = [a[0] for a in sort_item]
        sort_result = [a[1] for a in sort_item]
        sort_smile = [smiles_without_map[i] for i in sort_num]
        sort_results.append(sort_result)
        sort_smiles.extend(sort_smile)
        logger.info('sorted smiles is: {}'.format(sort_smiles))
        logger.info('sorted results is: {}'.format(sort_results))
        """
    
    preds_list_ = np.array(preds_list)
    try:
        preds_list_trans = np.vstack(((preds_list_[:,0]*stds)+means, np.sqrt(preds_list_[:,1]) * stds)).T
    except:
        print(preds_list_)
        print(test_data)
        print(rsmi)
        print(psmi)
        raise Exception('iteration stopped !!')
    
    return smiles_without_map_list, smiles_with_map_list, preds_list_trans
