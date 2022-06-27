"""
this script is used for rank reactions
Note: Both pairwise and listwise method generate scores
      Sort all reactions via scores
"""
import os
import logging
import torch
import numpy as np
import csv

from reactranker.data.load_reactions import get_data, Parsing_features
from reactranker.models.base_model import build_model
from reactranker.train.predict import test


path = r'./model_results/Listnet_Gaussian/b97d3_listnetdis_gauss_drop0.1'  # rates_b97d3_changed_loss1

if not os.path.exists(path):
    os.makedirs(path)
log_path = '/home/y_liu/Desktop/rank_reactions/tested_reactions' + '/test.log'
logging.basicConfig(filename=log_path,
                    level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger()

data = get_data('/home/y_liu/Desktop/rank_reactions/tested_reactions/step_1/generated_reactions.csv')
sorted_reaction_path = '/home/y_liu/Desktop/rank_reactions/tested_reactions/step_1/sorted_reactions.csv'
data.read_data(header=None, names=['rsmi', 'psmi'])
# data.filter_bacth(filter_szie=2)
data.drop_columns(label_list=['rsmi', 'psmi'], task_type='keep')
data = data.get_all_data()
k_fold = 9
gpu = None  # cuda
scores = []
var_scores = []
batch_size = 1
train_strategy = 'listnet'
smiles2graph_dic = Parsing_features()

for ii in range(k_fold):
    logging.info(
        '\n\n**********************************\n**    This is the fold [{}/{}]   **\n**********************************'.format(
            ii + 1, k_fold))
    print('**********************************')
    print('**   This is the fold [{}/{}]   **'.format(ii + 1, k_fold))
    print('**********************************')
    path_checkpoints = path
    seed = ii
    k_fold_str = str(ii) + '.pt'
    # shuffle data randomly
    path_checkpoints = os.path.join(path_checkpoints, k_fold_str)
    model = build_model(hidden_size=300,
                        mpnn_depth=3,
                        mpnn_diff_depth=3,
                        ffn_depth=3,
                        use_bias=True,
                        dropout=0,
                        task_num=2,
                        ffn_last_layer='no_softplus')

    logger.info('Model Sturture')
    logger.info(model)
    if gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
    print(path_checkpoints)
    smiles_without_map, smiles_with_map, results = test(model, data, path_checkpoints,
                                                        smiles2graph_dic, gpu=gpu, logger=logger, smiles_name=None)


    if train_strategy != 'evidential':
        score = [a[0] for a in results]
        log_var = np.array([a[1] for a in results])
        var_score = np.exp(log_var).tolist()
        var_scores.append(var_score)
        var_mean = np.mean(var_scores, axis=0)
        print("var_scores is: ", var_mean)
    else:
        score = results
    scores.append(score)
    score_mean = np.mean(scores, axis=0)


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
    sort_result = [[score_mean[i], var_mean[i]] for i in sort_num]
    sort_smile_with_map_and_scores = [[score_mean[i], var_mean[i], rsmi[i], psmi[i]] for i in sort_num]
else:
    sort_smile_with_map_and_scores = [[score_mean[i], rsmi[i], psmi[i]] for i in sort_num]
for i in range(len(sort_smile_with_map)):
    print(sort_smile_with_map_and_scores[i])

with open(sorted_reaction_path, 'w') as csvfile:
    writer = csv.writer(csvfile)
    for row in sort_smile_with_map_and_scores:
        writer.writerow(row)
