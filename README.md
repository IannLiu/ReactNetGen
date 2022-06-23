# ReactNetGen (Chemical Reaction Network Generator)

[![MIT license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](http://opensource.org/licenses/MIT)

Enable ranking elementary reactions and generating reaction networks. The rank information was learned by an uncertainty calibrated learning to rank (LTR) model on the [quantum chemical data](https://zenodo.org/record/3715478) and [RMG-database](https://github.com/ReactionMechanismGenerator/RMG-database).

* Uncertainty-calibrated deep learning approach for rapid identification of gas-phase reaction networks: peer-reviewed publication (open access).

## Requirements
For basic use, installing:
* python
* numpy
* pandas
* pytorch
* rdkit
* sklearn
* tqdm

RMG-database can be installed as the rank engine by installing
* [RMG-Py](https://github.com/ReactionMechanismGenerator/RMG-Py)
* [RMG-database](https://github.com/ReactionMechanismGenerator/RMG-database)

## Usage
### Basic usage (Finding possible reaction pathways)
1. Import the NetGenerator and load reaction temperates for generating reaction pathways 

```python
from NetGenerator.rank_reactions import form_data, rank_reactions, show_results
import NetGenerator.GenNet as GenNet

react_dic, react_dic_reverse = GenNet.read_temp('./templates/rmg_template.txt')
reactant = GenNet.assign_map_num('C1CC[CH]C1=O')
outcomes = GenNet.run_one_rank(reactant, react_dic, react_dic_reverse, use_forward_temp=True, use_reverse_temp=True)
```
2. Format input file for ranking reaction pathways

``` python
reaction_df = form_data([reactant], outcomes)
```

3. Rank reaction pathways using LTR models

```python
path = r'./reactranker/model_results/final_results/evidential_ranking' # load LTR models
sort_smile_with_map_and_scores, sort_smile = rank_reactions(path, reaction_df)
```

### Basic usage (Generating reaction networks)

1. Importing the NetGenerator and load reaction temperates for generating reaction pathways 

```python
from NetGenerator.GenNet import NetGen, read_temp, trans_to_graph
dic_path = './templates/rmg_template.txt'
react_dic, react_dic_reverse = read_temp(dic_path)
```

2. Genertaing reaction networks

```python
smi_list = ['C1C[CH]CC1=O', 'C1CC[CH]C1=O', '[CH2]CCC[C]=O']  # the reactant list
ranks = 4  # The species rank for terminating the generation of reaction networks
path = r'./reactranker/model_results/final_results/evidential_ranking'  # The LTR model path
cut_off = 2  # reserveing 2 reactions per rank
min_paths = 1  # reserveing 1 reactions at least per rank
use_uncertainty = True  # using uncertainty calibrated results
rank_pathways = True  # ranking the pathways
reaction_list, species_dic,info = NetGen(react_list=smi_list, ranks=ranks, cut_off=cut_off, min_paths=min_paths, 
                                    bi_molecule=False, react_dic=react_dic, react_dic_reverse=react_dic_reverse,
                                    use_uncertainty=use_uncertainty, use_reverse_temp=True, dropped_temp=[], 
                                    model_path=path, show_info=True, rank_pathways=True, using_rmg_database=False,
                                    rmg_estimator='rate rules')
```

## Advanced usage

### Template operation
#### Add new reaction template
To add new reaction template, following format should be added to ```templates\rmg_template.txt```:

```
template_name: {"reaction_smarts":reaction_smart_string, 
                "react_temp_radical": {"atom map number":int}, 
                "prod_temp_radical": {"atom map number":int}}
```

Where ```template_name``` is the string format of the template name, reaction_smart_string is the string format of atom mapped reaction SMARTS.
#### Dropping reaction template for reaction pathway generation
The reaction template can be forbiden by specifying variable ```dropped_temp``` of ```NetGen``` and ```run_one_rank``` function

### Adding LTR models
The user tranined LRT mdoel can be added to ```ReactNetGen\reactranker\model_results``` for ranking reaction pathways.
