# Bayesian Deep Learning for Interactive Question Answering

## Project Structure
```
├── main.py                            The entry of the whole project
├── evaluator                          Calculate accuracy and NDCG
│   ├── evaluation.py
│   └── rank_metrics.py
├── ./log                              saved results
├── load_data.py                       Load original data for creating dataset
|── dataset_collection.py              Create subclass of Dataset in pytorch
├── models                             Essential model 
│   └── bert_ranker.py
├── oracle                              Simulated users 
│   └── lno_ref_values.py
├── queries                             Query Strategy
│   ├── expected_improvement.py
│   ├── pairuncquerier.py
│   ├── random_querier.py
│   └── utils.py
├── resources.py                        Constants
├── reward_learner                      Surrogate models
│   ├── bayesian_bert.py                SWAG ranker
│   └── vallina_bert.py                 Conventional ranker
├── rouge                               Claculate Rouge score 
│   ├── ROUGE-RELEASE-1.5.5
│   ├── __init__.py
│   ├── __pycache__
│   ├── rouge.py
│   └── utils.py
├── scripts                             Scripts for conducting experiments
│   ├── final_interactive
│   ├── interactive_hyper
│   └── warmup
├── swag                                Implementation of SWAG 
│   ├── _assess_dimension.py
│   ├── subspaces.py
│   ├── swag.py
│   └── utils.py
└── utils                                  Utils
    ├── evaluator.py
    ├── misc.py
    └── rank_metrics.py
```
- For simulated users and query strategies, we refer to [the GPPL method](https://github.com/UKPLab/tacl2020-interactive-ranking/tree/master/data)
- For Rouge, we refer to [Rouge](https://github.com/kavgan/ROUGE-2.0)
- For the SWAG inference method, we refer to [SWAG](https://github.com/wjmaddox/swa_gaussian)


## Runing Experiments
All experiments are conducted on Slurm.

### Initial ranker in the warm start phase 
Based on different inference mthods, there are two types of models trained: the SWAG-based deep learning model and the conventional deep learning model.

- The script used to train the coventional deep learning model is `./scripts/warmup/train-nonbayes-topic.sh` where *topic* can be replaced by *apple, cooking, travel*.

- The script used to train the SWAG-based deep learning model is `./scripts/warmup/swag/train-swag-topic.sh` where *topic* can be replaced by *apple, cooking, travel*.

- The hyperparameters tuning can be done by changing parameters in the command line. 

### Interative deep rankers in the interaction phase
After getting the optimal initial rankers,  we uses them to fine-tune given questions according to the user's feedback.

Firslty, we tune hyperparameters of interactive deep rankers using scripts in `./scripts` 
- For the SWAG-based interactive ranker, the scritps are `./scripts/interactive_hyper/swag/topic-parallel.sh` where *topic* is *apple, cooking, travel*.

- For Dropout-based and non-Bayesian interactive rankers, the scripts are `./scripts/interactive_hyper/topic-parallel.sh` where *topic* is *apple, cooking, travel*. These results could be found in the Chapter 5 of the dissertation. 

Then, we use the optimal interactive rankers to get the best answer for a specific question via interacting with users. The scripts are in `./scripts/final_interactive/`. 
- For dropoutt-based rankers, they could be found in `./scripts/final_interactive/dropout`. 
- For non-bayesian rankers, they could found in `.scripts/final_interactive/unc`. 
- For SWAG-based rankers, they are in `scripts/final_interactive/swag`

