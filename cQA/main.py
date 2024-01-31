from importlib import util
from scipy.stats.stats import mode
import torch
from load_data import load_cqa
import pandas as pd
import re
import os
import numpy as np
from rouge.rouge import Rouge
from resources import *
from oracle.lno_ref_values import SimulatedUser
import argparse
from reward_learner.bayesian_bert import BayesainBert
from reward_learner.vallina_bert import VanillaBert
from models.bert_ranker import BertRanker
from utils.misc import normaliseList
from joblib.parallel import Parallel, delayed
import json
from queries.expected_improvement import ExpectedImprovementQuerier
from queries.pairuncquerier import PairUncQuerier
from tqdm import trange
import warnings
import swag.utils as utils
from evaluator.evaluation import evaluateReward
from torch.utils.data import DataLoader
from dataset_collection import PosNegDataset


def cal_ref_values(question_id, gold_filename, pool_answers):

    ref_values = []

    if len(pool_answers) == 1:
        print('SKIPPING A QUESTION WITH BAD DATA')
        print(f'question id is: {question_id}')
        return

    ref_values = [compute_ref_values(aidx, answer, gold_filename)
                  for aidx, answer in enumerate(pool_answers)]

    # ref_values = normaliseList(ref_values)
    return ref_values


def compute_ref_values(aidx, answer, gold_filename):
    answer = re.sub('<[^<]+>', "", answer)
    # if np.mod(aidx, 20) == 0:
    #     print('computing ref value for answer: %i' % aidx)
    rouge_scorer = Rouge(ROUGE_DIR, BASE_DIR, True)
    R1, R2, RL, RSU = rouge_scorer(
        answer, [[gold_filename, None]], len(answer))
    rouge_scorer.clean()

    return RL


def active_learner(start_q, model, checkpoint, question_list, n_iter_rounds, ref_values, topic, gold_file_dir, mode, sample_nums, save_dir, stop_epochs, querier_type, num_q, noise, swag=False):
    """[summary]

    Parameters
    ----------
    question_list : list[dict]
        each item is {qid:, gold_an:, pooled_an:}
    n_iter_rounds : int
        nums of iteration
    ref_values : list
        rouge scores
    querier : [type]
        [description]
    """
    oracle = SimulatedUser(ref_values, noise)  # LNO-0.1
    # {q_id:{(c1,c2):label,...},...}
    # log = {}
    ndcg, acc = 0, 0
    # for question_id in trange(len(question_list)):
    # num_question =100
    end_q = start_q + num_q
    for question_id in trange(start_q, end_q):
        if question_id >= len(question_list):
            print(f'the total number of questions is {len(question_list)}')
            end_q = question_id
            break
        model.load_state_dict(checkpoint['state_dict'])
        # if swag:
            # model.swag.n_models = torch.tensor(1, dtype=torch.long)
            # model.swag.sq_mean = torch.zeros(model.swag.num_parameters)
            # model.swag.mean = torch.zeros(model.swag.num_parameters)
        if querier_type == 'imp':
            querier = ExpectedImprovementQuerier(
                reward_learner_class=model, qa_lists=question_list)
        elif querier_type == 'unc':
            querier = PairUncQuerier(reward_learner_class=model, qa_lists=question_list)

        log = {question_id: {}}
        for round in trange(n_iter_rounds):
            # model.load_state_dict(checkpoint['state_dict'])
            print(
                f'question: {question_id}, round {round+1}/{n_iter_rounds}....')
       
            sum1, sum2 = querier.getQuery(
                log, question_id, sample_nums)
            pref = oracle.getPref(question_id, sum1, sum2)
      
            log[question_id][(sum1, sum2)] = pref

            querier.updateRanker(log, stop_epochs)

        utilities = querier.get_utilities(question_id)
        gold_values = ref_values[str(question_id)]
        metric_dict = evaluateReward(utilities, gold_values)
        ndcg += metric_dict['ndcg_at_5%']
        acc += (np.argmax(utilities) == np.argmax(gold_values))
        print(f'metric for {question_id} is {metric_dict}')
    print(
        f'accuracy is {acc/(end_q-start_q)}, ndcg5 is {ndcg/(end_q-start_q)}')

    # utils.save_checkpoint(save_dir=save_dir, epoch=epoch, name=f"interative_{topic}_{}epochs", state_dict=model.state_dict(
    #             ), optimizer=model.optimizer.state_dict(), scheduler=None)
    #     print(f'save {epoch} model to {os.path.join(save_dir, save_name)}')
    return acc, ndcg, num_q


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_datadir', type=str,
                        default='./data/cqa_data')
    parser.add_argument('--topic', type=str, default=None)
    parser.add_argument('--cache_dir', type=str,
                        default='./cache/')
    parser.add_argument('--cqa_dir', type=str,
                        default='./data/processed_data')
    parser.add_argument('--save_dir', type=str,
                        default='./model')
    parser.add_argument('--checkpoint', type=str,  
                        default=None)
                

    parser.add_argument('--n_iter_rounds', type=int,
                        default=1)

    parser.add_argument('--max_num_models', type=int, default=20,
                        help='maximum number of SWAG models to save')
    parser.add_argument('--lr_init', type=float,
                        default=5e-5, help='initial learning rate')
    parser.add_argument('--swag_lr', type=float, default=1e-5, help='SWA LR')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='SGD momentum (default: 0.9)')

    parser.add_argument('--swag_start', type=float, default=1,
                        metavar='N', help='SWA start epoch number')

    parser.add_argument('--ilr', type=float, default=1e-4,
                        metavar='N', help='learning rate for interaction')

    parser.add_argument('--wd', type=float, default=1e-2,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--epochs', type=int, default=3,
                        metavar='N', help='SWA start epoch number')

    parser.add_argument('--noise', type=float, default=0.3, metavar='N')                    
    parser.add_argument('--proportion', type=float, default=1.0,
                        metavar='N', help='the proportion of data used to train')

    parser.add_argument('--swag_c_epochs', type=int, default=1,
                        metavar='N', help='SWA start epoch number')
    parser.add_argument('--batch_size', type=int, default=16,
                        metavar='N', help='input batch size')
    parser.add_argument('--pretrained_model', type=str, default='bert-base-uncased',
                        metavar='N', help='the name of pretrained_model')
    parser.add_argument('--mode', type=str, default='train',
                        metavar='N', help='dataset')
    parser.add_argument('--test_mode', type=str, default='test',
                        metavar='N', help='dataset')

    parser.add_argument('--sample_nums', type=int, default=20, metavar='N')
    parser.add_argument('--model_name', type=str,
                        default='vanilla_bert')
    parser.add_argument('--querier_type', type=str, default='imp', metavar='N')
    parser.add_argument('--stop_epochs', type=int, default=3, metavar='N')
    parser.add_argument('--start_q', type=int, default=0, metavar='N')
    parser.add_argument('--num_q', type=int, default=None, metavar='N')

    parser.add_argument('--interactive', type=bool, default=False, metavar='N')
    parser.add_argument('--do_train', type=bool, default=None, metavar='N')
    parser.add_argument('--do_test', type=bool, default=None, metavar='N')

    args = parser.parse_args()

    warnings.filterwarnings('ignore')
    if args.interactive:
        # load data
        data_dir = os.path.join(args.cqa_dir, args.topic, args.test_mode)
        cqa_path = os.path.join(data_dir, 'qa_list.json')
        gold_file_dir = os.path.join(data_dir, 'gold_ans')
        ref_path = os.path.join(data_dir, 'ref_values.json')

        if os.path.exists(cqa_path):
            print('loading cqa data....')
            with open(cqa_path, 'r') as f:
                qa_list = json.load(f)

            with open(ref_path, 'r') as f:
                ref_values = json.load(f)
            assert len(qa_list) == len(ref_values)

        else:
            print('creating data...')
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            if not os.path.exists(gold_file_dir):
                os.mkdir(gold_file_dir)

            qa_list, ref_values = load_cqa(
                args.raw_datadir, args.topic, cqa_path=cqa_path, gold_dir=gold_file_dir, ref_path=ref_path, pretrained_model=args.pretrained_model, mode=args.test_mode, interactive=True)
            
    if args.do_train:
        data_pair = load_cqa(
            args.raw_datadir, args.topic, cqa_path=None, gold_dir=None, ref_path=None, mode=args.mode, pretrained_model=args.pretrained_model, interactive=False)
        dev_pair = load_cqa(
            args.raw_datadir, args.topic, cqa_path=None, gold_dir=None, ref_path=None, mode='valid', pretrained_model=args.pretrained_model, interactive=False)
        print('training data size', len(data_pair))
        print('dev data size', len(dev_pair))
        data_loader = DataLoader(PosNegDataset(
            data_pair[:int(len(data_pair)*args.proportion)], args.pretrained_model), batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(PosNegDataset(
            dev_pair, args.pretrained_model), batch_size=args.batch_size, shuffle=False)
    # select data(batch or single) according to query strategy, initial data 0.1%?
    # querier
    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))
    base_model = BertRanker(args.pretrained_model)
    base_model.to(device)


    if args.do_train:
        print('start training')

        if 'vanilla_bert' in args.model_name.lower():
            model = VanillaBert(base_model, lr=args.lr_init, ilr=args.ilr, epochs=args.epochs,
                             pretrained_model=args.pretrained_model, device=device, weight_decay=args.wd)
            model.to(device)
        elif 'swag_bert' in args.model_name.lower():
            model = BayesainBert(base_model=base_model, subspace='covariance', max_num_models=args.max_num_models,
                              device=device, lr_init=args.lr_init, momentum=args.momentum, weight_decay=args.wd,
                              swag_start=args.swag_start, swag_lr=args.swag_lr, swag_c_epochs=args.swag_c_epochs,
                              epochs=args.epochs, batch_size=args.batch_size, pretrained_model=args.pretrained_model)
            # model.to(device)
        if args.checkpoint:
            checkpoint = torch.load(os.path.join(
                args.save_dir, args.checkpoint+'.pt'))
        else:
            checkpoint = None
        model.train(data_loader, valid_loader=valid_loader, save_dir=args.save_dir,
                stop_epochs=args.stop_epochs, save_name=args.model_name, checkpoint =checkpoint)

    if args.interactive:
        print('interactive training...')
        if 'vanilla_bert' in args.model_name.lower():
            model = VanillaBert(base_model, lr=args.lr_init, ilr = args.ilr, epochs=args.epochs,
                             pretrained_model=args.pretrained_model, device=device, weight_decay=args.wd)
            checkpoint = torch.load(os.path.join(
                args.save_dir, args.topic, 'hyperparam', args.model_name+'best-val-acc-model'+'.pt'))
            model.to(device)
        elif 'swag_bert' in args.model_name.lower():
            model = BayesainBert(base_model=base_model, subspace='covariance', max_num_models=args.max_num_models,
                              device=device, lr_init=args.lr_init, momentum=args.momentum, weight_decay=args.wd,
                              swag_start=args.swag_start, swag_lr=args.swag_lr, swag_c_epochs=args.swag_c_epochs,
                              epochs=args.epochs, batch_size=args.batch_size, pretrained_model=args.pretrained_model)

            checkpoint = torch.load(os.path.join(
                args.save_dir, args.topic, 'hyperparam', args.model_name+'.pt'))
                # args.save_dir, args.topic, 'hyperparam', args.model_name+'-final-model'+'.pt'))

        active_learner(start_q=args.start_q, model=model, checkpoint=checkpoint, question_list=qa_list, n_iter_rounds=args.n_iter_rounds, ref_values=ref_values, topic=args.topic,
                       gold_file_dir=gold_file_dir, mode=args.mode, sample_nums=args.sample_nums, save_dir=args.save_dir, stop_epochs=args.stop_epochs, querier_type=args.querier_type, num_q=args.num_q, noise=args.noise, swag=('swag_bert' in args.model_name.lower()))
        
        


    if args.do_test:
        data_dir = os.path.join(args.cqa_dir, args.topic, args.test_mode)
        cqa_path = os.path.join(data_dir, 'qa_list.json')
        gold_file_dir = os.path.join(data_dir, 'gold_ans')
        ref_path = os.path.join(data_dir, 'ref_values.json')

        if os.path.exists(cqa_path):
            print('loading cqa data....')
            with open(cqa_path, 'r') as f:
                test_qa_list = json.load(f)
            with open(ref_path, 'r') as f:
                test_ref_values = json.load(f)
        else:
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            if not os.path.exists(gold_file_dir):
                os.mkdir(gold_file_dir)

            test_qa_list, test_ref_values = load_cqa(
                args.raw_datadir, args.topic, cqa_path=cqa_path, gold_dir=gold_file_dir, ref_path=ref_path, pretrained_model=args.pretrained_model, mode=args.test_mode, interactive=True)

        print('start testing...')

        checkpoint = torch.load(os.path.join(
                args.save_dir, args.topic, 'hyperparam', args.model_name+'best-val-acc-model'+'.pt'))
        if 'swag_bert' in args.model_name.lower():
            model = BayesainBert(base_model=base_model, subspace='covariance', max_num_models=args.max_num_models,
                              device=device, lr_init=args.lr_init, momentum=args.momentum, weight_decay=args.wd,
                              swag_start=args.swag_start, swag_lr=args.swag_lr, swag_c_epochs=args.swag_c_epochs,
                              epochs=args.epochs, batch_size=args.batch_size, pretrained_model=args.pretrained_model)
            
        elif 'vanilla_bert' in args.model_name.lower():
            print('vallina bert!!')
            model = VanillaBert(base_model, lr=args.lr_init, ilr = args.ilr, epochs=args.epochs,
                             device=device, pretrained_model=args.pretrained_model, weight_decay=args.wd)

        model.load_state_dict(checkpoint['state_dict'])
        print('finish loading model!')
        res, acc = 0, 0
        whole_answers, question, cumcnt = [], [], []
        prev = 0
        for question_id in trange(len(test_qa_list)):
            entry = test_qa_list[question_id]
            pooled = entry['pooled_answers']
            question.extend([entry['question']]*len(pooled))
            whole_answers.extend(pooled)
            prev += len(pooled)
            cumcnt.append(prev)
        print('total nums', len(whole_answers))
        assert len(question) == len(whole_answers)
        utilities = model.get_utilities(
            test_data=whole_answers, question=question, sample_nums=args.sample_nums)
        for i in trange(len(cumcnt)):
            if i == 0:
                single_utility = utilities[:cumcnt[i]]
            else:
                single_utility = utilities[cumcnt[i-1]:cumcnt[i]]
            gold_values = test_ref_values[str(i)]
            assert len(single_utility) == len(gold_values)
            single_utility = normaliseList(single_utility)
            acc += (np.argmax(single_utility) == np.argmax(gold_values))
            metric_dict = evaluateReward(single_utility, gold_values)
            res += metric_dict['ndcg_at_5%']
        # print('res......',res)
        print(res/len(test_qa_list))
        print('acc', acc/len(test_qa_list))
