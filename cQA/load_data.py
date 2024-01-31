import json
import pandas as pd
import os
import csv
import numpy as np
import tqdm
import re
from rouge.rouge import Rouge
from resources import ROUGE_DIR, BASE_DIR
from utils.misc import normaliseList
from tqdm import tqdm, trange

def construct_dataset(start_ix, batch_size, questions, vocab,  gold_file_dir, topic, mode, answers, data):
    print(f'running on {os.getpid()}')
    qa_list = []
    total_ref_values = {}
    if not os.path.exists(gold_file_dir):
        os.makedirs(gold_file_dir)
    index = 0
    for i in trange(len(data.index)):
    # for i in trange(start_ix, start_ix + batch_size):
        qid = data.index[i]
        # Reconstruct the text sequences for the training questions
        # tokids = questions.loc[qid].values[0].split(' ')
        tokids = questions.loc[qid,1].split(' ')
        toks = vocab[np.array(tokids).astype(int)]
        question = ' '.join(toks)
        question = re.sub('<[^<]+>', "", question)

        candidate_ans_id = data.loc[qid, 'ansids'].split(' ')
        if len(candidate_ans_id) == 1:
            print(f'Skip question {i} with bad data')
            continue

        # Reconstruct the text sequences for the true answers
        gold_ans_id = data.loc[qid,'goldid']

        # some of the lines seem to have two gold ids. Just use the first.
        gold_ans_ids = gold_ans_id.split(' ')
        gold_ans_id = gold_ans_ids[0]

        # tokids = answers.loc[gold_ans_id].values[0].split(' ')
        tokids = answers.loc[gold_ans_id, 1].split(' ')
        toks = vocab[np.array(tokids).astype(int)]
        gold_ans = ' '.join(toks)
        gold_ans = re.sub('<[^<]+>', "", gold_ans)

        gold_an_path = os.path.join(gold_file_dir,f'gold_ans_{topic}_{mode}_{i}.txt')
        with open(gold_an_path,'w') as f:
            f.writelines(gold_ans)

        # Join the sequences. Insert '[SEP]' between the two sequences
        # qa_gold = question + ' [SEP] ' + gold_ans
        # candidate_ans_id.remove(gold_ans_id)
        candidate_ans = []
        for c in candidate_ans_id:
            tokids = answers.loc[c, 1].split(' ')
            toks = vocab[np.array(tokids).astype(int)]
            candidate = " ".join(toks)
            candidate = re.sub('<[^<]+>', "", candidate)
            candidate_ans.append(candidate)

        ref_values = cal_ref_values(question_id=i, gold_filename=gold_an_path, pool_answers=candidate_ans)
        total_ref_values[index] = ref_values
        index += 1
        qa_list.append({'question':question, 'pooled_answers': candidate_ans, 'gold_answer':gold_ans})

    return qa_list, total_ref_values

def construct_pairwise_dataset(questions, vocab, answers, data, pretrained_model, n_neg_samples=10):
    """
    Function for constructing a pairwise training set where each pair consists of a matching QA sequence and a
    non-matching QA sequence.
    :param n_neg_samples: Number of pairs to generate for each question by sampling non-matching answers and pairing
    them with matching answers.
    :param dataframe:
    :return:
    """
    # Get the positive (matching) qs and as from traindata and put into pairs
    # Sample a number of negative (non-matching) qs and as from the answers listed for each question in traindata

    qa_pairs = []

    for i, qid in enumerate(data.index):
        # Reconstruct the text sequences for the training questions
        # tokids = questions.loc[qid].values[0].split(' ')
        tokids = questions.loc[qid,1].split(' ')
        toks = vocab[np.array(tokids).astype(int)]
        question = ' '.join(toks)
        
        # Reconstruct the text sequences for the true answers
        gold_ans_id = data.loc[qid,'goldid']

        # some of the lines seem to have two gold ids. Just use the first.
        gold_ans_ids = gold_ans_id.split(' ')
        gold_ans_id = gold_ans_ids[0]

        # tokids = answers.loc[gold_ans_id].values[0].split(' ')
        tokids = answers.loc[gold_ans_id, 1].split(' ')
        toks = vocab[np.array(tokids).astype(int)]
        gold_ans = ' '.join(toks)
        
       
        # Join the sequences. Insert '[SEP]' between the two sequences
        qa_gold = (question, gold_ans)

        # Reconstruct the text sequences for random wrong answers
        wrong_ans_ids = data.loc[qid]["ansids"]
        wrong_ans_ids = wrong_ans_ids.split(' ')
        if len(wrong_ans_ids) < n_neg_samples + 1:
            continue

        if n_neg_samples == 0:
            # use all the wrong answers (exclude the gold one that is mixed in)
            n_wrongs = len(wrong_ans_ids) - 1
            widx = 0
        else:
            # use a specified sample size
            n_wrongs = n_neg_samples

        qa_wrongs = []
        while len(qa_wrongs) < n_wrongs:
            if n_neg_samples == 0:
                # choose the next wrong answer, skip over the gold answer.
                wrong_ans_id = wrong_ans_ids[widx]
                widx += 1
                if wrong_ans_id == gold_ans_id:
                    wrong_ans_id = wrong_ans_ids[widx]
                    widx += 1
            else:
                # choose a new negative sample
                wrong_ans_id = gold_ans_id
                while wrong_ans_id == gold_ans_id:
                    wrong_ans_id = wrong_ans_ids[np.random.randint(len(wrong_ans_ids))]

            tokids = answers.loc[wrong_ans_id, 1].split(' ')
            toks = vocab[np.array(tokids).astype(int)]
            wrong_ans = ' '.join(toks)

            qa_wrong = (question, wrong_ans)
            qa_wrongs.append(qa_wrong)
            qa_pairs.append((qa_gold, qa_wrong))

    # data = next(iter(data_loader))
    return qa_pairs


def load_cqa(datadir, topic, cqa_path, gold_dir, ref_path, mode,  pretrained_model, interactive): 
    
    # load the vocab
    vocab = pd.read_csv(os.path.join(datadir, topic+'.stackexchange.com', 'vocab.tsv'), sep='\t', quoting=csv.QUOTE_NONE, header=None,
                        index_col=0, names=['tokens'], dtype=str, keep_default_na=False)["tokens"].values

    # load the questions
    questions = pd.read_csv(os.path.join(datadir, topic+'.stackexchange.com', 'questions.tsv'), sep='\t', header=None, index_col=0)

    # load the answers
    answers = pd.read_csv(os.path.join(datadir, topic+'.stackexchange.com', 'answers.tsv'), sep='\t', header=None, index_col=0)

    # Load the training set
    data = pd.read_csv(os.path.join(datadir, topic+'.stackexchange.com', mode+'.tsv'), sep='\t', header=None, names=['goldid', 'ansids'],
                            index_col=0)
    if interactive:
        tr_qa_list, ref_values = construct_dataset(0, 0, questions, vocab, gold_dir, topic, mode, answers, data)

        # tr_qa_list, ref_values = [], {}
        # batch_size = 300
        # if len(data) % batch_size:
        #     num_batch = len(data) // batch_size + 1
        # else:
        #     num_batch = len(data) // batch_size 

        # pool = multiprocessing.Pool(processes =4)
        # params = [(start_ix, batch_size, questions, vocab, gold_dir, topic, mode, answers, data) for start_ix in range(num_batch)]
        # res = pool.starmap_async(construct_dataset, params)
        # if res.get()[0]:
        #     for partition in res.get():
        #         tr_qa_list.extend(partition[0])
        #         ref_values.update(partition[1])
        # pool.close()
        # pool.join()

    else:
        return construct_pairwise_dataset(questions, vocab, answers, data, pretrained_model)
    
    with open(cqa_path,'w') as f:
        json.dump(tr_qa_list, f, ensure_ascii=False)
    print('Saved qa list.json ...')

    with open(ref_path,'w') as f:
        json.dump(ref_values, f, ensure_ascii=False)

    return tr_qa_list, ref_values

    
def compute_ref_values(aidx, answer, gold_filename):
    answer = re.sub('<[^<]+>', "", answer)
    # if np.mod(aidx, 20) == 0:
        # print('computing ref value for answer: %i' % aidx)
    rouge_scorer = Rouge(ROUGE_DIR, BASE_DIR, True)
    R1, R2, RL, RSU = rouge_scorer(answer, [[gold_filename, None]], len(answer))
    rouge_scorer.clean()

    return RL


def cal_ref_values(question_id, gold_filename, pool_answers):
    # ref_filename = ref_dir + '/ref_vals_rougel_lno03_%s_%i.txt' % \
                                # (topic, question_id)

    # if not os.path.exists(ref_filename):
    ref_values = []

    if len(pool_answers) == 1:
        print('SKIPPING A QUESTION WITH BAD DATA')
        print(f'question id is: {question_id}')
        return 
        
    ref_values = Parallel(n_jobs=25, backend='threading')(
        delayed(compute_ref_values)(aidx, answer, gold_filename)
        for aidx, answer in tqdm(enumerate(pool_answers)))

    ref_values = normaliseList(ref_values)

    # with open(ref_filename, 'w') as fh:
        # json.dump(ref_values, fh)
    # else:
    #     with open(ref_filename, 'r') as fh:
    #         ref_values = json.load(fh)
    return ref_values

if __name__ == '__main__':

    cqa_datadir = r'/Users/hsfang/Library/Mobile Documents/com~apple~CloudDocs/workspace/Bayesian deep learning/tacl2020-interactive-ranking/data/cqa_data/apple.stackexchange.com'
    load_cqa(cqa_datadir)