from tqdm import trange
from queries.random_querier import RandomQuerier
import numpy as np
from scipy.stats import norm


class PairUncQuerier(RandomQuerier):
    '''
    Used for Non-Bayesian methods
    '''
    def __init__(self, reward_learner_class, qa_lists, **kwargs):
        super().__init__(reward_learner_class, qa_lists, **kwargs)
        self.unc = True

    def _compute_pairwise_scores(self, f, var, _):
        prob = 1/(1+np.exp(-f))
        prob[prob>0.5] = 1-prob[prob>0.5]
        uncertainty = prob[:, None] + prob[None, :]
        uncertainty[range(uncertainty.shape[0]), range(uncertainty.shape[0])] = -np.inf
        return uncertainty, np.argmax(prob)

    def _get_candidates(self, mean, var):
        # v = self.reward_learner.predictive_var()

        # # consider only the top most uncertain items
        num = 100
        candidate_idxs = np.argsort(mean)[-num:]

        return candidate_idxs


    def getQuery(self, log, question_id, sample_num):

        # get the current best estimate
        test_data = self.qa_lists[question_id]['pooled_answers']
        question = self.qa_lists[question_id]['question']
        questions = [question]*len(test_data)

        pooled_mean, pooled_var = self.reward_learner.predict(test_data, questions, sample_num, eval=self.unc)
        # candidate_idxs = self._get_candidates(pooled_mean, pooled_var)
        # best_idx = np.argmax(pooled_mean)
        # print(best_idx)
        # exit()
        # pairwise_entropy, best_ix = self._compute_pairwise_scores(pooled_mean, pooled_var)
        pairwise_entropy, best_ix = self._compute_pairwise_scores(pooled_mean, pooled_var, log[question_id])
        # print('ei', pairwise_entropy)

        # Find out which of our candidates have been compared already
        for data_point in log[question_id]:

            pairwise_entropy[data_point[0], data_point[1]] = -np.inf
            pairwise_entropy[data_point[1], data_point[0]] = -np.inf
            
        select = np.argmax(pairwise_entropy[best_ix, :])
        pe_selected = pairwise_entropy[best_ix, select]
        if select == best_ix:
            print('entropy',pairwise_entropy)
            print(pairwise_entropy[best_ix,:])
            exit()
        selected = (best_ix, select)
        print(f'Chosen candidate: {selected[1]}, vs. best: {selected[0]}, with score = {pe_selected}')
        return selected[0], selected[1]



