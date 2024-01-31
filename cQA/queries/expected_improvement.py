import numpy as np
from queries.pairuncquerier import PairUncQuerier
from scipy.stats import norm
from itertools import chain


class ExpectedImprovementQuerier(PairUncQuerier):

    def __init__(self, reward_learner_class, qa_lists, **kwargs):
        super().__init__(reward_learner_class, qa_lists, **kwargs)
        self.unc = False


    # def _get_candidates(self, mean, var):
        
    #     # consider only the top ranked items.
    #     num = 100
    #     candidate_idxs = np.argsort(mean)[-num:]

    #     return candidate_idxs

    # def _compute_pairwise_scores(self, mean, var):
    #     best_idx = np.argmax(mean)
    #     mean_best = mean[best_idx]

    #     # for all candidates, compute u = (mu - f_best) / sigma
    #     u = (mean - mean_best) / np.sqrt(var) # mean improvement. Similar to preference likelihood, but that adds in 2
    #     # due to labelling noise
    #     cdf_u = norm.cdf(u) # probability of improvement
    #     pdf_u = norm.pdf(u) #
    #     E_improvement = np.sqrt(var) * (u * cdf_u + pdf_u)
    #     E_improvement[best_idx] = -np.inf

    #     # make it back into a matrix
    #     E_imp_mat = np.zeros((mean.size, mean.size))
    #     E_imp_mat[best_idx, :] = E_improvement
    #     return E_imp_mat, best_idx

    def _compute_pairwise_scores(self, mean, var, seen_points):
        if seen_points:    
            points =  list(chain(*seen_points))
            best_ix = np.argmax(mean[points])
            best_idx = points[best_ix]
        else:
            best_idx = np.argmax(mean)
        mean_best = mean[best_idx]
        # for all candidates, compute u = (mu - f_best) / sigma
        u = (mean - mean_best) / np.sqrt(var) # mean improvement. Similar to preference likelihood, but that adds in 2
        # due to labelling noise
        cdf_u = norm.cdf(u) # probability of improvement
        pdf_u = norm.pdf(u) #
        E_improvement = np.sqrt(var) * (u * cdf_u + pdf_u)
        E_improvement[best_idx] = -np.inf

        # make it back into a matrix
        E_imp_mat = np.zeros((mean.size, mean.size))
        E_imp_mat[best_idx, :] = E_improvement
        return E_imp_mat, best_idx