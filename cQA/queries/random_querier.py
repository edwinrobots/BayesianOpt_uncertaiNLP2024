import numpy as np
import random
from queries.utils import normaliseList
from sklearn.metrics.pairwise import cosine_similarity


class RandomQuerier:
    def __init__(self, reward_learner_class, qa_lists):

        self.reward_learner = reward_learner_class

        # use a random sample to initialise the AL process, then apply the AL strategy to subsequent iterations.
        # This flag only affects the classes that inherit from this class since the random querier always chooses
        # randomly
        self.qa_lists  = qa_lists
        self.random_initial_sample = True

    def inLog(self, sum1, sum2, log):
        if (sum1,sum2) in log:
            return True
        elif (sum2, sum1) in log:
            return True
        # for entry in log:
        #     if [sum1, sum2] in entry:
        #         return True
        #     elif [sum2, sum1] in entry:
        #         return True
        return False

    # def _get_good_and_dissimilar_pair(self):
    #     # find two distinctive items as an initial sample. To limit complexity, first choose the item with
    #     # strongest heuristic:
    #     first_item = np.argmax(self.heuristics)

    #     # now compare this item to the others according to the feature vectors:
    #     sims = cosine_similarity(self.summary_vectors[first_item][None, :],
    #                              self.summary_vectors)
    #     second_item = np.argmin(sims)

    #     return first_item, second_item

    def getQuery(self, log, question_id, sample_nums):
        # if self.reward_learner.n_labels_seen == 0 and not self.random_initial_sample:
        #     return self._get_good_and_dissimilar_pair()

        summary_num = len(self.qa_lists[question_id]['pooled_answers'])
        if summary_num  == 1:
            print('SKIPPING A QUESTION WITH BAD DATA')
            print(f'question id is: {question_id}')
            return 
        rand1 = random.randint(0, summary_num - 1)
        rand2 = random.randint(0, summary_num - 1)
        ### ensure the sampled pair has not been queried before
        while rand2 == rand1 or self.inLog(rand1, rand2, log[question_id]):
            rand1 = random.randint(0, summary_num - 1)
            rand2 = random.randint(0, summary_num - 1)

        return rand1, rand2

    def updateRanker(self, pref_log, stop_epochs):
        '''
        pref_log: {q_id:{(c1,c2):label,...},...}
        pref_log:[(question_id, candidate1_id, candidate2_id, label),...]
        '''
        # qa_list:[{question:..., pooled_answers:[..., ...], gold_answer:...}]
        trained_data = []
        for q in pref_log:
            q_dict = self.qa_lists[q]
            question = q_dict['question']
            for cand in pref_log[q]:
                cand1 = q_dict['pooled_answers'][cand[0]]
                cand2 = q_dict['pooled_answers'][cand[1]]
                trained_data.append([question, cand1, cand2, pref_log[q][cand]])

        # for candidate in pref_log:
        #     cand1 = q_dict['pooled_answers'][cand[0]]
        #     cand2 = q_dict['pooled_answers'][cand[1]]
        #     trained_data.append([question, cand1, cand2, pref_log[q][cand]])

        self.reward_learner.update(trained_data, stop_epochs)

    def getReward(self):
        values = self.reward_learner.get_rewards()
        return normaliseList(values)

    def get_utilities(self, question_id):
        test_data = self.qa_lists[question_id]['pooled_answers']
        question = self.qa_lists[question_id]['question']
        questions = [question]*len(test_data)
        return self.reward_learner.get_utilities(test_data, questions)