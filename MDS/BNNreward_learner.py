import torch
from BNN_prefLearning import BNNPrefLearning
#from BNN_prefLearning import SiameseBERTModel
import logging
import numpy as np
import sys


# class BNNRewardLearner:

#     def __init__(self, input_dim, n_threads=0, heuristics=None, rate=200, lspower=1):
#         #self.model = SiameseBERTModel(input_dim)
#         self.model = BNNPrefLearning(input_dim=1024)
#         #self.model.to(device)
#         self.n_labels_seen = 0
#         self.n_threads = n_threads
#         self.rewards = None
#         self.summary_vectors = None  # Add this line

#     def train(self, pref_log, summary_vectors, epochs=60, learning_rate=0.0001):
#         print("pref_log:", pref_log)
#         #print(np.array(summary_vectors).shape)

#         #print("summary_vectors:", summary_vectors)
#         self.summary_vectors = summary_vectors
        
#         items1_coords = [summary_vectors[pref[0][0]] for pref in pref_log]
#         items2_coords = [summary_vectors[pref[0][1]] for pref in pref_log]

#         #items2_coords = [pref[0][1] for pref in pref_log]
#         new_labels = [1 - pref[1] for pref in pref_log]  # Convert to BNN format

#         logging.debug('BNN fitting with %i pairwise labels' % len(new_labels))

#         self.model.fit(items1_coords, items2_coords, new_labels, epochs=epochs, learning_rate=learning_rate)
#         self.n_labels_seen = len(pref_log)

#         logging.debug(f'BNN trained with {self.n_labels_seen} labels.')

#         # After training, predict rewards for all summaries in the database
#         all_summary_coords = summary_vectors
#         self.rewards, self.reward_var = self.model.predict(all_summary_coords, all_summary_coords, n_samples=40, return_var=True)
#         logging.debug('...rewards obtained.')


#     def get_rewards(self):
#         # Predict the reward for each summary vector
#         if self.rewards is None:
#             raise ValueError("The model has not been trained yet. Please call the 'train' method before getting rewards.")
#         return self.rewards.detach().numpy().flatten()

#     def predictive_var(self, candidate_idxs=None):
#         # If candidate_idxs is provided, get the summary vectors for the candidates
#         if candidate_idxs is not None:
#             candidate_coords_0 = [self.summary_vectors[idx] for idx in candidate_idxs]
#             candidate_coords_1 = [self.summary_vectors[idx] for idx in candidate_idxs]
#         else:
#             candidate_coords_0 = self.summary_vectors
#             candidate_coords_1 = self.summary_vectors

#         _, var_scores = self.model.predict(candidate_coords_0, candidate_coords_1, n_samples=40, return_var=True)
    
#         return var_scores.detach().numpy().flatten()

#     def predictive_cov(self, candidate_idxs=None):
#         # If candidate_idxs is provided, get the summary vectors for the candidates
#         if candidate_idxs is not None:
#             candidate_coords_0 = [self.summary_vectors[idx] for idx in candidate_idxs]
#             candidate_coords_1 = [self.summary_vectors[idx] for idx in candidate_idxs]
#         else:
#             candidate_coords_0 = self.summary_vectors
#             candidate_coords_1 = self.summary_vectors

#         _, cov_scores = self.model.predict(candidate_coords_0, candidate_coords_1, n_samples=30, return_cov=True)
    
#         return cov_scores.detach().numpy().flatten()
    


import numpy as np
import logging

class BNNRewardLearner:
    """
    Bayesian Neural Network Reward Learner.
    """

    def __init__(self, input_dim: int, n_threads: int = 0, heuristics=None, rate: int = 200, lspower: int = 1):
        """
        Initialize the BNNRewardLearner.
        """
        self._model = BNNPrefLearning(input_dim=1024)
        self._n_labels_seen = 0
        self._n_threads = n_threads
        self._rewards = None
        self._summary_vectors = None

    def train(self, pref_log, summary_vectors, epochs: int = 60, learning_rate: float = 0.0001):
        """
        Train the model.
        """
        self._summary_vectors = summary_vectors
        items1_coords, items2_coords, new_labels = self._prepare_training_data(pref_log)

        logging.debug(f'BNN fitting with {len(new_labels)} pairwise labels')
        self._model.fit(items1_coords, items2_coords, new_labels, epochs=epochs, learning_rate=learning_rate)
        self._n_labels_seen = len(pref_log)
        logging.debug(f'BNN trained with {self._n_labels_seen} labels.')

        self._compute_rewards()

    def _prepare_training_data(self, pref_log):
        """
        Prepare training data from preference log.
        """
        items1_coords = [self._summary_vectors[pref[0][0]] for pref in pref_log]
        items2_coords = [self._summary_vectors[pref[0][1]] for pref in pref_log]
        new_labels = [1 - pref[1] for pref in pref_log]  # Convert to BNN format
        return items1_coords, items2_coords, new_labels

    def _compute_rewards(self):
        """
        Compute rewards after training.
        """
        all_summary_coords = self._summary_vectors
        self._rewards, self._reward_var = self._model.predict(all_summary_coords, all_summary_coords, n_samples=40, return_var=True)
        logging.debug('...rewards obtained.')

    def get_rewards(self) -> np.ndarray:
        """
        Get the computed rewards.
        """
        if self._rewards is None:
            raise ValueError("The model has not been trained yet. Please call the 'train' method before getting rewards.")
        return self._rewards.detach().numpy().flatten()

    def predictive_var(self, candidate_idxs=None) -> np.ndarray:
        """
        Get predictive variance.
        """
        candidate_coords_0, candidate_coords_1 = self._get_candidate_coords(candidate_idxs)
        _, var_scores = self._model.predict(candidate_coords_0, candidate_coords_1, n_samples=40, return_var=True)
        return var_scores.detach().numpy().flatten()

    def predictive_cov(self, candidate_idxs=None) -> np.ndarray:
        """
        Get predictive covariance.
        """
        candidate_coords_0, candidate_coords_1 = self._get_candidate_coords(candidate_idxs)
        _, cov_scores = self._model.predict(candidate_coords_0, candidate_coords_1, n_samples=30, return_cov=True)
        return cov_scores.detach().numpy().flatten()

    def _get_candidate_coords(self, candidate_idxs):
        """
        Helper method to get candidate coordinates.
        """
        if candidate_idxs is not None:
            candidate_coords_0 = [self._summary_vectors[idx] for idx in candidate_idxs]
            candidate_coords_1 = [self._summary_vectors[idx] for idx in candidate_idxs]
        else:
            candidate_coords_0 = self._summary_vectors
            candidate_coords_1 = self._summary_vectors
        return candidate_coords_0, candidate_coords_1


#so how do we calculate covariance in our case the BNN model we have made?