import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm
import torch.nn.functional as F
from summariser.utils.misc import sigmoid


def process_data_for_bnn(obs_coords_0, obs_coords_1, mu_0=None, mu_1=None):
    """
    Combine two sets of coordinates and optionally their mean values.
    """
    combined_coords = np.concatenate((obs_coords_0, obs_coords_1), axis=0)
    unique_coords = np.unique(combined_coords, axis=0)
    
    if mu_0 is not None and mu_1 is not None:
        combined_mu = np.concatenate((mu_0, mu_1), axis=0)
        return unique_coords, combined_mu
    return unique_coords

def bnn_pref_likelihood(network_output, subset_idxs=[], v=[], u=[], return_g_f=False):
    """
    Calculate the likelihood of preferences based on the network's output.
    """
    if subset_idxs:
        pair_subset = np.in1d(v, subset_idxs) & np.in1d(u, subset_idxs) if v and u else []
        v, u = (v[pair_subset], u[pair_subset]) if pair_subset else (v, u)
        network_output = network_output[subset_idxs] if not pair_subset else network_output

    network_output = network_output[:, np.newaxis] if network_output.ndim < 2 else network_output
    g_f = network_output[v, :] - network_output[u, :] if v and u else network_output - network_output.T
    phi = norm.cdf(g_f)

    return (phi, g_f) if return_g_f else phi

def temper_extreme_probs(probs, zero_only=False):
    """
    Temper extreme probabilities to avoid numerical instability.
    """
    if not zero_only:
        probs[probs > 1 - 1e-7] = 1 - 1e-7
    probs[probs < 1e-7] = 1e-7
    return probs

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(num_heads)
        ])
        self.final_linear = nn.Linear(num_heads * input_dim, input_dim)

    def forward(self, x):
        head_outputs = [F.softmax(head(x), dim=1) * x for head in self.attention_heads]
        concatenated = torch.cat(head_outputs, dim=1)
        return self.final_linear(concatenated)

class BNNPrefLearning(nn.Module):
    """
    Preference learning with Bayesian Neural Network using MC Dropout.
    """
    def __init__(self, input_dim, dropout_p=0.1, num_heads=4):
        super(BNNPrefLearning, self).__init__()

        # Twin network
        self.twin_network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Multi-Headed Attention Mechanism
        self.multi_head_attention = MultiHeadAttention(128*2, num_heads)

        # Comparator
        self.comparator = nn.Sequential(
            nn.Linear(128*2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward_one(self, x):
        return self.twin_network(x)

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        
        # Concatenate the outputs
        combined = torch.cat((output1, output2), 1)
        
        # Apply multi-headed attention
        combined = self.multi_head_attention(combined)
        
        # Pass through the comparator
        result = self.comparator(combined)
        return result
    
# class BNNPrefLearning(nn.Module):
#     """
#     Preference learning with Bayesian Neural Network.
#     """
#     def __init__(self, input_dim, dropout_p=0.1):
#         super(BNNPrefLearning, self).__init__()

        # # Shared encoder
        # self.encoder = nn.Sequential(
        #     nn.Linear(input_dim, 512),
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout_p),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout_p),
        #     nn.Linear(256, 128),
        #     nn.ReLU()
        

    #     # Comparison layer
    #     self.comparator = nn.Sequential(
    #         nn.Linear(128, 64),
    #         nn.ReLU(),
    #         nn.Dropout(p=dropout_p),
    #         nn.Linear(64, 1),
    #         nn.Sigmoid()
    #     )

    # def forward_one(self, x):
    #     # Forward pass for one input
    #     x = self.encoder(x)
    #     return x

    # def forward(self, x1, x2):
    #     # Forward pass for both inputs
    #     out1 = self.forward_one(x1)
    #     out2 = self.forward_one(x2)
    #     # Difference between outputs
    #     diff = torch.abs(out1 - out2)
    #     # Pass through comparator
    #     pred = self.comparator(diff)
    #     return pred
    

    # # Twin network with Residual Connections
    #     self.twin_network = nn.Sequential(
    #         nn.Linear(input_dim, 512),
    #         nn.ReLU(),
    #         nn.Dropout(dropout_p),
    #         nn.Linear(512, 512),  # Additional layer for residual connection
    #         nn.ReLU(),
    #         nn.Dropout(dropout_p),
    #         nn.Linear(512, 256),
    #         nn.ReLU(),
    #         nn.Dropout(dropout_p),
    #         nn.Linear(256, 256),  # Additional layer for residual connection
    #         nn.ReLU(),
    #         nn.Dropout(dropout_p),
    #         nn.Linear(256, 128),
    #         nn.ReLU(),
    #         nn.Dropout(dropout_p)
    #     )

    #     # Attention Mechanism for the comparator
    #     self.attention = nn.Linear(128*2, 1, bias=False)

    #     # Comparator with Attention
    #     self.comparator = nn.Sequential(
    #         nn.Linear(128*2, 64),
    #         nn.ReLU(),
    #         nn.Dropout(dropout_p),
    #         nn.Linear(64, 1),
    #         nn.Sigmoid()
    #     )

    # def forward_one(self, x):
    #     return self.twin_network(x)

    # def forward(self, input1, input2):
    #     output1 = self.forward_one(input1)
    #     output2 = self.forward_one(input2)
        
    #     # Concatenate the outputs
    #     combined = torch.cat((output1, output2), 1)
        
    #     # Apply attention
    #     attention_weights = F.softmax(self.attention(combined), dim=0)
    #     combined = combined * attention_weights
        
    #     # Pass through the comparator
    #     result = self.comparator(combined)
    #     return result
    
    #  # Twin network
    #     self.twin_network = nn.Sequential(
    #         nn.Linear(input_dim, 256),
    #         nn.ReLU(),
    #         nn.Dropout(dropout_p),
    #         nn.Linear(256, 128),
    #         nn.ReLU()
    #     )

    #     # Attention Mechanism for the comparator
    #     self.attention = nn.Linear(128*2, 1, bias=False)

    #     # Comparator with Attention
    #     self.comparator = nn.Sequential(
    #         nn.Linear(128*2, 64),
    #         nn.ReLU(),
    #         nn.Linear(64, 1),
    #         nn.Sigmoid()
    #     )

    # def forward_one(self, x):
    #     return self.twin_network(x)

    # def forward(self, input1, input2):
    #     output1 = self.forward_one(input1)
    #     output2 = self.forward_one(input2)
        
    #     # Concatenate the outputs
    #     combined = torch.cat((output1, output2), 1)
        
    #     # Apply attention
    #     attention_weights = F.softmax(self.attention(combined), dim=1)
    #     combined = combined * attention_weights
        
    #     # Pass through the comparator
    #     result = self.comparator(combined)
    #     return result

# import torch.nn as nn
# import transformers

# class SiameseBERTModel(nn.Module):
#     def __init__(self, feature_dim=1024, bert_model_name="bert-base-uncased", dropout_p=0.1):
#         super(SiameseBERTModel, self).__init__()

#         # Linear layer to map 1024-dimensional vectors to 768 dimensions
#         self.embedding_mapper = nn.Linear(feature_dim, 768)

#         # Load pre-trained BERT model
#         self.bert = transformers.BertModel.from_pretrained(bert_model_name)

#         # Dropout layer
#         self.dropout = nn.Dropout(dropout_p)

#         # Comparison layer (here we're using subtraction, but other methods can be used)
#         self.comparison = lambda x, y: x - y

#         # Fully connected layers for final prediction
#         self.fc1 = nn.Linear(768, 100)
#         self.fc2 = nn.Linear(100, 10)
#         self.fc3 = nn.Linear(10, 1)
#         self.activation = nn.ReLU()

#     def forward(self, feature_vector1, feature_vector2):
#         # Map the feature vectors to 768 dimensions
#         feature_vector1 = self.embedding_mapper(feature_vector1)
#         feature_vector2 = self.embedding_mapper(feature_vector2)
#         # Bypass the embedding layer and use the summary vectors as embeddings
#         # Pass through BERT
#         bert_output1 = self.bert(inputs_embeds=feature_vector1).last_hidden_state[:, 0, :]
#         bert_output2 = self.bert(inputs_embeds=feature_vector2).last_hidden_state[:, 0, :]

#         # Apply dropout
#         bert_output1 = self.dropout(bert_output1)
#         bert_output2 = self.dropout(bert_output2)

#         # Compare the two outputs
#         comparison_result = self.comparison(bert_output1, bert_output2)

#         # Pass through fully connected layers
#         x = self.activation(self.fc1(comparison_result))
#         x = self.dropout(x)  # Apply dropout after activation
#         x = self.activation(self.fc2(x))
#         x = self.dropout(x)  # Apply dropout after activation
#         x = self.fc3(x)

#         return x

    #  # Define the twin network architecture
    #     self.twin_network = nn.Sequential(
    #         nn.Linear(input_dim, 100),
    #         nn.ReLU(),
    #         nn.Dropout(dropout_p),
    #         nn.Linear(100, 10),
    #         nn.ReLU(),
    #         nn.Dropout(dropout_p)
    #     )

    #     # Comparator
    #     self.comparator = nn.Sequential(
    #         nn.Linear(10*2, 64),  # Concatenating the two vectors
    #         nn.ReLU(),
    #         nn.Dropout(dropout_p),
    #         nn.Linear(64, 1),
    #         nn.Sigmoid()
    #     )

    # def forward_one(self, x):
    #     return self.twin_network(x)

    # def forward(self, input1, input2):
    #     output1 = self.forward_one(input1)
    #     output2 = self.forward_one(input2)
        
    #     # Concatenate the outputs
    #     combined = torch.cat((output1, output2), 1)
        
    #     # Pass through the comparator
    #     result = self.comparator(combined)
    #     return result
    
    #     # Neural network architecture with dropout for MC Dropout
    #     self.fc1 = nn.Linear(input_dim, 512)
    #     self.drop1 = nn.Dropout(p=dropout_p)
    #     self.fc2 = nn.Linear(512, 256)
    #     self.drop2 = nn.Dropout(p=dropout_p)
    #     self.fc3 = nn.Linear(256, 1)
        
    #     self.activation = nn.ReLU()
    #     self.output_activation = nn.Sigmoid()

    # def forward(self, x):
    #     x = self.drop1(self.activation(self.fc1(x)))
    #     x = self.drop2(self.activation(self.fc2(x)))
    #     x = self.output_activation(self.fc3(x))
    #     return x

# import torch
# import torch.nn as nn
# import transformers

# class SiameseBERTModel(nn.Module):
#     def __init__(self, feature_dim=1024, roberta_model_name="distilroberta-base", dropout_p=0.1):
#         super(SiameseBERTModel, self).__init__()

#         # Linear layer to map 1024-dimensional vectors to 768 dimensions
#         self.embedding_mapper = nn.Linear(feature_dim, 768)

#         # Load pre-trained DistilRoberta model
#         self.roberta = transformers.RobertaModel.from_pretrained(roberta_model_name)

#         # Dropout layer
#         self.dropout = nn.Dropout(dropout_p)

#         # Fully connected layers for final prediction
#         self.fc1 = nn.Linear(768, 100)
#         self.fc2 = nn.Linear(100, 10)
#         self.fc3 = nn.Linear(10, 1)
#         self.activation = nn.ReLU()

#     def forward(self, feature_vector1, feature_vector2):
#         # Map the feature vectors to 768 dimensions
#         feature_vector1 = self.embedding_mapper(feature_vector1)
#         feature_vector2 = self.embedding_mapper(feature_vector2)

#         # Pass through DistilRoberta
#         roberta_output1 = self.roberta(inputs_embeds=feature_vector1).last_hidden_state[:, 0, :]
#         roberta_output2 = self.roberta(inputs_embeds=feature_vector2).last_hidden_state[:, 0, :]

#         # Apply dropout
#         roberta_output1 = self.dropout(roberta_output1)
#         roberta_output2 = self.dropout(roberta_output2)

#         # Pass through fully connected layers to get utility scores
#         utility_score1 = self.fc3(self.dropout(self.activation(self.fc2(self.dropout(self.activation(self.fc1(roberta_output1)))))))
#         utility_score2 = self.fc3(self.dropout(self.activation(self.fc2(self.dropout(self.activation(self.fc1(roberta_output2)))))))

#         # Compare the utility scores
#         comparison_result = utility_score1 - utility_score2

#         return comparison_result

    # def process_input_data(self, obs_coords_0, obs_coords_1):
    #     obs_coords_0 = torch.tensor(np.array(obs_coords_0), dtype=torch.float32).unsqueeze(1)  # Add sequence length dimension
    #     obs_coords_1 = torch.tensor(np.array(obs_coords_1), dtype=torch.float32).unsqueeze(1)  # Add sequence length dimension
    #     return obs_coords_0, obs_coords_1

    def process_input_data(self, obs_coords_0, obs_coords_1):
        obs_coords_0 = torch.tensor(np.array(obs_coords_0), dtype=torch.float32)  # Add sequence length dimension
        obs_coords_1 = torch.tensor(np.array(obs_coords_1), dtype=torch.float32)  # Add sequence length dimension
        return obs_coords_0, obs_coords_1
    
    def fit(self, items1_coords, items2_coords, preferences, epochs=10, learning_rate=0.0001, weight_decay=0.001, margin=0.5):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.MarginRankingLoss(margin=margin)
        items1_coords, items2_coords = self.process_input_data(items1_coords, items2_coords)
        preferences = torch.tensor(np.array(preferences) * 2 - 1, dtype=torch.float32).unsqueeze(1)

        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = self(items1_coords, items2_coords)
            loss = criterion(predictions, torch.ones_like(predictions), preferences)
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    def predict(self, test_coords_0, test_coords_1, n_samples=30, return_var=True):
        test_coords_0, test_coords_1 = self.process_input_data(test_coords_0, test_coords_1)
        
        self.train()
        
        samples = [self(test_coords_0, test_coords_1) for _ in range(n_samples)]
        samples = torch.stack(samples, dim=0)
        
        mean_scores = torch.mean(samples, dim=0)
        var_scores = torch.var(samples, dim=0)
        
        if return_var:
            return mean_scores, var_scores
        else:
            return mean_scores


    # #def forward_model(self, obs_coords_0, obs_coords_1):
    #  #   combined_input = torch.cat((obs_coords_0, obs_coords_1), dim=1)
    #   #  preference_score = self(combined_input)
    #    # return preference_score
    

    # def forward_model(self, obs_coords_0, obs_coords_1):
    #     preference_score_0 = self(obs_coords_0)
    #     preference_score_1 = self(obs_coords_1)
    
    #     # Compute the difference in preference scores
    #     preference_difference = preference_score_0 - preference_score_1
    
    #     # Compute the preference likelihood based on the difference
    #     phi = torch.sigmoid(preference_difference)
    
    #     return phi
    
    # def forward_model(self, obs_coords_0, obs_coords_1):
    #     preference_score_0 = self(obs_coords_0)
    #     preference_score_1 = self(obs_coords_1)
    #     return preference_score_0, preference_score_1

    # def fit(self, items1_coords, items2_coords, preferences, epochs=10, learning_rate=0.001, weight_decay=0.01, margin=0.1):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #     criterion = nn.MarginRankingLoss(margin=margin)
    #     items1_coords, items2_coords = self.process_input_data(items1_coords, items2_coords)
    #     #items1_coords = torch.tensor(np.array(items1_coords), dtype=torch.float32)
    #     #items2_coords = torch.tensor(np.array(items2_coords), dtype=torch.float32)
    #     preferences = torch.tensor(np.array(preferences) * 2 - 1, dtype=torch.float32).unsqueeze(1)

    #     for epoch in range(epochs):
    #         optimizer.zero_grad()
    #         predictions = self(items1_coords, items2_coords)
    #         loss = criterion(predictions, torch.ones_like(predictions), preferences)
    #         loss.backward()
    #         optimizer.step()

    #         print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    # def fit(self, items1_coords, items2_coords, preferences, epochs=10, learning_rate=0.001, weight_decay=0.01, margin=0.1):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
    #     # Using BCELoss
    #     criterion = nn.BCELoss()

    #     # Using MarginRankingLoss
    #     #criterion = nn.MarginRankingLoss(margin=margin)

    #     items1_coords, items2_coords = self.process_input_data(items1_coords, items2_coords)
    #     # Convert preferences to [-1, 1] where -1 indicates the first item is preferred and 1 indicates the second item is preferred
    #     preferences = torch.tensor(np.array(preferences) * 2 - 1, dtype=torch.float32).unsqueeze(1)

    #     for epoch in range(epochs):
    #         optimizer.zero_grad()
            
    #         preference_scores = self.forward_model(items1_coords, items2_coords)
    #         #preference_score_0, preference_score_1 = self.forward_model(items1_coords, items2_coords)
            
    #         loss = criterion(preference_scores, preferences)
    #         #loss = criterion(preference_score_0, preference_score_1, preferences)
            
    #         # Print the loss and current epoch during training
    #         print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
        
    #         # Get mean scores and uncertainties for the training data
    #         #mean_scores, uncertainties = self.predict(items1_coords, items2_coords)
        
    #         # Print the mean score and uncertainty
    #         #print(f"Mean Score: {mean_scores.mean().item()}, Uncertainty: {uncertainties.mean().item()}")

    # def predict(self, test_coords_0, test_coords_1, n_samples=20, return_var=True):
    #     test_coords_0, test_coords_1 = self.process_input_data(test_coords_0, test_coords_1)
        
    #     self.train()
        
    #     samples = [self(test_coords_0, test_coords_1) for _ in range(n_samples)]
    #     samples = torch.stack(samples, dim=0)
        
    #     mean_scores = torch.mean(samples, dim=0)
    #     var_scores = torch.var(samples, dim=0)
        
    #     if return_var:
    #         return mean_scores, var_scores
    #     else:
    #         return mean_scores
        
    # def predict(self, test_coords_0, test_coords_1, n_samples=20, return_var=True):
    #     test_coords_0, test_coords_1 = self.process_input_data(test_coords_0, test_coords_1)
    
    #     self.train()
    
    #     samples = [self.forward_model(test_coords_0, test_coords_1) for _ in range(n_samples)]
    #     preference_score_0, preference_score_1 = zip(*samples)
    #     preference_score_0 = torch.stack(preference_score_0, dim=0)
    #     preference_score_1 = torch.stack(preference_score_1, dim=0)
    
    #     preference_difference = preference_score_0 - preference_score_1
    #     phi = torch.sigmoid(preference_difference)
    
    #     mean_scores = torch.mean(phi, dim=0)
    #     var_scores = torch.var(phi, dim=0)
    
    #     if return_var:
    #          # Flatten the mean_scores before returning
    #         return mean_scores, var_scores
    #     else:
    #         return mean_scores

    def _logpt(self):
        preference_scores = self.forward_model(self.items1_coords, self.items2_coords)
        return F.logsigmoid(preference_scores), F.logsigmoid(1 - preference_scores)

    def _post_rough(self, test_coords_0, test_coords_1, n_samples=20):
        mean_scores, var_scores = self.predict(test_coords_0, test_coords_1, n_samples=n_samples, return_var=True)
        mean_scores = temper_extreme_probs(mean_scores)
        not_mean_scores = 1 - mean_scores
        return mean_scores, not_mean_scores, var_scores
    
    def _post_sample(self, n_samples=20):
        self.train()
        
        samples = [self.forward_model(self.items1_coords, self.items2_coords) for _ in range(n_samples)]
        samples = torch.stack(samples, dim=0)
        
        mean_scores = torch.mean(samples, dim=0)
        var_scores = torch.var(samples, dim=0)
        
        return mean_scores, var_scores

# Ensure to move the model to the appropriate device before training or inference:
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = BNNPrefLearning(input_dim=1024).to(device)

