import random
from utils.misc import sigmoid


class SimulatedUser:
    def __init__(self, ref_values, m=2.5):
        self.ref_values = ref_values
        self.temperature = m

    def getPref(self, question_id, idx1, idx2):
        question_id = str(question_id)
        # print('idx1 = %i, idx2 = %i, available reference vals =%i' % (idx1, idx2, len(self.ref_values)))
        # prob = sigmoid(ref_values[0]-ref_values[1], self.temperature)
        prob = sigmoid(self.ref_values[question_id][idx1]-self.ref_values[question_id][idx2], self.temperature)
        # prob = sigmoid(self.ref_values[question_id][idx1]-self.ref_values[question_id][idx2], self.temperature)

        if random.random() <= prob:
            return 1  # summary1 is preferred
        else:
            return -1  # summary2 is preferred

