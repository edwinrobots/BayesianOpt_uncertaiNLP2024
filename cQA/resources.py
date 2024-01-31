import os

#BASE_DIR = os.path.abspath('.')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROUGE_DIR = os.path.join(BASE_DIR,'rouge','ROUGE-RELEASE-1.5.5/') #do not delete the '/' at the end
PROCESSED_PATH = os.path.join(BASE_DIR,'data','processed_data')
# SUMMARY_DB_DIR = os.path.join(BASE_DIR,'data','sampled_summaries')
# DOC_SEQUENCE_PATH = os.path.join(BASE_DIR,'utils','DocsSequence.txt')

LANGUAGE = 'english'