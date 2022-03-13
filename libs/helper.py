import os
import pandas as pd

# Paths
PROJECT_FILEPATH = os.getcwd()
DATA_FILEPATH = os.path.join(PROJECT_FILEPATH, 'data')
CORPUS_FILEPATH = os.path.join(DATA_FILEPATH, 'corpus_jobs.jsonl')

LABEL_ENCODERS_FILEPATH = os.path.join(PROJECT_FILEPATH, 'label_encoders')
VECTORIZERS_FILEPATH = os.path.join(PROJECT_FILEPATH, 'vectorizers')
MODEL_FILEPATH = os.path.join(PROJECT_FILEPATH, 'model')

DEV_GOLD_ANSWERS = os.path.join(DATA_FILEPATH, 'dev_gold_answers.jsonl')
CORPUS_GOLD_ANSWERS = os.path.join(DATA_FILEPATH, ' corpus_pairs')

# Column lists
TEXT_COLS = ['title', 'description', 'requirement_summary', 'benefit_summary', 'user_keywords', 'soft_skills',
             'technical_skills']
COLUMNS_WITH_ARRAYS = ['user_keywords', 'soft_skills', 'technical_skills']
COLUMNS_TO_DELETE = ['created_at', 'industry', 'benefit_summary']
COLUMNS_TO_ENUMERATE = ['employment_type', 'function', 'education', 'collar_color']
COLUMNS_TO_COUNT_FREQUENCIES = ['title', 'description', 'requirement_summary', 'user_keywords', 'soft_skills',
                                'technical_skills']
COLUMNS_TO_NORMALIZE = ['account_id', 'employment_type', 'function', 'education', 'collar_color']

# Dictionaries for easy mapping of job id to company, title & description
corpus_df = pd.read_json(CORPUS_FILEPATH, lines=True)
job_to_company_id = dict(zip(list(corpus_df.job_id.values), list(corpus_df.account_id.values)))
job_to_title = dict(zip(list(corpus_df.job_id.values), list(corpus_df.title.values)))
job_to_description = dict(zip(list(corpus_df.job_id.values), list(corpus_df.description.values)))
del corpus_df
