import os
import pandas as pd
import json

# Paths
PROJECT_FILEPATH = os.getcwd()
DATA_FILEPATH = os.path.join(PROJECT_FILEPATH, 'data')
CORPUS_FILEPATH = os.path.join(DATA_FILEPATH, 'corpus_jobs.jsonl')

LABEL_ENCODERS_FILEPATH = os.path.join(PROJECT_FILEPATH, 'label_encoders')
VECTORIZERS_FILEPATH = os.path.join(PROJECT_FILEPATH, 'vectorizers')
MODEL_FILEPATH = os.path.join(PROJECT_FILEPATH, 'model')

DEV_GOLD_ANSWERS_FILEPATH = os.path.join(DATA_FILEPATH, 'dev_gold_answers.json')
with open(DEV_GOLD_ANSWERS_FILEPATH, 'r') as fin:
	dev_gold_answers = json.load(fin)

CORPUS_PAIRS_FILEPATH = os.path.join(DATA_FILEPATH, 'corpus_pairs.json')
DEV_JOB_QUERIES_FILEPATH = os.path.join(DATA_FILEPATH, 'dev_job_queries.jsonl')
TEST_JOB_QUERIES_FILEPATH = os.path.join(DATA_FILEPATH, 'test_job_queries.jsonl')

# Column lists
TEXT_COLS = ['title', 'description', 'requirement_summary', 'benefit_summary', 'user_keywords', 'soft_skills',
			 'technical_skills']
COLUMNS_WITH_ARRAYS = ['user_keywords', 'soft_skills', 'technical_skills']
COLUMNS_TO_DELETE = ['created_at', 'industry', 'benefit_summary']
COLUMNS_TO_ENUMERATE = ['employment_type', 'function', 'education', 'collar_color']
COLUMNS_TO_COUNT_FREQUENCIES = ['title', 'description', 'requirement_summary', 'user_keywords', 'soft_skills',
								'technical_skills']
COLUMNS_TO_NORMALIZE = ['account_id', 'employment_type', 'function', 'education', 'collar_color']

# Used for evaluating on validation set and tuning our final model.
N_ITERS = [3, 5, 7]
BEST_ITER = 5

# Dictionaries for easy mapping of job id to company, title & description
corpus_df = pd.read_json(CORPUS_FILEPATH, lines=True)
dev_df = pd.read_json(DEV_JOB_QUERIES_FILEPATH, lines=True)
test_df = pd.read_json(TEST_JOB_QUERIES_FILEPATH, lines=True)

all_ids = []
all_titles = []
all_descriptions = []
for temp_df in [corpus_df, dev_df, test_df]:
	all_ids.extend(list(temp_df.job_id.values))
	all_titles.extend(list(temp_df.title.values))
	all_descriptions.extend(list(temp_df.description.values))

job_dict = {}
for _id, _title, _description in zip(all_ids, all_titles, all_descriptions):
	job_dict[_id] = {}
	job_dict[_id]['title'] = _title
	job_dict[_id]['description'] = _description

corpus_job_ids = list(corpus_df.job_id.values)
dev_job_ids = pd.read_json(DEV_JOB_QUERIES_FILEPATH, lines=True)
dev_job_ids = dev_job_ids['job_id'].tolist()

test_job_ids = pd.read_json(TEST_JOB_QUERIES_FILEPATH, lines=True)
test_job_ids = test_job_ids['job_id'].tolist()

del temp_df