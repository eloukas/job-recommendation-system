import streamlit as st
from libs.helper import *
from run import get_similarity_matrix, get_recommendations, prepare_dataset
import re
import time
import pickle

TOTAL_SECONDS = 1
TOTAL_TICKS = 4

# Load model & corpus
best_model_filename = f'svd_{BEST_ITER}.pkl'
best_model_filename = os.path.join(MODEL_FILEPATH, best_model_filename)

corpus_reduced_dims_filepath = f'processed_corpus_reduced_dims_{BEST_ITER}.csv'
corpus_reduced_dims_filepath = os.path.join(DATA_FILEPATH, corpus_reduced_dims_filepath)


def clean_html(raw_html):
	cleantext = re.sub(re.compile('<.*?>'), '', raw_html)
	return cleantext


st.write("""
# 💼  Job Recommender
""")

query_id = st.text_input(label='Enter the job query ID:',
						 help='Enter the id of the job query you want to get recommendations for. '
							  'Possible values are all the integers between 0 and 23054.',
						 placeholder='22744')

if (query_id):
	try:
		query_id = int(query_id)
	except ValueError:
		st.write("Job query id not an integer. Please try all the integers between 0 and 23054.")
		exit()

	if query_id < 0 or query_id > 23054:
		st.write("Job query id not in range. Please try  all the integers between 0 and 23054.")
		exit()
	else:

		progress_bar = st.progress(0)
		for i in range(TOTAL_TICKS + 1):
			progress_bar.progress(i * TOTAL_SECONDS / TOTAL_TICKS)
			time.sleep(TOTAL_SECONDS / TOTAL_TICKS)

		st.header(f"{job_dict[query_id]['title']}")
		st.caption(f"{clean_html(job_dict[query_id]['description'])}")

		recommend_button = st.button(label='Recommend me a similar job 👌')

		if recommend_button:

			# TODO: Find row of query example in train/dev/test dfs
			# dataset =

			corpus_df = pd.read_csv(corpus_reduced_dims_filepath)

			with open(best_model_filename, 'rb') as fin:
				svd = pickle.load(fin)

			# Transform inference dataset
			infer_df = pd.DataFrame(svd.transform(dataset))

			job_ids = dev_job_ids if mode == 'dev' else test_job_ids

			# For each job id vector in the inference set, count its cosine similarity
			dataframe_used_for_similarity = corpus_df.copy()
			predictions_dict = dict()
			for iter_tuple in zip(infer_df.iterrows(), job_ids):
				row = iter_tuple[0][1]
				job_id = iter_tuple[1]

				dataframe_used_for_similarity.loc[dataframe_used_for_similarity.shape[0]] = list(row.values)
				similarity_matrix = get_similarity_matrix(dataframe_used_for_similarity)

				# Append to the final dictionary
				# TODO: Modify get recommendations to return the title, cosine similarity, and string
				predictions_dict[str(job_id)] = get_recommendations(similarity_matrix, job_id)

				# Drop added row & reiterate
				dataframe_used_for_similarity.drop(dataframe_used_for_similarity.tail(1).index, inplace=True)
