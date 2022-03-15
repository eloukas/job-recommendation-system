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
corpus_df_reduced_dims = pd.read_csv(corpus_reduced_dims_filepath)


def find_df(job_id):
	# Find dataframe which the query belong sto
	for df in [corpus_df, dev_df, test_df]:
		if not df[df.job_id == job_id].empty:
			break

	# Return the row as a new isolated dataframe
	return df[df.job_id == job_id]


def clean_html(raw_html):
	cleantext = re.sub(re.compile('<.*?>'), '', raw_html)
	return cleantext


st.write("""
# 💼  Job Recommender
""")

query_id = st.text_input(label='Enter the job query ID:',
						 help='Enter the id of the job query you want to get recommendations for. '
							  'Possible values are all the integers between 0 and 23054.',
						 placeholder='22984')

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

		if query_id in corpus_job_ids:
			job_ids = corpus_job_ids
		elif query_id in dev_job_ids:
			job_ids = dev_job_ids
		else:
			job_ids = test_job_ids

		progress_bar = st.progress(0)
		for i in range(TOTAL_TICKS + 1):
			progress_bar.progress(i * TOTAL_SECONDS / TOTAL_TICKS)
			time.sleep(TOTAL_SECONDS / TOTAL_TICKS)

		st.header(f"{job_dict[query_id]['title']}")
		st.caption(f"{clean_html(job_dict[query_id]['description'])}")

		recommend_button = st.button(label='Recommend me a similar job 👌')

		if recommend_button:

			df = find_df(query_id)

			# TODO: Debug 142 line at run.py -- dataset returns 2 rows
			df = prepare_dataset(df, 'test')

			with open(best_model_filename, 'rb') as fin:
				svd = pickle.load(fin)

			# Transform inference dataset
			infer_df = pd.DataFrame(svd.transform(df))

			# For each job id vector in the inference set, count its cosine similarity
			dataframe_used_for_similarity = corpus_df_reduced_dims.copy()
			list_to_display = []
			for _, row in infer_df.iterrows():
				job_id = query_id
				# Append query job to last row of the corpus df
				dataframe_used_for_similarity.loc[dataframe_used_for_similarity.shape[0]] = list(row.values)
				similarity_matrix = get_similarity_matrix(dataframe_used_for_similarity)

				# Append to the final dictionary
				# TODO: Modify get recommendations to return the title, cosine similarity, and string
				list_to_display = get_recommendations(similarity_matrix, job_id, demo=True)
				dataframe_used_for_similarity.drop(dataframe_used_for_similarity.tail(1).index, inplace=True)

				st.header(f"{list_to_display[1]}")
				st.caption(f"{clean_html(list_to_display[2])}")

				# Drop added row & reiterate

