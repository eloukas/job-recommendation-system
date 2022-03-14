from ir_measures import AP, Rprec
import ir_measures

import json
import os
import pandas as pd
from dataprep.eda import *
from nltk.stem import PorterStemmer
from libs.helper import *
import os
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
import click
from dataprep.clean import clean_text
import pickle


# def reduce_dimensionality(dataset, mode):
# 	model_filename = os.path.join(MODEL_FILEPATH, 'svd.pkl')
# 	corpus_reduced_dims_filepath = os.path.join(DATA_FILEPATH, 'processed_corpus_reduced_dims.csv')
#
# 	if mode == 'train':
# 		svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
# 		svd.fit(dataset)
#
# 		with open(model_filename, 'wb') as fout:  # Save the model
# 			pickle.dump(svd, fout)
#
# 		# Save the training corpus
# 		# It will be used in dev/test mode later to compute similarities between jobs and generate recommendations
# 		dataset_reduced_dims = pd.DataFrame(svd.transform(dataset))
# 		dataset_reduced_dims.to_csv(corpus_reduced_dims_filepath, index=False)
# 	else:
#
# 		with open(model_filename, 'rb') as fin:
# 			svd = pickle.load(fin)
#
# 		corpus_df = pd.read_csv(corpus_reduced_dims_filepath)
# 		infer_df = pd.DataFrame(svd.transform(dataset))
#
# 		if mode == 'dev':
# 			pass
# 		# TODO: Fine-tune params
# 		elif mode == 'test':
# 			pass
#
# 	dataset_reduced_dims = pd.DataFrame(svd.transform(dataset))
# 	similarity_matrix_reduced_dims = pd.DataFrame(
# 		cosine_similarity(train_dataset_reduced_dims, train_dataset_reduced_dims))


def get_similarity_matrix(dataset):
	similarity_matrix = cosine_similarity(dataset, dataset)
	similarity_matrix_df = pd.DataFrame(similarity_matrix)

	return similarity_matrix_df


# TODO: Print samples from train/dev/test
def get_recommendations(similarity_matrix, input_id, jobs_to_return=1):
	# print(f"Input:")
	# print(f"Title: {job_dict[input_id]['title']}")
	# print(f"Description: {job_dict[input_id]['description']}\n\n")

	last_idx = similarity_matrix.tail(1).index
	temp_df = similarity_matrix.iloc[last_idx].T
	temp_df.rename(columns={temp_df.columns[0]: 'given_job_id'}, inplace=True)

	recommended_jobs_df = temp_df.nlargest(jobs_to_return + 1, 'given_job_id')  # Get top n similar jobs
	list_with_similar_job_ids = list(recommended_jobs_df.iloc[1:].index)  # Discard itself

	dict_with_relevant_jobs = dict()
	# Return similarity scores, titles, descriptions
	similarity_scores = [item for sublist in recommended_jobs_df.iloc[1:].values.tolist() for item in sublist]
	for current_id, sim_score in zip(list_with_similar_job_ids, similarity_scores):

		# print(f'Current id: {current_id}')
		# print(f"Cosine Similarity Score: {sim_score}")
		# print(f"Title: {job_dict[current_id]['title']}")
		# print(f"Description: {job_dict[current_id]['description']}\n")

		# TODO: Fine-tune it
		if (not dict_with_relevant_jobs and sim_score < 0.70):
			continue

		dict_with_relevant_jobs[str(current_id)] = round(sim_score * 10, 4)

	return dict_with_relevant_jobs


def normalize(dataset):
	for col in COLUMNS_TO_NORMALIZE:
		scaler = MinMaxScaler()
		dataset[col] = scaler.fit_transform(dataset[[col]])

	return dataset


def get_tfidf(dataset, mode):
	merged_df = dataset.copy()

	if not os.path.exists(VECTORIZERS_FILEPATH):
		os.mkdir(VECTORIZERS_FILEPATH)

	for col in COLUMNS_TO_COUNT_FREQUENCIES:

		vectorizer_filename = 'vectorize_' + col + '.pickle'
		vectorizer_filename = os.path.join(VECTORIZERS_FILEPATH, vectorizer_filename)

		dataset[col] = dataset[col].fillna('')

		if mode == 'train':

			if col in ['title', 'user_keywords']:
				max_df = 0.99
				tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 1), max_df=max_df)
			elif col in ['description', 'requirement_summary', 'soft_skills', 'technical_skills']:
				max_df = 0.95
				min_df = 0.05
				tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 1), max_df=max_df, min_df=min_df)

			tfidf_matrix = tfidf.fit_transform(dataset[col])

			with open(vectorizer_filename, 'wb') as fout:
				pickle.dump(tfidf, fout, pickle.HIGHEST_PROTOCOL)
		else:
			with open(vectorizer_filename, 'rb') as fin:
				tfidf = pickle.load(fin)

			tfidf_matrix = tfidf.transform(dataset[col])

		merged_df.drop(col, axis=1, inplace=True)  # Delete old column
		tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
		merged_df = pd.concat([merged_df, tfidf_df], axis=1)

	return merged_df


def label_encode(dataset, mode):
	if not os.path.exists(LABEL_ENCODERS_FILEPATH):
		os.mkdir(LABEL_ENCODERS_FILEPATH)

	for col in COLUMNS_TO_ENUMERATE:
		encoder_filename = 'encode_' + col + '.pickle'
		encoder_filename = os.path.join(LABEL_ENCODERS_FILEPATH, encoder_filename)

		if mode == 'train':  # Create label encoder, fit, transform, save
			le = preprocessing.LabelEncoder()
			dataset[col] = le.fit_transform(dataset[col].values)

			with open(encoder_filename, 'wb') as fout:
				pickle.dump(le, fout, pickle.HIGHEST_PROTOCOL)

		else:  # Load label encoder, fit, transform, save
			with open(encoder_filename, 'rb') as fin:
				le = pickle.load(fin)

			dataset[col] = le.transform(dataset[col].values)

	return dataset


def drop_unnecessary_cols(dataset):
	dataset.drop('job_id', 1, inplace=True)
	dataset.drop('account_id', 1, inplace=True)
	dataset.fillna(0.0, inplace=True)

	return dataset


def stem_sentences(sentence):
	porter_stemmer = PorterStemmer()
	tokens = sentence.split()
	stemmed_tokens = [porter_stemmer.stem(token) for token in tokens]
	return ' '.join(stemmed_tokens)


def clean_dataset(dataset):
	# Convert NaNs to empty strings so they can be processed
	for col in COLUMNS_WITH_ARRAYS:
		dataset[col] = dataset[col].fillna("")
		dataset[col] = [','.join(map(str, l)) for l in dataset[col]]

	# Clean text data
	# Perform removal of HTML, URLs, punctuation, accents, stopwords, extra whitespaces
	# Convert to lowercase
	# Stem the tokens
	for col in TEXT_COLS:
		dataset = clean_text(dataset, col)
		dataset[col].fillna('', inplace=True)  # Replace NaNs with empty strings
		dataset[col] = dataset[col].apply(stem_sentences)  # Stem sentences

	# Delete unnecessary columns
	for col in COLUMNS_TO_DELETE:
		dataset.drop(col, axis=1, inplace=True)

	return dataset


def prepare_dataset(df, mode):
	dataset = clean_dataset(df)

	# Label encode columns with low number of unique values
	dataset = label_encode(dataset, mode)
	dataset = get_tfidf(dataset, mode)
	dataset = normalize(dataset)

	dataset = drop_unnecessary_cols(dataset)
	return dataset


@click.command()
@click.option('--dataset_filepath', default='./data/dev_job_queries.jsonl')
@click.option('--mode', default='dev', help='Pick between <train>, <dev>, <test>.')
def main(dataset_filepath, mode):
	print('[START]')
	print(f'--dataset_filepath: {dataset_filepath}')
	print(f'--mode: {mode}\n')
	if mode not in ['train', 'dev', 'test']:
		print(f'Mode is not <train> or <inference>. Exiting..')
		exit()

	if not os.path.exists(dataset_filepath):
		print(f'The dataset path you entered does not exist. Exiting..')
		exit()

	# Read dataset
	df = pd.read_json(dataset_filepath, lines=True)

	# Prepare dataset: clean, encode, get tf-idf vectors, normalize & delete unwanted cols
	dataset = prepare_dataset(df, mode)

	if mode == 'train':

		N_ITERS = [3, 5, 7]  # Used for evaluating on validation set and tuning our final model.
		for n_iters in N_ITERS:
			print(f'Building SVD model with n_iters: {n_iters}')
			svd = TruncatedSVD(n_components=100, n_iter=n_iters, random_state=42)
			svd.fit(dataset)

			model_filename = f'svd_{n_iters}.pkl'
			model_filename = os.path.join(MODEL_FILEPATH, model_filename)
			with open(model_filename, 'wb') as fout:  # Save the model
				pickle.dump(svd, fout)
			print(f'Saved model to {model_filename}')

			# Save the training corpus
			# It will be used in dev/test mode later to compute similarities between jobs and generate recommendations
			corpus_reduced_dims_filepath = f'processed_corpus_reduced_dims_{n_iters}.csv'
			corpus_reduced_dims_filepath = os.path.join(DATA_FILEPATH, corpus_reduced_dims_filepath)
			dataset_reduced_dims = pd.DataFrame(svd.transform(dataset))
			dataset_reduced_dims.to_csv(corpus_reduced_dims_filepath, index=False)
			print(f'Saved corpus to {corpus_reduced_dims_filepath}')

		job_ids = corpus_job_ids
	else:
		# TODO: Put dev flag here
		best_iter = 3
		best_model_filename = f'svd_{best_iter}.pkl'
		best_model_filename = os.path.join(MODEL_FILEPATH, best_model_filename)

		corpus_reduced_dims_filepath = f'processed_corpus_reduced_dims_{best_iter}.csv'
		corpus_reduced_dims_filepath = os.path.join(DATA_FILEPATH, corpus_reduced_dims_filepath)

		with open(best_model_filename, 'rb') as fin:
			svd = pickle.load(fin)

		corpus_df = pd.read_csv(corpus_reduced_dims_filepath)
		infer_df = pd.DataFrame(svd.transform(dataset))

		job_ids = dev_job_ids if mode == 'dev' else test_job_ids

		# if mode == 'dev':

		dataframe_used_for_similarity = corpus_df.copy()
		predictions_dict = dict()
		for iter_tuple in zip(infer_df.iterrows(), job_ids):
			row = iter_tuple[0][1]
			job_id = iter_tuple[1]
			dataframe_used_for_similarity.loc[dataframe_used_for_similarity.shape[0]] = list(row.values)
			similarity_matrix = get_similarity_matrix(dataframe_used_for_similarity)

			predictions_dict[str(job_id)] = get_recommendations(similarity_matrix, job_id)
			dataframe_used_for_similarity.drop(dataframe_used_for_similarity.tail(1).index,
											   inplace=True)  # Drop added row

		dev_predictions_json_filename = 'dev_predictions.json'
		dev_predictions_json_filename = os.path.join(DATA_FILEPATH, dev_predictions_json_filename)
		with open(dev_predictions_json_filename, 'w') as fout:
			json.dump(predictions_dict, fout)

		aggr_result = ir_measures.calc_aggregate([AP, Rprec], dev_gold_answers, predictions_dict)
		map_score = aggr_result[AP]
		rprec_score = aggr_result[Rprec]

		print(f'Avg of metrics: {(map_score + rprec_score) / 2}')

	if mode == 'dev':
		pass
	elif mode == 'test':
		pass
	# TODO: Put test flag here. Move most of the dev code here except the metrics one.

	print('[DONE]')
	return None


if __name__ == '__main__':
	main()
