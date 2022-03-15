from ir_measures import AP, Rprec
import ir_measures
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


def get_similarity_matrix(dataset):
	"""
	Calculates and returns the cosine similarity matrix
	:param dataset: the dataset to calculate similarity
	:return: the cosine similarity matrix
	"""
	similarity_matrix = cosine_similarity(dataset, dataset)
	similarity_matrix_df = pd.DataFrame(similarity_matrix)

	return similarity_matrix_df


# TODO: Fine-tune n jobs to return
def get_recommendations(similarity_matrix, job_id, demo=False, jobs_to_return=1):
	"""
	Returns the top n similar jobs
	:param similarity_matrix: the similarity matrix
	:param job_id: the query job id
	:param demo: boolean to indicate if the recommendations should be printed. Use it only for reporting purposes.
	:param jobs_to_return: the number of jobs to return
	:return: a dict with the recommended jobs
	"""

	# Used for showcasing input/output
	if demo:
		print(f"Input Query:")
		print(f"Title: {job_dict[job_id]['title']}")
		print(f"Description: {job_dict[job_id]['description']}\n\n")

	last_idx = similarity_matrix.tail(1).index  # The query is represented by the last row on the last row of the data
	temp_df = similarity_matrix.iloc[last_idx].T
	temp_df.rename(columns={temp_df.columns[0]: 'given_job_id'}, inplace=True)

	recommended_jobs_df = temp_df.nlargest(jobs_to_return + 1, 'given_job_id')  # Get top n similar jobs
	list_with_similar_job_ids = list(recommended_jobs_df.iloc[1:].index)  # Discard itself

	dict_with_relevant_jobs = dict()

	# Return similarity scores, titles, descriptions
	similarity_scores = [item for sublist in recommended_jobs_df.iloc[1:].values.tolist() for item in sublist]

	if demo: print(f'Recommended Jobs:')

	for current_id, sim_score in zip(list_with_similar_job_ids, similarity_scores):

		# Used for showcasing input/output
		if demo:
			print(f"Cosine Similarity Score: {sim_score}")
			print(f"Title: {job_dict[current_id]['title']}")
			print(f"Description: {job_dict[current_id]['description']}\n")

		# TODO: Fine-tune similarity score threshold
		if not dict_with_relevant_jobs and sim_score < 0.70:
			continue

		dict_with_relevant_jobs[str(current_id)] = round(sim_score * 10, 4)

	return dict_with_relevant_jobs


def normalize(dataset):
	"""
	Normalizes columns in a 0,1 range
	:param dataset: the dataset
	:return: the normalized dataset
	"""
	for col in COLUMNS_TO_NORMALIZE:
		scaler = MinMaxScaler()
		dataset[col] = scaler.fit_transform(dataset[[col]])

	return dataset


def get_tfidf(dataset, mode):
	"""
	Vectorizes text columns
	:param dataset: the dataset
	:param mode: <train>, <dev>, <test>. In train mode, the vectorizers are saved. Otherwise, they get loaded from disk.
	:return: the vectorized dataframe
	"""

	merged_df = dataset.copy()

	# Create directory to save objects
	if not os.path.exists(VECTORIZERS_FILEPATH):
		os.mkdir(VECTORIZERS_FILEPATH)

	# For text columns to be vectorized
	for col in COLUMNS_TO_COUNT_FREQUENCIES:

		vectorizer_filename = 'vectorize_' + col + '.pickle'
		vectorizer_filename = os.path.join(VECTORIZERS_FILEPATH, vectorizer_filename)

		# Replace NaNs with empty string
		dataset[col] = dataset[col].fillna('')

		if mode == 'train':

			# TODO: Fine-tune threshold on dev set
			if col in ['title', 'user_keywords']:
				max_df = 0.99
				tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 1), max_df=max_df)
			elif col in ['description', 'requirement_summary', 'soft_skills', 'technical_skills']:
				max_df = 0.95
				min_df = 0.05
				tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 1), max_df=max_df, min_df=min_df)

			# Fit with TFIDF object and transform
			tfidf_matrix = tfidf.fit_transform(dataset[col])

			# Serialize vectorizer to disk
			with open(vectorizer_filename, 'wb') as fout:
				pickle.dump(tfidf, fout, pickle.HIGHEST_PROTOCOL)
		else:  # Inference mode (<dev> or <test>)

			# Load vectorizer from disk
			with open(vectorizer_filename, 'rb') as fin:
				tfidf = pickle.load(fin)

			# Transform data
			tfidf_matrix = tfidf.transform(dataset[col])

		# Return new dataset without text columns
		merged_df.drop(col, axis=1, inplace=True)  # Delete old column
		tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
		merged_df = pd.concat([merged_df, tfidf_df], axis=1)

	return merged_df


def label_encode(dataset, mode):
	"""
	Enumerates feature columns with low amount of unique values (like education or categories)
	:param dataset: the dataset
	:param mode: <train>, <dev>, <test>. In train mode, the encoders are saved. Otherwise, they get loaded from disk.
	:return: the dataset with integer-encoded columns
	"""

	# Create directory to save objects
	if not os.path.exists(LABEL_ENCODERS_FILEPATH):
		os.mkdir(LABEL_ENCODERS_FILEPATH)

	# For each column to be enumerated/integer-encoded
	for col in COLUMNS_TO_ENUMERATE:
		encoder_filename = 'encode_' + col + '.pickle'
		encoder_filename = os.path.join(LABEL_ENCODERS_FILEPATH, encoder_filename)

		if mode == 'train':  # Create label encoder, fit, transform, save
			le = preprocessing.LabelEncoder()
			dataset[col] = le.fit_transform(dataset[col].values)

			# Serialize vectorizer to disk
			with open(encoder_filename, 'wb') as fout:
				pickle.dump(le, fout, pickle.HIGHEST_PROTOCOL)

		else:  # Inference mode

			# Load vectorizer from disk
			with open(encoder_filename, 'rb') as fin:
				le = pickle.load(fin)

			# Transform columns
			dataset[col] = le.transform(dataset[col].values)

	return dataset


def drop_unnecessary_cols(dataset):
	"""
	Drops unnecessary columns
	:param dataset: the dataset
	:return: the dataset without job_id or account_id cols.
	"""

	dataset.drop('job_id', 1, inplace=True)
	dataset.drop('account_id', 1, inplace=True)
	dataset.fillna(0.0, inplace=True)

	return dataset


def stem_sentence(sentence):
	"""
	Stems a sentence using the Porter Stemmer
	:param sentence: the sentence to stem
	:return: the stemmed sentence
	"""
	porter_stemmer = PorterStemmer()
	tokens = sentence.split()
	stemmed_tokens = [porter_stemmer.stem(token) for token in tokens]
	return ' '.join(stemmed_tokens)


def clean_dataset(dataset):
	"""
	Cleans a dataset by converting NaNs and cleaning text columns
	:param dataset: the dataset as a dataframe
	:return: the cleaned dataset
	"""

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
		dataset[col] = dataset[col].apply(stem_sentence)  # Stem sentences

	# Delete unnecessary columns
	for col in COLUMNS_TO_DELETE:
		dataset.drop(col, axis=1, inplace=True)

	return dataset


def prepare_dataset(df, mode):
	"""
	Prepares a dataset through the preprocessing pipeline.
	Performs text cleaning, lowercasing, stemming, TF-IDF vectorization and [0-1] normalization
	:param df: the dataset as a dataframe
	:param mode: <train>, <dev>, or <test>. In train mode, encoders/vectorizers are saved into the disk
	:return: returns the prepared dataset
	"""
	dataset = clean_dataset(df)

	# Label encode columns with low number of unique values
	dataset = label_encode(dataset, mode)
	dataset = get_tfidf(dataset, mode)
	dataset = normalize(dataset)

	dataset = drop_unnecessary_cols(dataset)
	return dataset


@click.command()
@click.option('--dataset_filepath', default='./data/test_job_queries.jsonl')
@click.option('--mode', default='test', help='Pick between <train>, <dev>, <test>.')
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

		for n_iters in N_ITERS:
			# Build SVD model for dimensionality reduction
			print(f'Building SVD model with n_iters: {n_iters}')
			svd = TruncatedSVD(n_components=100, n_iter=n_iters, random_state=42)
			svd.fit(dataset)

			# Serialize model for dimensionality reduction
			model_filename = f'svd_{n_iters}.pkl'
			model_filename = os.path.join(MODEL_FILEPATH, model_filename)
			with open(model_filename, 'wb') as fout:  # Save the model
				pickle.dump(svd, fout)
			print(f'Saved model to {model_filename}')

			# Save the training corpus to load it from memory during inference phase
			# It will be used to compute similarities between jobs and generate recommendations
			corpus_reduced_dims_filepath = f'processed_corpus_reduced_dims_{n_iters}.csv'
			corpus_reduced_dims_filepath = os.path.join(DATA_FILEPATH, corpus_reduced_dims_filepath)
			dataset_reduced_dims = pd.DataFrame(svd.transform(dataset))
			dataset_reduced_dims.to_csv(corpus_reduced_dims_filepath, index=False)
			print(f'Saved corpus to {corpus_reduced_dims_filepath}')

	elif mode in ['dev', 'test']:

		# Load model & corpus
		best_model_filename = f'svd_{BEST_ITER}.pkl'
		best_model_filename = os.path.join(MODEL_FILEPATH, best_model_filename)

		corpus_reduced_dims_filepath = f'processed_corpus_reduced_dims_{BEST_ITER}.csv'
		corpus_reduced_dims_filepath = os.path.join(DATA_FILEPATH, corpus_reduced_dims_filepath)

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
			predictions_dict[str(job_id)] = get_recommendations(similarity_matrix, job_id)

			# Drop added row & reiterate
			dataframe_used_for_similarity.drop(dataframe_used_for_similarity.tail(1).index, inplace=True)

		# Serialize to [dev,test]_predictions.json
		predictions_json_filename = f'{mode}_predictions.json'
		predictions_json_filename = os.path.join(DATA_FILEPATH, predictions_json_filename)
		with open(predictions_json_filename, 'w') as fout:
			json.dump(predictions_dict, fout)

		print(f'Serialized predictions to {predictions_json_filename}')

		# If in dev, calculate the aggregated metric in order to choose the best hyperparameters
		if mode == 'dev':
			aggr_result = ir_measures.calc_aggregate([AP, Rprec], dev_gold_answers, predictions_dict)
			map_score = aggr_result[AP]
			rprec_score = aggr_result[Rprec]

			print(f'Avg of metrics: {(map_score + rprec_score) / 2}')

	print('[DONE]')

	return None


if __name__ == '__main__':
	main()
