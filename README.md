# job-recommendation-system
Get recommendations of similar jobs when job hunting.

The system uses unsupervised learning to generate the recommendations. More specifically, the pipeline includes:
- Extensive data preprocessing pipeline (HTML/URL/stopwords removal, lowercasing, stemming)
- Integer Encoding 
- TF-IDF text vectorization
- SVD dimensionality reduction
- Cosine similarity metric 

## Example
![Test Automation Lead Example](/images/demo_lead_software_tester.gif)

## Install
- Before starting, ideally, it's recommended to switch to a virtual environment first via `conda`, using Python 3.8.
- Install dependencies in your virtual environment via `pip install -r requirements.txt`

## Run
- To train or inference the model, run `python run.py --dataset_path <DATASET_PATH> --mode <MODE>`. The `<MODE>` parameter should be between `train`, `dev`, `test`.
- To run the demo, run `streamlit run demo.py`.
