Overview
The Fake News Detector project uses machine learning algorithms to classify news articles as either real or fake. By leveraging natural language processing (NLP) techniques, the model analyzes news titles along with additional features, such as source domain and tweet count, to identify whether a news article is real or fake. The goal is to provide a reliable way to detect fake news based on text and associated metadata.

Dataset
The dataset used to train the model is the Fake and Real News Dataset, which includes:

title: The headline of the news article
news_url: URL link to the news article
source_domain: The domain name of the news source (e.g., cnn.com)
tweet_num: The number of tweets referencing the article
real: The target variable (1 = real news, 0 = fake news)
You can download the dataset from Fake and Real News Dataset.

Approach
The project follows these steps to preprocess the data and build the model:

Preprocessing
Convert all text to lowercase
Remove punctuation and stopwords
Apply lemmatization and stemming to reduce words to their base forms

Feature Engineering
Use TF-IDF vectorization to convert the text data into numerical features
Include numerical features like tweet_num (number of tweets)

Model
The model is built using Logistic Regression, with hyperparameters optimized to achieve the best performance. The model is evaluated based on metrics such as accuracy, precision, recall, and F1-score.

Performance Metrics
Accuracy: 88%
Precision (Fake News): 0.78
Precision (Real News): 0.91
Recall (Fake News): 0.70
Recall (Real News): 0.94
F1-Score (Fake News): 0.74
F1-Score (Real News): 0.92

Requirements
To run this project, you need the following Python libraries:
Python 3.6+
Pandas
Scikit-learn
NLTK
Streamlit (for deployment)
joblib (for saving and loading the trained model)

Installation

Clone the repository:
git clone https://github.com/Jedi3301e/fake-news-detector.git
cd fake-news-detector

Install the required dependencies:
pip install -r requirements.txt


Run the Streamlit app to interact with the model:
streamlit run app/app.py

License
This project is licensed under the MIT License.
