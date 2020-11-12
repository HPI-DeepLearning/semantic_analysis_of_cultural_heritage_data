from sklearn.feature_extraction.text import CountVectorizer


def build_vocab(dataset):
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2), min_df=5)
    vectorizer.fit(dataset)

    return vectorizer
