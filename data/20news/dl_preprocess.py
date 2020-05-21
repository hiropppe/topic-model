import gensim
import nltk

from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.datasets import fetch_20newsgroups


nltk.download('stopwords')
nltk.download('wordnet')

stop_words = nltk.corpus.stopwords.words('english')


def main():
    for subset in ('train', 'test'):
        bunch = download(subset)
        texts = preprocess(bunch.data)
        save(texts, f"{subset}.txt")


def download(subset):
    return fetch_20newsgroups(subset=subset, download_if_missing=True, shuffle=True)


def preprocess(docs):
    texts = []
    for doc in docs:
        tokens = tokenize(doc, stop_words=stop_words, lemmatize_fun=lemmatize_stemming)
        texts.append(tokens)
    return texts


def save(texts, out):
    with open(out, "w") as fout:
        for text in texts:
            print(' '.join(text), file=fout)


def lemmatize_stemming(text):
    stemmer = PorterStemmer()
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def tokenize(text, stop_words=[], lemmatize_fun=None):
    tokens = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS:
            if token not in stop_words:
                tokens.append(lemmatize_fun(token))
    return tokens


if __name__ == "__main__":
    main()
