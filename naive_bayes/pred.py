from preprocess import preprocessGPQI
from string import punctuation
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

GPQI_fn = './data/masterGQPIcodes.csv'
corpus, labels, codes = preprocessGPQI(GPQI_fn)

filtered_corpus = []
filtered_labels = []
labels_set = set()
nums = set(["1","2","3","4","5","6","7","8","9"])
keywords = set()#"oz")
stpwrds = stopwords.words('english')

def clean(text):
    text = text.lower()
    no_punct = "".join([char for char in text if char not in punctuation and char not in nums])
    words = text.split()
    lemmatizer = WordNetLemmatizer()
    good_words = [word for word in words if word not in stpwrds]
    good_words = [word for word in words if len(word) > 1]
    good_words = [word for word in good_words if word not in keywords]
    cleaned = " ".join([lemmatizer.lemmatize(word) for word in good_words if word != None and len(word) > 0])
    return cleaned

for i in range(len(labels)):
    label = labels[i]
    text = corpus[i]

    if label != 999 and label != 99:
        clean_text = clean(text)
        filtered_corpus.append(text)
        filtered_labels.append(label)
        labels_set.add(label)
labelwords = {}
wordlabels = {}
for label in labels_set:
    labelwords[label] = {}

for i in range(len(filtered_corpus)):
    curr_label = filtered_labels[i]
    curr_descrip = filtered_corpus[i]

    if curr_descrip not in labelwords[curr_label]:
        labelwords[curr_label][curr_descrip] = 0
    labelwords[curr_label][curr_descrip] += 1

    if curr_descrip not in wordlabels:
        wordlabels[curr_descrip] = set()
    wordlabels[curr_descrip].add(curr_label)

