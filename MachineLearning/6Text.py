from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from nltk.tag import TrigramTagger
from nltk.tag import BigramTagger
from nltk.tag import UnigramTagger
from nltk.corpus import brown
from nltk import word_tokenize
from nltk import pos_tag
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import sys
import unicodedata
from bs4 import BeautifulSoup
import re
from sklearn.preprocessing import MultiLabelBinarizer

print('=======================|textcleaning|=======================')

# Create text
text_data = ["   Interrobang. By Aishwarya Henriette     ",
             "Parking And Going. By Karl Gautier",
             "    Today Is The night. By Jarek Prakash   "]

print(text_data)

# Strip whitespaces
strip_whitespace = [string.strip() for string in text_data]

# Show text
print(strip_whitespace)


['Interrobang. By Aishwarya Henriette',
 'Parking And Going. By Karl Gautier',
 'Today Is The night. By Jarek Prakash']

# Remove periods
remove_periods = [string.replace(".", "") for string in strip_whitespace]

# Show text
print(remove_periods)

# Create function


def capitalizer(string: str) -> str:
    return string.upper()


# Apply function
caps = [capitalizer(string) for string in remove_periods]

print(caps)

print('=======================|regex|=======================')

# Import library

# Create function


def replace_letters_with_X(string: str) -> str:
    return re.sub(r"[a-zA-Z]", "X", string)


# Apply function
regex = [replace_letters_with_X(string) for string in remove_periods]

print(regex)

print('=======================|parse clean html|=======================')

# Load library

# Create some HTML code
html = """
       <div class='full_name'><span style='font-weight:bold'>
       Masego</span> Azra</div>"
       """

# Parse html
soup = BeautifulSoup(html, "lxml")

# Find the div with the class "full_name", show text
print(soup.find("div", {"class": "full_name"}).text)

print('=======================|remove punctuation|=======================')
# Load libraries

# Create text
text_data = ['Hi!!!! I. Love. This. Song....',
             '10000% Agree!!!! #LoveIT',
             'Right?!?!']

# Create a dictionary of punctuation characters
punctuation = dict.fromkeys(i for i in range(
    sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))

# For each string, remove any punctuation characters
translated = [string.translate(punctuation) for string in text_data]
print(translated)

print('=======================|tokenizing words|=======================')

# Load library

# Create text
string = "The science of today is the technology of tomorrow"

# Tokenize words
print(word_tokenize(string))


print('=======================|tokenize sentences|=======================')

# Load library

# Create text
string = "The science of today is the technology of tomorrow. Tomorrow is today."

# Tokenize sentences
print(sent_tokenize(string))

print('=======================|removing stopwords (common no info words)|=======================')

# Load library

# You will have to download the set of stop words the first time
# import nltk
# nltk.download('stopwords')

# Create word tokens
tokenized_words = ['i',
                   'am',
                   'going',
                   'to',
                   'go',
                   'to',
                   'the',
                   'store',
                   'and',
                   'park']

# Load stop words
stop_words = stopwords.words('english')

# Remove stop words
stopwords_removed = [
    word for word in tokenized_words if word not in stop_words]
print(stopwords_removed)

# Show stop words
print(stop_words[:5])

print('=======================|wrd tkn root form|=======================')
# Load library

# Create word tokens
tokenized_words = ['i', 'am', 'humbled',
                   'by', 'this', 'traditional', 'meeting']

# Create stemmer
porter = PorterStemmer()

# Apply stemmer
stemmed = [porter.stem(word) for word in tokenized_words]

print(tokenized_words)
print(stemmed)

print('=======================|tagging parts of speech|=======================')
"""
Penn Treebank tags
Tag
    Part of speech
    NNP, Proper noun, singular
    NN, Noun, singular or mass
    RB, Adverb
    VBD, Verb, past tense    
    VBG, Verb, gerund or present participle
    JJ, Adjective
    PRP, Personal pronoun
"""

# Load libraries

# Create text
text_data = "Chris loved outdoor running"

# Use pre-trained part of speech tagger
text_tagged = pos_tag(word_tokenize(text_data))

# Show parts of speech
print(text_tagged)

# Filter words
allnouns = [word for word, tag in text_tagged if tag in [
    'NN', 'NNS', 'NNP', 'NNPS']]
print(allnouns)

print('====================================================')

# Create text
tweets = ["I am eating a burrito for breakfast",
          "Political science is an amazing field",
          "San Francisco is an awesome city"]

# Create list
tagged_tweets = []

# Tag each word and each tweet
for tweet in tweets:
    #tweet_tag = nltk.pos_tag(word_tokenize(tweet))
    tweet_tag = pos_tag(word_tokenize(tweet))
    tagged_tweets.append([tag for word, tag in tweet_tag])

# Use one-hot encoding to convert the tags into features
one_hot_multi = MultiLabelBinarizer()
one_hot_multi.fit_transform(tagged_tweets)

# Show feature names
print(one_hot_multi.classes_)

print('=======================|trian our own tagger, labor intensive, last resort|=======================')
"""
All that said, if we had a tagged corpus and wanted to train a tagger,
the following is an example of how we could do it. The corpus we are using is the Brown Corpus, one of the most popular sources of tagged text. Here we use a backoff n-gram tagger, where n is the number of previous words we take into account when predicting a word’s part-of-speech tag. First we take into account the previous two words using TrigramTagger; if two words are not present, we “back off” and take into account the tag of the previous one word using BigramTagger, and finally if that fails we only look at the word itself using UnigramTagger. To examine the accuracy of our tagger, we split our text data into two parts, train our tagger on one part, and test how well it predicts the tags of the second part:
"""

# Load library

# Get some text from the Brown Corpus, broken into sentences
sentences = brown.tagged_sents(categories='news')

# Split into 4000 sentences for training and 623 for testing
train = sentences[:4000]
test = sentences[4000:]

# Create backoff tagger
unigram = UnigramTagger(train)
bigram = BigramTagger(train, backoff=unigram)
trigram = TrigramTagger(train, backoff=bigram)

# Show accuracy
trigram.evaluate(test)

print('=======================|Encode text as word bag|=======================')

# Load library

# Create text
text_data = np.array(['I love Brazil. Brazil!',
                      'Sweden is best',
                      'Germany beats both'])

# Create the bag of words feature matrix
count = CountVectorizer()
bag_of_words = count.fit_transform(text_data)

# Show feature matrix
print(type(bag_of_words))
print(bag_of_words)

# bag of words to array
print(bag_of_words.toarray())

# Show feature names
print(count.get_feature_names_out())

print('====================================================')

# Create feature matrix with arguments
count_2gram = CountVectorizer(ngram_range=(1, 2),
                              stop_words="english",
                              vocabulary=['brazil'])
bag = count_2gram.fit_transform(text_data)

# View feature matrix
print(bag.toarray())

# View the 1-grams and 2-grams
print(count_2gram.vocabulary_)

print('=======================|word importance weighing|=======================')
#compare frequency of word in doc

# Load libraries
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Create text
text_data = np.array(['I love Brazil. Brazil!',
                      'Sweden is best',
                      'Germany beats both'])

# Create the tf-idf feature matrix
tfidf = TfidfVectorizer()
feature_matrix = tfidf.fit_transform(text_data)

# Show tf-idf feature matrix
print(type(feature_matrix))
print(feature_matrix)

# Show tf-idf feature matrix as dense matrix
print(feature_matrix.toarray())

# Show feature names
print(tfidf.vocabulary_)







