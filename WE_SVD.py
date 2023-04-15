import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import randomized_svd
import pprint
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE




stop_words = set(stopwords.words('english'))
punctuations = list(string.punctuation)
stop_words.update(punctuations)


sentences = []
with open('reviews_Movies_and_TV.json', 'r') as f:
    count = 0
    
    for line in f:
        data = json.loads(line)
        
        sentence = data['reviewText']
        sentence = sentence.lower()
        sentence = re.sub(r'[^\w\s]', '', sentence)
        sentence = re.sub(r'[0-9]', '', sentence)

        stop_count = 0
        no_of_words = 0
        for word in sentence:
            no_of_words+=1
            if word in stop_words:
                stop_count+=1

        if no_of_words > 0:
            if stop_count/no_of_words > 0.5:
                continue
            else:
                count+=1
                sentences.append(sentence)
        if count == 70000:
            break

vocab = {}
for sentence in sentences:
    for word in sentence.split():
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1

for word in list(vocab.keys()):
    if vocab[word] < 8:
        del vocab[word]

vocab_size = len(vocab)
vocab["UNK"] = vocab_size 

vocab_file = open("vocab.txt", "w")
print(vocab, file = vocab_file)

word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for word, i in word_to_idx.items()}

word2idx = open("word2idx.txt", "w")
pprint.pprint(word_to_idx, word2idx)

print("Word to index of UNK is: ", word_to_idx["UNK"])



def get_co_occurrence_matrix(sentences, vocab_size, window_size=2):
    matrix = np.zeros((vocab_size+10, vocab_size+10), dtype=np.float64)
    for sentence in sentences:
        # indices = [word_to_idx[word] for word in sentence.split() if word in word_to_idx]
        indices = []
        for word in sentence.split():
            if word in word_to_idx:
                indices.append(word_to_idx[word])
            else:
                indices.append(word_to_idx["UNK"])
        for center_i, center in enumerate(indices):
            for w in range(-window_size, window_size+1):
                context_i = center_i + w
                if context_i < 0 or context_i >= len(indices) or center_i == context_i:
                    continue
                context = indices[context_i]
                matrix[center, context] += 1
    return matrix

def train_embeddings(matrix, embedding_size):
    print("Shape = ", matrix.shape)
    U, _, _ = randomized_svd(matrix , embedding_size)
    return U


matrix = np.zeros((vocab_size+10, vocab_size+10), dtype=np.float64)
matrix = get_co_occurrence_matrix(sentences, vocab_size, window_size=1)
matrix_sum = np.sum(matrix)

print("Sum of the matrix = ", matrix_sum)

np.save("matrix.npy", matrix)

embedding_size = 10
embeddings = train_embeddings(matrix , embedding_size)

out = open("WE_SVD.txt", "w")

for i in range(vocab_size):
    print(idx_to_word[i],"---> has frequency:", vocab[idx_to_word[i]], "--->", embeddings[i], file = out)

print(vocab_size)

# print 10 closest words to the word "titanic"
distances = []
for word in vocab.keys():
        if word != "titanic":
            distances.append((np.linalg.norm(embeddings[word_to_idx["titanic"]] - embeddings[word_to_idx[word]]), word))
distances.sort()

for i in range(10):
    print(distances[i])

# word_list = ["computer", "friends", "famous", "running", "pretty", "review", "badly"]

# # plot the embeddings of the given word list:
# tsne = TSNE(n_components=2, random_state=0)
# np.set_printoptions(suppress=True)
# T = tsne.fit_transform(embeddings[:100, :])
# labels = list(vocab.keys())[:100]
# plt.figure(figsize=(20, 20))
# for i, label in enumerate(labels):
#     x, y = T[i, :]
#     plt.scatter(x, y)
#     plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
# plt.legend()
# plt.show()

# generate glove embeddings for the word titanic
glove_embeddings = {}
with open("glove.6B.50d.txt", "r") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        glove_embeddings[word] = vector

# print 10 closest words to the word "titanic"
distances = []
for word in vocab.keys():
        try :
            glove_embeddings[word]
            if word != "titanic":
                distances.append((np.linalg.norm(glove_embeddings["titanic"] - glove_embeddings[word]), word))
        except:
            continue
distances.sort()

for i in range(10):
    print(distances[i])





