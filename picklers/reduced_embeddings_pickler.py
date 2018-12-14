import pickle
from word_embeddings import WordEmbeddings
from nltk import word_tokenize


def findKMostFrequentWords(data_x, k):
    data = []
    for ele in data_x:
        sentence = ele[1]
        sentence = word_tokenize(sentence)
        data.append(sentence)
    ctr = Counter(tuple([word for sublist in data for word in sublist]))
    sorted_ctr = sorted(ctr.items(), key=operator.itemgetter(1), reverse=True)
    return [item[0] for item in sorted_ctr[0:k]]


full_embeddings = WordEmbeddings()
full_embeddings.create_embeddings_from_file(args.embedding_path)

words = findKMostFrequentWords(train_dataset.final_data, args.vocab_size)
reduced_embeddings = WordEmbeddings()
reduced_embeddings.create_reduced_embeddings(full_embeddings, words)

