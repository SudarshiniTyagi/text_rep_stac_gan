import numpy as np
from collections import Counter
import operator

from nltk import word_tokenize

UNKNOWN_WORD = "<UNK>"
START_WORD = "<START>"
END_WORD = "<END>"
PAD_WORD = "<PAD>"
ABUSE_WORD = "<ABUSE>"

extra_tokens = 5



class WordEmbeddings:
    def __init__(self):
        self._word_to_idx = {}
        self._idx_to_word = []

        self.UNKNOWN_TOKEN = self._add_word(UNKNOWN_WORD)
        self.START_TOKEN = self._add_word(START_WORD)
        self.END_TOKEN = self._add_word(END_WORD)
        self.PAD_TOKEN = self._add_word(PAD_WORD)
        self.ABUSE_TOKEN = self._add_word(ABUSE_WORD)

    def look_up_word(self, word):
        return self._word_to_idx.get(word, self.UNKNOWN_TOKEN)

    def look_up_token(self, token):
        return self._idx_to_word[token]

    def _add_word(self, word):
        idx = len(self._idx_to_word)
        self._word_to_idx[word] = idx
        self._idx_to_word.append(word)
        return idx

    def create_embeddings_from_file(self, file_path):
        with open(file_path, encoding="utf-8") as f:
            line = f.readline()
            chunks = line.split(" ")
            dimensions = len(chunks) - 1
            f.seek(0)

            vocab_size = sum(1 for line in f)
            vocab_size += extra_tokens
            f.seek(0)

            self.glove = np.ndarray((vocab_size, dimensions), dtype=np.float32)
            self.glove[self.UNKNOWN_TOKEN] = np.zeros(dimensions)
            self.glove[self.START_TOKEN] = -np.ones(dimensions)
            self.glove[self.END_TOKEN] = np.ones(dimensions)
            self.glove[self.PAD_TOKEN] = -np.ones(dimensions) * 2
            self.glove[self.ABUSE_TOKEN] = -np.ones(dimensions) * 3

            lines_read = 0
            for line in f:
                # if lines_read > 2000:
                #     break
                chunks = line.split(" ")
                idx = self._add_word(chunks[0])
                self.glove[idx] = [float(chunk) for chunk in chunks[1:]]
                if len(self._idx_to_word) >= vocab_size:
                    break
                lines_read+=1
                print("Read {}/{} lines in glove".format(lines_read, vocab_size), end = '\r')

    def create_reduced_embeddings(self, embedding, words):
        self.glove = []

        dimensions = embedding.glove.shape[1]
        vocab_size = len(words) + extra_tokens

        self.glove.append(np.zeros(dimensions))
        self.glove.append(-np.ones(dimensions))
        self.glove.append(np.ones(dimensions))
        self.glove.append(np.ones(dimensions) * 2)
        self.glove.append(np.ones(dimensions) * 3)

        tokens = [self.UNKNOWN_TOKEN, self.START_TOKEN, self.END_TOKEN, self.PAD_TOKEN, self.ABUSE_TOKEN]
        for word in words:
            l = embedding.look_up_word(word)
            if l not in tokens:
                idx = self._add_word(word)
                self.glove.append(embedding.glove[l])
            if(len(self.glove) == vocab_size):
                break
        self.glove = np.array(self.glove)


    def clip_or_pad(self, sentence, word_sentence_length):
        words = word_tokenize(sentence)
        original_length = len(words)
        if len(words) < word_sentence_length:
            words += [PAD_WORD] * (word_sentence_length - len(words))
        elif len(words) > word_sentence_length:
            words = words[:word_sentence_length]
            original_length = word_sentence_length
        words = [START_WORD] + words + [END_WORD]
        return words, original_length

    def convert_to_tokens(self, sentence, sentence_length):
        sentence, original_length = self.clip_or_pad(sentence, sentence_length)
        return np.array([self.look_up_word(word) for word in sentence], dtype=np.long), original_length


