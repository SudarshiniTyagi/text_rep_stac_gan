import numpy as np

UNKNOWN_CHAR = "<UNK>"
START_CHAR = "<START>"
END_CHAR = "<END>"
PAD_CHAR = "<PAD>"


class CharEmbeddings:
    def __init__(self):
        self._char_to_idx = {}
        self._idx_to_char = []
        self.embedding = np.eye(260)

    def look_up_char(self, char):
        if char == START_CHAR:
            return 256
        elif char == END_CHAR:
            return 257
        elif char == PAD_CHAR:
            return 258
        else:
            my_val = ord(char)
            if my_val > 255:
                return 259
            else:
                return my_val

    def look_up_token(self, idx):
        if idx < 256:
            return chr(idx)
        elif idx == 256:
            return START_CHAR
        elif idx == 257:
            return END_CHAR
        elif idx == 258:
            return PAD_CHAR
        elif idx == 259:
            return UNKNOWN_CHAR

    def clip_or_pad(self, sentence, char_sentence_length):
        chars = list(sentence)
        original_length = len(chars)
        if len(chars) < char_sentence_length:
            chars += [PAD_CHAR] * (char_sentence_length - len(chars))
        elif len(chars) > char_sentence_length:
            chars = chars[:char_sentence_length]
            original_length = char_sentence_length
        chars = [START_CHAR] + chars + [END_CHAR]
        return chars


    def convert_to_tokens(self, sentence, sentence_length):

        sentence = self.clip_or_pad(sentence, sentence_length)
        return np.array([self.look_up_char(char) for char in sentence], dtype=np.long),
