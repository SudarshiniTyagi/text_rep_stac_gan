from torch.utils import data
import pickle

class DataLoader(data.Dataset):
    """Data Loader Class"""

    def __init__(self, data_filepath, offset_ratio, ratio, embedding, sentence_length):
        file = open(data_filepath, 'rb')
        self.final_data = pickle.load(file)
        self.offset = offset_ratio * len(self.final_data)
        self.length = int(ratio * len(self.final_data))
        self.embedding = embedding
        self.sentence_length = sentence_length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        (img, caption) = self.final_data[self.offset + idx]
        tokens, original_length = self.embedding.convert_to_tokens(caption, self.sentence_length)
        return (img, tokens, original_length)


