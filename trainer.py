import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from models import Vanilla_Text_Encoder
import numpy as np


class Trainer:
    def __init__(self, lr, batch_size, embedding, cell_size, num_layers, sentence_length, bidirectional, GPU, \
                                                                                        gpu_number, batch_first
    = True, dropout_probability = 0.5):

        self.model = Vanilla_Text_Encoder(batch_size, embedding, cell_size, num_layers, bidirectional, GPU, gpu_number, batch_first,
                 dropout_probability)

        self.GPU = GPU
        self.gpu_number = gpu_number
        self.batch_size = batch_size
        self.embedding = embedding
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.999))
        self.sentence_length = sentence_length

    def sentence_mask(self, original_length_sentence):
        if (original_length_sentence < self.sentence_length):
            mask = [1] * (original_length_sentence+1) + [0] * (self.sentence_length - original_length_sentence)
        else:
            mask = [1] * (self.sentence_length + 1)
        return mask



    def loss(self, predictions, captions, original_length):
        mask = []
        for idx in range(captions.shape[0]):
            mask.append(self.sentence_mask(original_length.data.cpu().numpy()[idx]))
        mask = torch.from_numpy(np.array(mask, dtype=np.float32))
        if self.GPU:
            mask = mask.cuda(self.gpu_number)
        loss = 0
        predictions = predictions.permute(1,0,2)
        captions = captions.permute(1,0)
        mask = mask.permute(1,0)

        captions = captions[1:]
        predictions = predictions[:-1]

        for idx in range(captions.shape[0]):
            crossentropy = nn.CrossEntropyLoss(reduce=False)(predictions[idx], captions[idx]) * mask[idx]
            loss += torch.mean(crossentropy)
        return loss

    def fit(self, images, captions, original_length):
        self.model.train()
        if self.GPU:
            images = images.cuda(self.gpu_number)
            captions = captions.cuda(self.gpu_number)
            self.model = self.model.cuda(self.gpu_number)

        self.optimizer.zero_grad()
        predictions = self.model(images, captions)
        loss_value = self.loss(predictions, captions, original_length)
        loss_value.backward()
        self.optimizer.step()
        return loss_value.item()

    def eval(self, images, captions):
        self.model.eval()
        with torch.no_grad():
            if self.GPU:
                images = images.cuda(self.gpu_number)
                captions = captions.cuda(self.gpu_number)
                self.model = self.model.cuda(self.gpu_number)

            predictions = self.model(images, captions)
            loss_value = self.loss(predictions, captions)
        return loss_value.item()