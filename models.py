import torch
import torch.nn as nn
from torchvision import models


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class RNN_LSTM(nn.Module):
    def __init__(self, batch_size, embedding, cell_size, num_layers, bidirectional, batch_first = True,
                 dropout_probability = 0.5):
        super(RNN_LSTM, self).__init__()

        self.batch_size = batch_size
        self.char_embeddings = nn.Embedding(embedding.shape[0], embedding.shape[1])
        self.char_embeddings.load_state_dict({'weight': torch.from_numpy(embedding)})
        self.char_embeddings.weight.requires_grad = False

        self.lstm = nn.LSTM(embedding.shape[1], cell_size,
                            num_layers=num_layers, bidirectional=bidirectional, batch_first=batch_first,
                            dropout=dropout_probability)

    def update_batch_size(self, batch_size):
        self.batch_size = batch_size

    def forward(self, hidden_state, cell_state, captions):
        embeds = self.char_embeddings(captions)
        outputs, (hidden_state, cell_state) = self.lstm(embeds, (hidden_state, cell_state))
        return outputs, (hidden_state, cell_state)




class Vanilla_Text_Encoder(nn.Module):
    def __init__(self, batch_size, embedding, cell_size, num_layers, bidirectional, GPU, gpu_nummber, batch_first =
    True,
                 dropout_probability = 0.5):

        super(Vanilla_Text_Encoder, self).__init__()

        self.c_dim = 256

        self.cnn = models.resnet18()
        self.cnn.fc = Identity()

        self.rnn = RNN_LSTM(batch_size, embedding, cell_size, num_layers, bidirectional, batch_first,
                 dropout_probability)

        self.fc = nn.Sequential(
            nn.Linear(cell_size, embedding.shape[0], bias=True),
            nn.BatchNorm1d(embedding.shape[0]),
            nn.ReLU()
        )

        self.GPU = GPU
        self.gpu_number = gpu_nummber
        self.batch_size = batch_size
        self.cell_size = cell_size

    def cond_aug_network(self, img_encoding):

        mu = img_encoding[:, :self.c_dim]
        logvar = img_encoding[:, self.c_dim:]
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        if self.GPU:
            eps = eps.cuda(self.gpu_number)
        hidden = eps.mul(std).add_(mu)
        return hidden

    def forward(self, images, captions):
        cnn_output = self.cnn(images)
        hidden_state = self.cond_aug_network(cnn_output)
        cell_state = torch.FloatTensor(hidden_state.shape[0], hidden_state.shape[1])
        cell_state.data.normal_(0, 1)
        if self.GPU:
            cell_state = cell_state.cuda(self.gpu_number)

        hidden_state = hidden_state.unsqueeze(0)
        cell_state = cell_state.unsqueeze(0)
        outputs, _ = self.rnn(hidden_state, cell_state, captions)

        predictions = None

        outputs = outputs.permute(1,0,2)

        for output in outputs:
            if predictions is None:
                predictions = self.fc(output).unsqueeze(1)
            else:
                predictions = torch.cat((predictions, self.fc(output).unsqueeze(1)), dim = 1)

        return predictions


