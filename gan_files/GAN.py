from gan_files.Generator1 import *
from gan_files.Discriminator1 import *
from models import Vanilla_Text_Encoder


class GAN(nn.Module):
    def __init__(self, rnn_params, gen1_params, disc1_params, batch_size, embedding, GPU, gpu_number, dropout):
        super(GAN, self).__init__()
        self.batch_size = 32 #TODO


        self.vanilla_encoder = Vanilla_Text_Encoder(batch_size, embedding, rnn_params['cell_size'], rnn_params['num_layers']
            ,bidirectional=False, GPU = GPU, gpu_nummber = gpu_number, batch_first=True,
                                                    dropout_probability=dropout)
        self.vanilla_encoder.load_state_dict(torch.load("/Users/sudarshinityagi/PycharmProjects/text_rep_stac_gan/models"
                                          "/char_text_rep__199"))

        self.RNN = self.vanilla_encoder.rnn
        for param in self.RNN.lstm.parameters():
            param.requires_grad = False


        # self.RNN = CharRNN_Torch(
        #     cell_size = rnn_params.get("cell_size", 256),
        #     n_layers = rnn_params.get("n_layers", 1),
        #     embedding = rnn_params.get("embedding"),
        #     cell_type = rnn_params.get("cell_type", "LSTM"),
        #     batch_size = rnn_params.get("batch_size", 32),
        #     GPU = rnn_params.get("GPU", True),
        #     gpu_number = rnn_params.get("gpu_number", 0),
        #     dropout_probability = rnn_params.get("dropout_probability", 0.5)
        # )

        self.Gen1 = Generator1()
        self.Disc1 = Discriminator1(96, 128)

        self.Gen1.apply(self.init_weights)
        self.Disc1.apply(self.init_weights)

    def init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, image, true_caption, false_caption):
        hidden_state = torch.FloatTensor(true_caption.shape[0], 256)
        cell_state = torch.FloatTensor(true_caption.shape[0], 256)
        hidden_state.data.normal_(0,1)
        cell_state.data.normal_(0, 1)
        hidden_state = hidden_state.unsqueeze(0)
        cell_state = cell_state.unsqueeze(0)
        true_output, (true_hidden_state, true_cell_state) = self.RNN(hidden_state, cell_state, true_caption)
        false_output, (false_hidden_state, false_cell_state) = self.RNN(hidden_state, cell_state, false_caption)
        true_encoded = torch.cat((true_hidden_state, true_cell_state), 0)
        false_encoded = torch.cat((false_hidden_state, false_cell_state), 0)
        true_encoded = true_encoded.view(true_caption.shape[0], -1)
        false_encoded = false_encoded.view(false_caption.shape[0], -1)
        _, generated1_images, mu, logvar = self.Gen1(true_encoded)

        _, _, false_mu, _ = self.Gen1(false_encoded)

        cond = mu.detach()
        lv = logvar.detach()
        gen_img = generated1_images.detach()

        false_cond = false_mu.detach()

        real_disc_preds = self.Disc1(image, cond)
        gen_disc_preds1 = self.Disc1(gen_img, cond)
        gen_disc_preds2 = self.Disc1(gen_img, cond)
        #incorrect_disc_preds = self.Disc1(image[:self.batch_size-1], cond[1:])
        incorrect_disc_preds = self.Disc1(image, false_cond)

        return real_disc_preds, gen_disc_preds1, gen_disc_preds2, incorrect_disc_preds, cond, lv



class GANTrainer():
    def __init__(self, rnn_params, gen1_params, disc1_params, batch_size, embedding, GPU, gpu_number, dropout = 0.5):

        self.GAN = GAN(rnn_params, gen1_params, disc1_params, batch_size, embedding, GPU, gpu_number, dropout)
        self.loss_function = nn.BCELoss(reduce=True)
        self.rnn_Optim = optim.Adam(self.GAN.RNN.parameters(), lr=rnn_params.get("lr", 0.0002), betas=(0.5, 0.999))
        self.gen1_Optim = optim.Adam(self.GAN.Gen1.parameters(), lr=gen1_params.get("lr", 0.0002), betas=(0.5, 0.999))
        self.disc1_Optim = optim.Adam(self.GAN.Disc1.parameters(), lr=disc1_params.get("lr", 0.0002), betas=(0.5,
                                                                                                             0.999))
        self.loss_fn = nn.BCELoss(reduce=True)
        self.GPU = True

    def KL_loss(self, mu, logvar):
        # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.mean(KLD_element).mul_(-0.5)
        return KLD

    def fit(self, image, true_caption, false_caption, train_disc, train_gen):
        # image_tensor = torch.from_numpy(image).type(torch.FloatTensor)
        # true_caption_tensor = torch.from_numpy(true_caption)
        # false_caption_tensor = torch.from_numpy(false_caption)

        real_disc_preds, gen_disc_preds1, gen_disc_preds2, incorrect_disc_preds, mu, logvar = self.GAN(image,
                                                                                          true_caption,
                                                                         false_caption)

        self.gen1_Optim.zero_grad()
        self.rnn_Optim.zero_grad()


        gen1_loss = self.loss_fn(gen_disc_preds1, torch.ones_like(gen_disc_preds1))
        kl_loss = self.KL_loss(mu, logvar)
        errG_total = gen1_loss + kl_loss * 2.0
        errG_total.backward()

        if train_gen:
            self.gen1_Optim.step()

        self.disc1_Optim.zero_grad()

        disc1_loss = 0;
        disc1_loss += self.loss_fn(real_disc_preds, torch.ones_like(real_disc_preds))
        disc1_loss += (self.loss_fn(gen_disc_preds2, torch.zeros_like(gen_disc_preds2)) +
                      self.loss_fn(incorrect_disc_preds, torch.zeros_like(incorrect_disc_preds))) * 0.5

        disc1_loss.backward()
        self.gen1_Optim.zero_grad()
        self.rnn_Optim.zero_grad()

        if train_disc:
            self.disc1_Optim.step()

        return gen1_loss.data[0], disc1_loss.data[0]


    def train(self, images, true_captions, false_captions, batch_num, epoch_num, gpu=True):
        self.GAN.train()
        if gpu:
            self.GAN = self.GAN

        gen1_loss, disc1_loss = self.fit(images, true_captions, false_captions,
                                         train_gen=True, train_disc=True)

        print("Epoch num: ", epoch_num, "Batch_num: ", batch_num, "Gen1 Loss : ", gen1_loss.item())
        print("Epoch num: ", epoch_num, "Batch_num: ", batch_num, "Disc1 Loss : ", disc1_loss.item())

    def save(self, filepath):
        print("Saving model to file {}".format(filepath))
        if self.GPU:
            torch.save(self.GAN.cpu().state_dict(), filepath)
            self.GAN = self.GAN
        else:
            torch.save(self.GAN.state_dict(), filepath)
        print("Model saved successfully !")
