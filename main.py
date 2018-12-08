import argparse
import json
import logging
import random
import sys
from datetime import datetime
from data_loader import DataLoader
from char_embeddings import CharEmbeddings
import torch
from trainer import Trainer

with open("config.json", 'r') as file:
    config = json.load(file)

parser = argparse.ArgumentParser(description="Toxicity and Sarcasm Detection")

# HyperParameters
parser.add_argument('--num_epochs', default=config['num_epochs'], type=int,
                    help='number of training epochs')

parser.add_argument('--batch_size', type=int, default=config['batch_size'],
                    help='Batch Size')

parser.add_argument('--lr', type=float, default=config['lr'],
                    help='Learning rate')

parser.add_argument('--seed', default=config['seed'], type=int,
                    help='Random Seed to Set')

# GPU Options
parser.add_argument('--gpu', action='store_true', default=config['gpu'],
                    help='Use GPU')

parser.add_argument('--gpu_number', default=config['gpu_number'], type=int,
                    help='Which GPU to run on')

# Maybe Remove later ?
parser.add_argument('--train_ratio', type=float, default=config['train_ratio'],
                    help='Ratio for Training Examples')

# Input and Output Paths

parser.add_argument('--embedding_path', default=config['embedding_path'], type=str,
                    help='path to embeddings')

parser.add_argument('--train_data', default=config['train_data'], type=str,
                    help='path to train data')

parser.add_argument('--save_model', default=config['save_model'], type=str,
                    help='path to directory to save model weights')

parser.add_argument('--log_dir', default=config['log_dir'], type=str,
                    help='path to directory to save logs')

parser.add_argument('--log_name', default=config['log_name'], type=str,
                    help='name of the log file starting string')

# Print and save Frequency Info

parser.add_argument('--print_after', default=config['print_after'], type=int,
                    help='Print Loss after every n iterations')

parser.add_argument('--validate_after', default=config['validate_after'], type=int,
                    help='Validate after every n iterations')

parser.add_argument('--save_after', default=config['save_after'], type=int,
                    help='Save after every n iterations')

parser.add_argument('--print', action='store_true', default=config['print'],
                    help='Print Log Output to stdout')

# Word Level Configs

parser.add_argument('--word_cell_size', type=int, default=config['word_cell_size'],
                    help='Cell cize for word RNN model')

parser.add_argument('--word_num_layers', type=int, default=config['word_num_layers'],
                    help='Number of RNN layers in Word level model')

parser.add_argument('--word_sentence_length', type=int, default=config['word_sentence_length'],
                    help='Max length of sentence for word level model')

parser.add_argument('--word_cell_type', type=str, default=config['word_cell_type'],
                    help='Type of cell (LSTM/GRU) for word level RNN')

parser.add_argument('--vocab_size', type=int, default=config['vocab_size'],
                    help='Vocab size for word level model')

# Char Level Configs

parser.add_argument('--char_cell_size', type=int, default=config['char_cell_size'],
                    help='Cell cize for char RNN model')

parser.add_argument('--char_num_layers', type=int, default=config['char_num_layers'],
                    help='Number of RNN layers in Char level model')

parser.add_argument('--char_sentence_length', type=int, default=config['char_sentence_length'],
                    help='Max length of sentence for char level model')

parser.add_argument('--char_cell_type', type=str, default=config['char_cell_type'],
                    help='Type of cell (LSTM/GRU) for char level RNN')

args = parser.parse_args()

random.seed(args.seed)

logging.basicConfig(level=logging.INFO,
                    filename=args.log_dir + args.log_name + datetime.now().strftime('%d_%m_%Y_%H_%M_%S.log'),
                    filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s'
                    )

if args.print:
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logging.getLogger().addHandler(ch)

logging.info(args)



def train():
    char_embedding = CharEmbeddings()
    train_dataset = DataLoader(args.train_data, 0, args.train_ratio, char_embedding, args.char_sentence_length)
    val_dataset = DataLoader(args.train_data, args.train_ratio, 1-args.train_ratio, char_embedding, args.char_sentence_length)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True,
                                               num_workers = 0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=0)

    logging.info("Number of training examples : {}".format(len(train_dataset)))
    logging.info("Number of val examples : {}".format(len(val_dataset)))

    gpu = torch.cuda.device_count() >= 1 and args.gpu

    trainer = Trainer(args.lr, args.batch_size, char_embedding.embedding, args.char_cell_size, args.char_num_layers,
                      args.char_sentence_length, False, gpu, args.gpu_number, True, 0.5)


    for epoch in range(1, args.num_epochs):
        epoch_loss = 0
        for idx, (images, captions, original_length) in enumerate(train_loader):
            batch_loss = trainer.fit(images, captions, original_length)
            logging.info("Epoch {} Batch {} Loss : {}".format(epoch, idx, batch_loss))
            epoch_loss += batch_loss

        epoch_loss /= len(train_loader)
        logging.info("For Train, overall loss for epoch {} = {}".format(epoch, epoch_loss))

        #TODO write Eval


if __name__ == "__main__":
    train()