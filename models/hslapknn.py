# File containing new class HslapknnDataWriter()
#
# Writes embedded episode data in a format consumable by A JupyterNotebook script to perform HSlaPKnn
# classification - based on algorithms 3, 4 and 5 from the paper "One Line To Rule Them All:
# Generating LO-Shot Soft-Label Prototypes"
#
# This file maintains the to the implementation style of the other models within the models directory and
# where code performs the same functions as in the other models - it is copied and re-used here directly
# - all such credit goes to Nithin Holla the other authors of "Learning to Learn to Disambiguate: Meta-Learning
# for Few-Shot Word Sense Disambiguation", unmodified codebase and information here:
# https://github.com/Nithin-Holla/MetaWSD


import coloredlogs
import logging
import torch
import torchtext
from allennlp.modules import Elmo
from allennlp.modules.elmo import batch_to_ids
import numpy as np
from transformers import BertTokenizer, BertModel
import csv
from pathlib import Path

logger = logging.getLogger('HSLaPkNN Log')
coloredlogs.install(logger=logger, level='DEBUG', fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class HslapknnDataWriter():
    def __init__(self, config):
        self.vectors = config.get('vectors', 'elmo')
        self.device = torch.device(config.get('device', 'cpu'))
        self.write_directory = config.get('write_directory')

        if self.vectors == 'elmo':
            self.elmo = Elmo(options_file="https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json",
                             weight_file="https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5",
                             num_output_representations=1,
                             dropout=0,
                             requires_grad=False)
            self.elmo.to(self.device)
        elif self.vectors == 'glove':
            self.glove = torchtext.vocab.GloVe(name='840B', dim=300)
        elif self.vectors == 'bert':
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            self.bert = BertModel.from_pretrained('bert-base-cased')
            self.bert.to(self.device)

        logger.info('Hslapknn data writer instantiated')

    def vectorize(self, batch_x, batch_len, batch_y):
        with torch.no_grad():
            if self.vectors == 'elmo':
                char_ids = batch_to_ids(batch_x)
                char_ids = char_ids.to(self.device)
                batch_x = self.elmo(char_ids)['elmo_representations'][0]
            elif self.vectors == 'glove':
                max_batch_len = max(batch_len)
                vec_batch_x = torch.ones((len(batch_x), max_batch_len, 300))
                for i, sent in enumerate(batch_x):
                    sent_emb = self.glove.get_vecs_by_tokens(sent, lower_case_backup=True)
                    vec_batch_x[i, :len(sent_emb)] = sent_emb
                batch_x = vec_batch_x.to(self.device)
            elif self.vectors == 'bert':
                max_batch_len = max(batch_len) + 2
                input_ids = torch.zeros((len(batch_x), max_batch_len)).long()
                for i, sent in enumerate(batch_x):
                    sent_token_ids = self.bert_tokenizer.encode(sent, add_special_tokens=True)
                    input_ids[i, :len(sent_token_ids)] = torch.tensor(sent_token_ids)
                batch_x = input_ids.to(self.device)
                attention_mask = (batch_x.detach() != 0).float()
                batch_x, _ = self.bert(batch_x, attention_mask=attention_mask)
                batch_x = batch_x[:, 1:-1, :]
        batch_len = torch.tensor(batch_len).to(self.device)
        batch_y = torch.tensor(batch_y).to(self.device)
        return batch_x, batch_len, batch_y

    def training(self, train_episodes, val_episodes):
        # In the case of hslapknn, where training does not occur, this is solely responsible for performing tuning
        # on the validation set
        for episode_id, episode in enumerate(val_episodes):
            logger.info('Writing validation data for episode: ' + str(episode_id))
            batch_x, batch_len, batch_y = next(iter(episode.support_loader))
            support_repr, _, support_labels = self.vectorize(batch_x, batch_len, batch_y)
            support_repr = support_repr.reshape(support_repr.shape[0] * support_repr.shape[1], -1)
            support_labels = support_labels.view(-1)
            support_repr = support_repr[support_labels != -1].cpu().numpy()
            support_labels = support_labels[support_labels != -1].cpu().numpy()

            batch_x, batch_len, batch_y = next(iter(episode.query_loader))
            query_repr, _, true_labels = self.vectorize(batch_x, batch_len, batch_y)
            query_repr = query_repr.reshape(query_repr.shape[0] * query_repr.shape[1], -1)
            true_labels = true_labels.view(-1)
            query_repr = query_repr[true_labels != -1].cpu().numpy()
            true_labels = true_labels[true_labels != -1].cpu().numpy()

            target_directory = "validation_data/" + self.write_directory + "/support_sets/"
            Path(target_directory).mkdir(parents=True, exist_ok=True)
            support_set_file_name = "episode_" + str(episode_id) + "_support_set.csv"
            with open(target_directory + support_set_file_name, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                for i in range(len(support_labels)):
                    writer.writerow(np.append(support_repr[i], support_labels[i]))

            target_directory = "validation_data/" + self.write_directory + "/query_sets/"
            Path(target_directory).mkdir(parents=True, exist_ok=True)
            query_set_file_name = "episode_" + str(episode_id) + "_query_set.csv"
            with open(target_directory + query_set_file_name, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                for i in range(len(true_labels)):
                    writer.writerow(np.append(query_repr[i], true_labels[i]))

        logger.info('All hslapknn validation data created')
        return 0

    def testing(self, test_episodes):
        for episode_id, episode in enumerate(test_episodes):
            logger.info('Writing test data for episode: ' + str(episode_id))
            batch_x, batch_len, batch_y = next(iter(episode.support_loader))
            support_repr, _, support_labels = self.vectorize(batch_x, batch_len, batch_y)
            support_repr = support_repr.reshape(support_repr.shape[0] * support_repr.shape[1], -1)
            support_labels = support_labels.view(-1)
            support_repr = support_repr[support_labels != -1].cpu().numpy()
            support_labels = support_labels[support_labels != -1].cpu().numpy()

            batch_x, batch_len, batch_y = next(iter(episode.query_loader))
            query_repr, _, true_labels = self.vectorize(batch_x, batch_len, batch_y)
            query_repr = query_repr.reshape(query_repr.shape[0] * query_repr.shape[1], -1)
            true_labels = true_labels.view(-1)
            query_repr = query_repr[true_labels != -1].cpu().numpy()
            true_labels = true_labels[true_labels != -1].cpu().numpy()

            target_directory = self.write_directory + "/support_sets/"
            Path(target_directory).mkdir(parents=True, exist_ok=True)
            support_set_file_name = "episode_" + str(episode_id) + "_support_set.csv"
            with open(target_directory + support_set_file_name, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                for i in range(len(support_labels)):
                    writer.writerow(np.append(support_repr[i], support_labels[i]))

            target_directory = self.write_directory + "/query_sets/"
            Path(target_directory).mkdir(parents=True, exist_ok=True)
            query_set_file_name = "episode_" + str(episode_id) + "_query_set.csv"
            with open(target_directory + query_set_file_name, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                for i in range(len(true_labels)):
                    writer.writerow(np.append(query_repr[i], true_labels[i]))

        logger.info('All hslapknn test data created')
        return 0
