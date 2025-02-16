import glob
import json
import os
import random

from torch.utils import data
from torch.utils.data import Subset

from datasets.episode import Episode
from datasets.wsd_dataset import WordWSDDataset, MetaWSDDataset


def write_json(json_dict, file_name):
    with open(file_name, 'w', encoding='utf8') as f:
        json.dump(json_dict, f, indent=4)


def read_json(file_name):
    with open(file_name, 'r', encoding='utf8') as f:
        json_dict = json.load(f)
    return json_dict


def get_max_batch_len(batch):
    return max([len(x[0]) for x in batch])


def prepare_batch(batch):
    max_len = get_max_batch_len(batch)
    x = []
    lengths = []
    y = []
    for inp_seq, target_seq in batch:
        lengths.append(len(inp_seq))
        target_seq = target_seq + [-1] * (max_len - len(target_seq))
        x.append(inp_seq)
        y.append(target_seq)
    return x, lengths, y


def prepare_task_batch(batch):
    return batch


def generate_semcor_wsd_episodes(wsd_dataset, n_episodes, n_support_examples, n_query_examples, task):
    word_splits = {k: v for (k, v) in wsd_dataset.word_splits.items() if len(v['sentences']) >
                   (n_support_examples + n_query_examples)}

    if n_episodes > len(word_splits):
        raise Exception('Not enough data available to generate {} episodes'.format(n_episodes))

    episodes = []
    for word in word_splits.keys():
        if len(episodes) == n_episodes:
            break
        indices = list(range(len(word_splits[word]['sentences'])))
        random.shuffle(indices)
        start_index = 0
        train_subset = WordWSDDataset(sentences=[word_splits[word]['sentences'][i] for i in indices[start_index: start_index + n_support_examples]],
                                      labels=[word_splits[word]['labels'][i] for i in indices[start_index: start_index + n_support_examples]],
                                      n_classes=len(wsd_dataset.sense_inventory[word]))
        support_loader = data.DataLoader(train_subset, batch_size=n_support_examples, collate_fn=prepare_batch)
        start_index += n_support_examples
        test_subset = WordWSDDataset(sentences=[word_splits[word]['sentences'][i] for i in indices[start_index: start_index + n_query_examples]],
                                     labels=[word_splits[word]['labels'][i] for i in indices[start_index: start_index + n_query_examples]],
                                     n_classes=len(wsd_dataset.sense_inventory[word]))
        query_loader = data.DataLoader(test_subset, batch_size=n_query_examples, collate_fn=prepare_batch)
        episode = Episode(support_loader=support_loader,
                          query_loader=query_loader,
                          base_task=task,
                          task_id=task + '-' + word,
                          n_classes=train_subset.n_classes)
        episodes.append(episode)
    return episodes


def generate_wsd_episodes(dir, n_episodes, n_support_examples, n_query_examples, task, meta_train=True):
    episodes = []
    for file_name in sorted(glob.glob(os.path.join(dir, '*.json'))):
        if len(episodes) == n_episodes:
            break
        word = file_name.split(os.sep)[-1].split('.')[0]
        word_wsd_dataset = MetaWSDDataset(file_name)
        train_subset = Subset(word_wsd_dataset, range(0, n_support_examples))
        support_loader = data.DataLoader(train_subset, batch_size=n_support_examples, collate_fn=prepare_batch)
        if meta_train:
            test_subset = Subset(word_wsd_dataset, range(n_support_examples, n_support_examples + n_query_examples))
        else:
            test_subset = Subset(word_wsd_dataset, range(n_support_examples, len(word_wsd_dataset)))
        query_loader = data.DataLoader(test_subset, batch_size=n_query_examples, collate_fn=prepare_batch)
        episode = Episode(support_loader=support_loader,
                          query_loader=query_loader,
                          base_task=task,
                          task_id=task + '-' + word,
                          n_classes=word_wsd_dataset.n_classes)
        episodes.append(episode)
    return episodes
