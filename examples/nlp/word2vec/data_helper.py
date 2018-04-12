import random
import itertools
import collections
import numpy as np
import huaytools as hy
import tensorflow as tf


def get_raw_data(filename):
    """
    The raw data is text8, you can download it from http://mattmahoney.net/dc/text8

    Args:
        filename:

    """
    with open(filename) as f:
        data = f.read().split()
    return data


def build_dataset(raw_data, vocabulary_size=50000):
    """
    build the dataset, include word2id and id2word

    Args:
        raw_data:
        vocabulary_size:

    Returns:

    """
    word_cnt = [['UNK', -1]]
    word_cnt.extend(
        collections.Counter(raw_data).most_common(vocabulary_size - 1))

    word2id = dict()
    for word, _ in word_cnt:
        word2id[word] = len(word2id)

    id2word = dict(zip(word2id.values(), word2id.keys()))

    # word to id
    data = []
    unk_cnt = 0
    for word in raw_data:
        if word in word2id:
            _id = word2id[word]
        else:
            _id = word2id['UNK']  # 0
            unk_cnt += 1
        data.append(_id)
    word_cnt[0][1] = unk_cnt

    return data, word_cnt, word2id, id2word


def gen_batch(data, batch_size=64, skip_window=1):
    """
    generate the batch data,
    return a cycle iterator

    Examples:
        "the dog barked at the mailman"

            skip_window=1
            batch: [['the','barked'],['dog','at'],['barked','the'],['at','mailman']]
            labels: ['dog','barked','at','the']

            skip_window=2
            batch: [['the', 'dog','at', 'the'],['dog','barked', 'the', 'mailman']
            labels: ['barked','at']

        The words has been turned to id

    Args:
        data:
        batch_size:
        skip_window:

    Returns:
        batch, labels
    """
    span_size = 2 * skip_window + 1  # [ skip_window target skip_window ]
    word = itertools.cycle(data)  # cycle yield

    span = collections.deque(maxlen=span_size)

    # run once to fill the queue
    for _ in range(span_size):
        span.append(next(word))

    while True:
        batch_features = []
        batch_labels = []
        for _ in range(batch_size):
            mid = span_size // 2
            batch_features.append([span[i] for i in range(span_size) if i != mid])
            # batch_labels.append(span[mid])
            batch_labels.append([span[mid]])

            span.append(next(word))  # get next word, and auto pop the first word

        # assert len(batch_features) == batch_size and len(batch_features[0]) == span_size - 1
        yield batch_features, batch_labels


def get_all_data(filename, batch_size=64, skip_window=1, valid_size=10, valid_window=100):
    raw_data = get_raw_data(filename)

    tf.logging.info("Test8 语料总词数：%d", len(raw_data))

    data, word_cnt, word2id, id2word = build_dataset(raw_data)

    assert valid_size % 2 == 0, "valid_size 需要偶数"
    # while True:
    #     valid_data = np.array(random.sample(range(valid_window), valid_size))
    #     yield valid_data
    valid_data = np.array(random.sample(range(1, valid_window + 1), valid_size), dtype=np.int32)

    batch_data = gen_batch(data, batch_size, skip_window)
    return batch_data, valid_data, word2id, id2word
    # return lambda: gen_batch(data, batch_size, skip_window), valid_data, word2id, id2word
