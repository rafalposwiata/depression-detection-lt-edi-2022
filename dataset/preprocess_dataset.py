import pandas as pd
from collections import Counter
from dataset.utils import get_data, labels


dev_probability = {
    'severe': 0.09,
    'moderate': 0.51,
    'not depression': 0.4
}


def preprocess(with_test=False):
    train = get_data('train')
    statistics('train', train)

    dev = get_data('dev')
    statistics('dev', dev)

    buckets = {l: [] for l in labels}
    all_texts = set()
    for data in [train, dev]:
        for idx, row in data.iterrows():
            pid = row['pid']
            text = row['text']
            label = row['labels']
            if text not in all_texts:
                buckets[labels[label]].append([pid, text, label])
                all_texts.add(text)

    train_dataset, dev_dataset = [], []
    for label, prob in dev_probability.items():
        v = int(prob * 1000)
        train_dataset += buckets[label][:-v]
        dev_dataset += buckets[label][-v:]

    train = pd.DataFrame(train_dataset, columns=['pid', 'text', 'labels'])
    print_stats('Train after preprocessing', train)
    train.to_csv('../data/preprocessed_dataset/train.csv', index=False)

    dev = pd.DataFrame(dev_dataset, columns=['pid', 'text', 'labels'])
    print_stats('Dev after preprocessing', dev)
    dev.to_csv('../data/preprocessed_dataset/dev.csv', index=False)

    if with_test:
        test = get_data('test')
        statistics('test', test)
        test.to_csv('../data/preprocessed_dataset/test.csv', index=False)


def statistics(data_split, dataset):
    unique_data = []
    all_texts = set()
    for idx, row in dataset.iterrows():
        if row['text'].lower() not in all_texts:
            unique_data.append([row['pid'], row['text'], row['labels']])
            all_texts.add(row['text'].lower())

    print_stats(f'Original {data_split}', dataset)

    df = pd.DataFrame(unique_data, columns=['pid', 'text', 'labels'])
    print_stats(f'Original {data_split} - without duplicates', df)


def print_stats(description, dataset):
    print(description)
    counts = Counter(list(dataset['labels'].values))
    for idx, label in enumerate(labels):
        print(f'{label}: {counts[idx]}')
    print(f'all: {len(dataset)}')
    print('-------------------------------------')


if __name__ == '__main__':
    preprocess()
