import operator
import pandas as pd
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def clean_text(text):
    return text.replace("\n", " ").replace(",,,,,,,,,,,", "").strip()


def create():
    unique_texts = set()
    stats = {}

    get_texts_from_reddit_mental_health_dataset(unique_texts, stats)
    get_texts_from_reddit_depression_suicidewatch(unique_texts, stats)

    print_stats(stats)
    save(unique_texts)


def get_texts_from_reddit_mental_health_dataset(unique_texts, stats):
    dir_with_dataset = '../data/reddit_depression_corpora/reddit_mental_health_dataset'

    if os.path.exists(dir_with_dataset):
        main_categories = ['suicidewatch', 'socialanxiety', 'healthanxiety', 'anxiety', 'lonely', 'bpd', 'depression',
                           'ptsd', 'bipolarreddit']

        for file_name in tqdm(os.listdir(dir_with_dataset)):
            category = file_name.split("_")[0]

            if category not in stats:
                stats[category] = 0

            df = pd.read_csv(os.path.join(dir_with_dataset, file_name), header=0)
            for idx, row in df.iterrows():
                text = clean_text(row["post"])
                if category not in main_categories and stats[category] == 10000:
                    break
                if text not in unique_texts:
                    stats[category] += 1
                    unique_texts.add(text)
    else:
        print('The directory "reddit_mental_health_dataset" does not exist. Create this directory and put to it the '
              'following files (downloaded from https://zenodo.org/record/3941387#.YFfi3EhJHL8): ')
        for i, file_name in enumerate(['anxiety_2018_features_tfidf_256.csv', 'anxiety_2019_features_tfidf_256.csv',
                                       'anxiety_post_features_tfidf_256.csv', 'anxiety_pre_features_tfidf_256.csv',
                                       'bipolarreddit_2018_features_tfidf_256.csv', 'bipolarreddit_2019_features_tfidf_256.csv',
                                       'bipolarreddit_post_features_tfidf_256.csv', 'bipolarreddit_pre_features_tfidf_256.csv',
                                       'bpd_2018_features_tfidf_256.csv', 'bpd_2019_features_tfidf_256.csv',
                                       'bpd_post_features_tfidf_256.csv',
                                       'bpd_pre_features_tfidf_256.csv', 'depression_2018_features_tfidf_256.csv',
                                       'depression_2019_features_tfidf_256.csv', 'depression_post_features_tfidf_256.csv',
                                       'depression_pre_features_tfidf_256.csv', 'fitness_2018_features_tfidf_256.csv',
                                       'fitness_2019_features_tfidf_256.csv', 'fitness_post_features_tfidf_256.csv',
                                       'fitness_pre_features_tfidf_256.csv', 'healthanxiety_2018_features_tfidf_256.csv',
                                       'healthanxiety_2019_features_tfidf_256.csv', 'healthanxiety_post_features_tfidf_256.csv',
                                       'healthanxiety_pre_features_tfidf_256.csv', 'jokes_2018_features_tfidf_256.csv',
                                       'jokes_2019_features_tfidf_256.csv', 'jokes_post_features_tfidf_256.csv',
                                       'jokes_pre_features_tfidf_256.csv',
                                       'legaladvice_2018_features_tfidf_256.csv', 'legaladvice_2019_features_tfidf_256.csv',
                                       'legaladvice_post_features_tfidf_256.csv', 'legaladvice_pre_features_tfidf_256.csv',
                                       'lonely_2018_features_tfidf_256.csv', 'lonely_2019_features_tfidf_256.csv',
                                       'lonely_post_features_tfidf_256.csv', 'lonely_pre_features_tfidf_256.csv',
                                       'parenting_2018_features_tfidf_256.csv', 'parenting_2019_features_tfidf_256.csv',
                                       'parenting_post_features_tfidf_256.csv', 'parenting_pre_features_tfidf_256.csv',
                                       'personalfinance_2018_features_tfidf_256.csv', 'personalfinance_2019_features_tfidf_256.csv',
                                       'personalfinance_post_features_tfidf_256.csv', 'personalfinance_pre_features_tfidf_256.csv',
                                       'ptsd_2018_features_tfidf_256.csv', 'ptsd_2019_features_tfidf_256.csv',
                                       'ptsd_post_features_tfidf_256.csv',
                                       'ptsd_pre_features_tfidf_256.csv', 'relationships_2018_features_tfidf_256.csv',
                                       'relationships_2019_features_tfidf_256.csv', 'relationships_post_features_tfidf_256.csv',
                                       'relationships_pre_features_tfidf_256.csv', 'socialanxiety_2018_features_tfidf_256.csv',
                                       'socialanxiety_2019_features_tfidf_256.csv', 'socialanxiety_post_features_tfidf_256.csv',
                                       'socialanxiety_pre_features_tfidf_256.csv', 'suicidewatch_2018_features_tfidf_256.csv',
                                       'suicidewatch_2019_features_tfidf_256.csv', 'suicidewatch_post_features_tfidf_256.csv',
                                       'suicidewatch_pre_features_tfidf_256.csv', 'teaching_2018_features_tfidf_256.csv',
                                       'teaching_2019_features_tfidf_256.csv', 'teaching_post_features_tfidf_256.csv',
                                       'teaching_pre_features_tfidf_256.csv']):
            print(f'{i + 1}. {file_name}')


def get_texts_from_reddit_depression_suicidewatch(unique_texts, stats):
    reddit_depression_suicidewatch = '../data/reddit_depression_corpora/reddit_depression_suicidewatch.csv'

    if os.path.exists(reddit_depression_suicidewatch):
        df = pd.read_csv(reddit_depression_suicidewatch, header=0)
        for idx, row in df.iterrows():
            text = clean_text(row["text"])
            label = row["label"]
            if text not in unique_texts:
                stats[label.lower()] += 1
                unique_texts.add(text)
    else:
        print('The file "reddit_depression_suicidewatch.csv" does not exist. You can download it from '
              'https://www.kaggle.com/datasets/xavrig/reddit-dataset-rdepression-and-rsuicidewatch.')


def print_stats(stats):
    all = sum(stats.values())
    for category, count in sorted(stats.items(), key=operator.itemgetter(1), reverse=True):
        print(f'{category}: {count} ({round(100 * (count / all), 1)}%)')


def save(unique_texts):
    df = pd.DataFrame(unique_texts)
    df = shuffle(df)
    train_df, val_df = train_test_split(df, test_size=0.02)

    print(f'\nUnique texts: {len(unique_texts)} (train: {train_df.shape[0]} / validation: {val_df.shape[0]}).')
    train_df.to_csv('../data/reddit_depression_corpora/reddit_depression_corpora_train.txt', index=False, header=False)
    val_df.to_csv('../data/reddit_depression_corpora/reddit_depression_corpora_val.txt', index=False, header=False)


if __name__ == '__main__':
    create()
