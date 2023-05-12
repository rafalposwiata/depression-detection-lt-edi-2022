# Detecting Signs of Depression from Social Media Text

This repository contains the code of our winning solution for [the Shared Task on Detecting Signs of Depression
from Social Media Text](https://competitions.codalab.org/competitions/36410) at [LT-EDI-ACL2022](https://sites.google.com/view/lt-edi-2022/home).

More information can be found in the following paper: [OPI@LT-EDI-ACL2022: Detecting Signs of Depression from Social Media Text using RoBERTa Pre-trained Language Models](https://aclanthology.org/2022.ltedi-1.40/).

## Task description

The task was to create a system that, given social media posts in English, should detect the level of depression as **‘not depressed’**, **‘moderately depressed’** or **‘severely depressed’**.

## Requirements

1. python 3.8+
2. transformers	4.13.0
3. simpletransformers 0.63.7
4. pandas 1.2.5
5. scikit-learn	0.23.1
6. tqdm	4.62.3

## Prepared datasets

We prepared two datasets. The first is a preprocessed dataset provided by the competition organizers.
The second, Reddit Depression Corpora, was used to train the DepRoBERTa language model.
 
### Preprocessed competition dataset

Dataset was prepared by removing duplicates and transfer some examples from the dev set to the train set.
Files are available in the [./data/preprocessed_dataset](data/preprocessed_dataset) folder.

### Reddit Depression Corpora

We built a corpus based on **[the Reddit Mental Health Dataset](https://zenodo.org/record/3941387#.Y5L6O_fMKUl)**
(Low et al., 2020) and a [dataset](https://www.kaggle.com/datasets/xavrig/reddit-dataset-rdepression-and-rsuicidewatch)
of **20,000** posts from **r/depression** and **r/SuicideWatch** subreddits. We filtered the data appropriately, leaving
mainly those related to **depression (31,2%)**, **anxiety (20,5%)** and **suicide (18.1%)**, which resulted in a corpora consisting
of **396,968** posts.

## Trained models

### DepRoBERTa

**DepRoBERTa (RoBERTa for Depression Detection)** - language model based on RoBERTa-large and further pre-trained on the
Reddit Depression Corpora.

[rafalposwiata/deproberta-large-v1](https://huggingface.co/rafalposwiata/deproberta-large-v1)

### Models for detecting depression

[rafalposwiata/roberta-large-depression](https://huggingface.co/rafalposwiata/roberta-large-depression)

[rafalposwiata/deproberta-large-depression](https://huggingface.co/rafalposwiata/deproberta-large-depression)

## Citation
If you use the code, models or datasets from this repository, please cite:

```bib
@inproceedings{poswiata-perelkiewicz-2022-opi,
    title = "{OPI}@{LT}-{EDI}-{ACL}2022: Detecting Signs of Depression from Social Media Text using {R}o{BERT}a Pre-trained Language Models",
    author = "Po{\'s}wiata, Rafa{\l} and Pere{\l}kiewicz, Micha{\l}",
    booktitle = "Proceedings of the Second Workshop on Language Technology for Equality, Diversity and Inclusion",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.ltedi-1.40",
    doi = "10.18653/v1/2022.ltedi-1.40",
    pages = "276--282",
}
```
If you use original competition dataset or preprocessed version, please also cite the below papers:

```bib
@inproceedings{10.1007/978-3-031-16364-7_11,
    title={Data Set Creation and Empirical Analysis for Detecting Signs of Depression from Social Media Postings},
    author= {Kayalvizhi, Sampath
    and Thenmozhi, Durairaj},
    editor={Kalinathan, Lekshmi
    and R., Priyadharsini
    and Kanmani, Madheswari
    and S., Manisha},
    booktitle={Computational Intelligence in Data Science},
    year={2022},
    publisher={Springer International Publishing},
    address={Cham},
    pages={136--151},
    isbn={978-3-031-16364-7}
}
```

```bib
@inproceedings{s-etal-2022-findings,
    title = {Findings of the Shared Task on Detecting Signs of Depression from Social Media},
    author = {S, Kayalvizhi  and
      Durairaj, Thenmozhi  and
      Chakravarthi, Bharathi Raja  and
      C, Jerin Mahibha},
    booktitle = {Proceedings of the Second Workshop on Language Technology for Equality, Diversity and Inclusion},
    month = {May},
    year = {2022},
    address = {Dublin, Ireland},
    publisher = {{Association for Computational Linguistics}},
    url = {https://aclanthology.org/2022.ltedi-1.51},
    doi = {10.18653/v1/2022.ltedi-1.51},
    pages = {331--338}
}
```