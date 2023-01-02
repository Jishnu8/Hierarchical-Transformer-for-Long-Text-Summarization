# Hierarchical Transformer for Long Text Summarization
A Hierarchical Transformer Encoder for Long Text Extractive Summarization by jointly learning salient and redundant features

## 1. Overview
Despite the advent of transformer architectures, document summarization has been limited to short texts (1024 to 2048 tokens at best) because of the quadratic computational and memory complexities of computing self attention. To this extent, the following repository proposes HiExtSumm, a novel hierarchichal transformer encoder for long text extractive summarization by jointly learning salient and redudant features. First, a sentence level encoder (e.g Bert) is used to learn embeddings for each sentence. Then the sentence embeddings are fed through a document level encoder (2 self attention layers in this case) to learn the dependencies between sentences. Attentive pooling is then used to extract a global document embedding which is then concatenated with the sentence embeddings to learn salient features. The entire model is trained in an end-to-end fashion with a combination of two loss terms, the cross entropy loss Lce and a redundancy loss term Lrd. During evalution, an MMR-based selection process is used to generate variable-length extractive summaries.

For more details regarding each step of the process, refer to each section below.

## 2. Usage
### 2.1 Create dataset
Given the link of a dataset repo in Hugginface or the path to a local dataset which contains articles and their respective summaries, the following two datasets are created:

a) Dataset of articles and their respective extractive summaries

b) Dataset of indexes and their respective gold summaries

```
$ python create_ext_sum_dataset
```

The following flags can be specified:

```
$ python create_ext_sum_dataset --help
usage: create_ext_sum_dataset.py [-h] [--dataset_path DATASET_PATH]
                                 [--file_extension FILE_EXTENSION]
                                 [--max_no_of_sentences MAX_NO_OF_SENTENCES]
                                 [--extractive_summ_train_path EXTRACTIVE_SUMM_TRAIN_PATH]
                                 [--extractive_summ_val_path EXTRACTIVE_SUMM_VAL_PATH]
                                 [--extractive_summ_test_path EXTRACTIVE_SUMM_TEST_PATH]
                                 [--gold_summ_train_path GOLD_SUMM_TRAIN_PATH]
                                 [--gold_summ_val_path GOLD_SUMM_VAL_PATH]
                                 [--gold_summ_test_path GOLD_SUMM_TEST_PATH]
                                 [--train_splits TRAIN_SPLITS]
                                 [--val_splits VAL_SPLITS]
                                 [--test_splits TEST_SPLITS]                                 
```
By default, the dataset used is [arxiv-dataset](https://huggingface.co/datasets/ccdv/arxiv-summarization). When specifying a custom dataset, the format of the [arxiv-dataset](https://huggingface.co/datasets/ccdv/arxiv-summarization) must be used. 


