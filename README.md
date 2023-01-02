# Hierarchical Transformer for Long Text Summarization
A Hierarchical Transformer Encoder for Long Text Extractive Summarization by jointly learning salient and redundant features

## 1. Abstract
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
If ``` dataset_path ``` is not specified, the default dataset used is [arxiv-dataset](https://huggingface.co/datasets/ccdv/arxiv-summarization). When specifying a custom dataset, the format of the [arxiv-dataset](https://huggingface.co/datasets/ccdv/arxiv-summarization) must be used. 

The following link contains the final processed dataset for arxiv_dataset ready to be used for training: (link) 

For more details regarding how the dataset to be used for training is created, refer to (this section).

### 2.2 Training 
Start pretraining or finetuning the model with: 

```
$ python train.py 
```

The following flags can be specified:


```
$ python train.py --help
usage: train.py [-h] [--checkpoint_path CHECKPOINT_PATH] 
                [--extractive_summ_train_path EXTRACTIVE_SUMM_TRAIN_PATH] 
                [--gold_summ_train_path GOLD_SUMM_TRAIN_PATH]
                [--extractive_summ_val_path EXTRACTIVE_SUMM_VAL_PATH] 
                [--gold_summ_val_path GOLD_SUMM_VAL_PATH] 
                [--train_splits TRAIN_SPLITS]
                [--val_splits VAL_SPLITS] 
                [--train_data_ratio TRAIN_DATA_RATIO]
                [--val_data_ratio VAL_DATA_RATIO]
                [--batch_size BATCH_SIZE] 
                [--alpha ALPHA]
                [--gamma GAMMA]
                [--base_lr BASE_LR] 
                [--warmup_steps WARMUP_STEPS] 
                [--beta1 BETA1] 
                [--beta2 BETA2]
                [--no_of_epochs NO_OF_EPOCHS]
                [--model_freq MODEL_FREQ]
                [--val_freq VAL_FREQ] 
                [--train_loss_output_freq TRAIN_LOSS_OUTPUT_FREQ]
```

```checkpoint_path``` points to a file ```checkpoint.pth``` which is a dictionary with the following structure: 
```
checkpoint = { 
  'model': model.state_dict(),
  'optimizer': optimizer.state_dict(),
  'step_num': step_num
  'base_lr': base_lr
}
```
I specified the following hyperparameters:

```
epochs = 1
batch_size = 4
base_lr = 1e-3
warmup_steps = 2500
alpha = 0.8
gamma = 0.98
beta1 = 0.9
beta2 = 0.999
```

For more details regarding how the model is trained, refer to (this section)

### 2.3 Testing 

Evaluate a model's performance on the test set with:

```
$ python test.py
```

The following flags can be specified:

```
$ python test.py --help

usage: test.py [-h] [--extractive_summ_test_path EXTRACTIVE_SUMM_TEST_PATH] 
               [--gold_summ_test_path GOLD_SUMM_TEST_PATH] 
               [--test_splits TEST_SPLITS]
               [--batch_size BATCH_SIZE] 
               [--alpha ALPHA] [--gamma GAMMA]
               [--test_data_ratio TEST_DATA_RATIO] 
               [--rouge_output_freq ROUGE_OUTPUT_FREQ]
               model_path

```

For more details regarding how the model is evaluted, refer to (this section)

## 3. Method

## 4. Citations
