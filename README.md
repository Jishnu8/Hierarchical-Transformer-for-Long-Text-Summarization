# Hierarchical Transformer for Long Text Summarization
A Hierarchical Transformer Encoder for Long Text Extractive Summarization by jointly learning salient and redundant features

## 1. Abstract
Despite the advent of transformer architectures, document summarization has been limited to short texts (1024 to 2048 tokens at best) because of the quadratic computational and memory complexities of computing self attention. To this extent, the following repository proposes HiExtSumm, a novel hierarchichal transformer encoder for long text extractive summarization by jointly learning salient and redudant features. First, a sentence level encoder (e.g Bert) is used to learn embeddings for each sentence. Then the sentence embeddings are fed through a document level encoder (2 self attention layers in this case) to learn the dependencies between sentences. Attentive pooling is then used to extract a global document embedding which is then concatenated with the sentence embeddings to learn salient features. The entire model is trained in an end-to-end fashion with a combination of two loss terms, the cross entropy loss L<sub>ce</sub> and a redundancy loss term L<sub>rd</sub>. During evalution, an MMR-based selection process is used to generate variable-length extractive summaries.

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

For more details regarding how the dataset to be used for training is created, refer to [3.1 Creating Dataset](https://github.com/Jishnu8/Hierarchical-Transformer-for-Long-Text-Summarization#31-creating-dataset).


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

For more details regarding how the model is trained, refer to [3.2 Model Architecture](https://github.com/Jishnu8/Hierarchical-Transformer-for-Long-Text-Summarization#32-model-architecture) and [3.3 Training](https://github.com/Jishnu8/Hierarchical-Transformer-for-Long-Text-Summarization#33-training)

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

For more details regarding how the model is evaluted, refer to [3.3 Evaluation](https://github.com/Jishnu8/Hierarchical-Transformer-for-Long-Text-Summarization#34-evaluation)

## 3. Method

### 3.1 Creating Dataset 

#### A brief note on extractive vs abstractive summarization 

**Extractive summariztion** aims to identiy the most important phrases and lines from a document. These important lines are then combined to create a summary.

On the hand, **abstractive summarization** aims to generate summaries that capture the salient features of a document. This method contrasts from the extractive approach as the summary could contain phrases and terms not present in the actual document. 

#### How to create extractive summaries?

Most summarization datasets contain only articles and their respective abstractive summaries. However in order to train our model to generate extractive summaries, we require the extractive summary for each article. Similar to [Nallapati et al. (2017)](https://github.com/Jishnu8/Hierarchical-Transformer-for-Long-Text-Summarization#4-references), we generate the extractive labels for each sentence so that they maximise the Rouge score with respect to the gold summaries (i.e the abstractive summary). This is done using a greedy algorithm where we add one sentence of the document at a time so that the Rouge score of the set of selected sentences at that time is maximised with respect to the gold summary. We stop this process either when adding a sentence to the summary does not increase the Rouge score or by setting a limit to the maximum number of sentences in a summary. While a greedy approach does not guarentee an extractive summary that maximises the Rouge score with respect to the gold summary, it provides us with a good approximation as generating the optimal extractive summary is computationally expensive.  

### 3.2 Model Architecture and Traning 

A good extractive summarizer should pick sentences that highlight the salient features of a document while also taking into account redundancy. The importance of addressing redundancy in extractive summaries is particularly important for long documents as they tend to be substantially more redundant than short ones ([Xiao and Carenini 2020](https://github.com/Jishnu8/Hierarchical-Transformer-for-Long-Text-Summarization#4-references)).

#### Architecture

The diagram above depicts the entire model architecture. First, a sentence level encoder (Bert) is used to learn sentence **s<sub>i,1</sub>** embeddings for each sentence. Then all the **s<sub>i,1</sub>** combined with positional encodings p<sub>i</sub> are fed through a document level encoder (2 self attention layers in this case) to learn the dependencies between sentences. The document level encoder outputs document aware sentence representations **c<sub>i,1</sub>**. Attentive pooling is then used to extract a global document embedding **D**. Finally **D** and **c<sub>i,1</sub>** are concatenated and fed through a linear and sigmoid layer to get probabilities of the importance of sentence i with respect to the global context of the document. The entire model is trained in an end-to-end fashion with a combination of two loss terms, the cross entropy loss **L<sub>ce</sub>** and a redundancy loss term **L<sub>rd</sub>**. 

#### Training Loss

As mentioned above, the training loss consists of a combination of two loss terms, the cross entropy loss **L<sub>ce</sub>** and a redundancy loss term **L<sub>rd</sub>**. The expression of the redundancy loss as described in [Xiao and Carenini 2020](https://github.com/Jishnu8/Hierarchical-Transformer-for-Long-Text-Summarization#4-references), is:


```

```

### 3.3 Evaluation

## 4. References

1. Ramesh Nallapati, Feifei Zhai, and Bowen Zhou. 2017.
SummaRuNNer: [A recurrent neural network based
sequence model for extractive summarization of
documents](https://arxiv.org/pdf/1611.04230.pdf). *In Proceedings of the 31st AAAI Con-
ference on Artificial Intelligence*, pages 3075–3081,
San Francisco, California.

2. Wen Xiao and Giuseppe Carenini. 2019. [Extractive
summarization of long documents by combining
global and local context](https://arxiv.org/pdf/2012.00052.pdf)). In Proceedings of the
2019 *Conference on Empirical Methods in Natural Language Processing and the 9th International
Joint Conference on Natural Language Processing
(EMNLP-IJCNLP)*, pages 3011–3021, Hong Kong,
China. Association for Computational Linguistics.
