# Two Sep. transformers

## Overview
To get more control over sentiment-specific generation, we train two separate transformer models for positive and negative sentiment, using only sentences of the respective target sentiment. During inference, the model is fed with inputs of the opposite sentiment, which it did not see during training.

## Instructions
- Please create a directory and mention the same in the 'dir' option in [config]() file. This directory will contain the 'SRC' and 'TRG' [fields](https://torchtext.readthedocs.io/en/latest/data.html#fields), the vocabulary along with the checkpoint(s).
- Follow the data pre-processing [BPE process]() and mention the paths in 'train_pos_data_path', 'train_neg_data_path', 'valid_pos_data_path, 'valid_neg_data_path', 'test_pos_data_path', 'test_neg_data_path' options in [config]() file.
- Execute the [notebook]().