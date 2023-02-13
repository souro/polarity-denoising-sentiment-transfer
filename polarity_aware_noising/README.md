# Polarity-Aware Denoising

## Overview
The idea of our pre-training scheme—polarity-aware denoising is to first introduce noise, i.e. delete or mask a certain proportion of words in the intermediate German input to the back-translation step, then train the model to remove this noise ([see this]()), i.e. produce the original English sentence with no words deleted or masked.
We use polarity-aware denoising during encoder pretraining, following the shared encoder and separate decoders [setup](). The encoder is further fine-tuned during the sentiment transfer training.

## Instructions
- Please create a directory and mention the same in the 'dir' option in [config]() file. This directory will contain the 'SRC' and 'TRG' [fields](https://torchtext.readthedocs.io/en/latest/data.html#fields), the vocabulary along with the checkpoint(s).
- Follow the data pre-processing [BPE process]() and mention the paths in 'train_data_path', 'valid_data_path', 'test_data_path', 'train_pos_data_path', 'train_neg_data_path', 'valid_pos_data_path, 'valid_neg_data_path', 'test_pos_data_path', 'test_neg_data_path' options in [config]() file.
- Execute the [notebook]().
