# Pre Training Enc

## Overview
We introduce a variant where the shared encoder is pretrained for back-translation on general-domain data. The pre-trained encoder is then further fine-tuned during sentiment transfer training.

## Instructions
- Please create a directory and mention the same in the 'dir' option in [config]() file. This directory will contain the 'SRC' and 'TRG' [fields](https://torchtext.readthedocs.io/en/latest/data.html#fields), the vocabulary along with the checkpoint(s).
- Follow the data pre-processing [BPE process]() and mention the paths in 'train_data_path', 'valid_data_path', 'test_data_path', 'train_pos_data_path', 'train_neg_data_path', 'valid_pos_data_path, 'valid_neg_data_path', 'test_pos_data_path', 'test_neg_data_path' options in [config]() file.
- Execute the [notebook]().
