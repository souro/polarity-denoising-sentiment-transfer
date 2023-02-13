# Shrd Enc + Two Sep Decoders

## Overview
We extend the [Two Sep. transformers]() approach by keeping decoders separate, but using a shared encoder. During training, all examples are passed through the shared encoder, but each decoder is trained to only generate samples of one sentiment. Sentiment transfer is achieved by using the decoder for the opposite sentiment.

## Instructions
- Please create a directory and mention the same in the 'dir' option in [config]() file. This directory will contain the 'SRC' and 'TRG' [fields](https://torchtext.readthedocs.io/en/latest/data.html#fields), the vocabulary along with the checkpoint(s).
- Follow the data pre-processing [BPE process]() and mention the paths in 'train_pos_data_path', 'train_neg_data_path', 'valid_pos_data_path, 'valid_neg_data_path', 'test_pos_data_path', 'test_neg_data_path' options in [config]() file.
- Execute the [notebook]().
