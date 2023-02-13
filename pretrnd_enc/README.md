# Pre Training Enc

## Overview
We introduce a variant where the shared encoder is pretrained for back-translation on general-domain data. The pre-trained encoder is then further fine-tuned during sentiment transfer training.

## Instructions
- Please create a directory and mention the same in the 'dir' option in [config](https://github.com/SOURO/polarity-denoising-sentiment-transfer/blob/main/pretrnd_enc/config.json) file. This directory will contain the 'SRC' and 'TRG' [fields](https://torchtext.readthedocs.io/en/latest/data.html#fields), the vocabulary along with the checkpoint(s).
- Follow the data pre-processing [BPE process](https://github.com/SOURO/polarity-denoising-sentiment-transfer/blob/main/data/README.md) and mention the paths in 'train_data_path', 'valid_data_path', 'test_data_path', 'train_pos_data_path', 'train_neg_data_path', 'valid_pos_data_path, 'valid_neg_data_path', 'test_pos_data_path', 'test_neg_data_path' options in [config](https://github.com/SOURO/polarity-denoising-sentiment-transfer/blob/main/pretrnd_enc/config.json) file.
- Please mention the directory details in the option 'pretrn_encoder' in the [config](https://github.com/SOURO/polarity-denoising-sentiment-transfer/blob/main/pretrnd_enc/config.json) file. This directory is where you saved the checkpoint and other details from [vanila backtranslation](https://github.com/SOURO/polarity-denoising-sentiment-transfer/blob/main/back-translation/README.md) process.
- Other options in the [config](https://github.com/SOURO/polarity-denoising-sentiment-transfer/blob/main/pretrnd_enc/config.json) file is filled with the dafault values used in our experiments. Please adjust according to the requirements if required.
- Execute the [notebook](https://github.com/SOURO/polarity-denoising-sentiment-transfer/blob/main/pretrnd_enc/trn_st.ipynb).
