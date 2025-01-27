# Back-Translation

## Overview

A pure back-translation approach (without any specific provisions for sentiment) is referred to as Back-Translation in our experiments. We use translation into German and subsequent encoding in a back-translation model to get a latent text representation (in our other following model variants) for our sentiment transfer task. We work with English as base language and German as intermediate language. We used the WMT14 English-German (en-de) dataset (1M sentences).

## Variants

* Vanila Back-Translation

    This uses simple WMT data. Encoder trained in this process used in the [Pre Training Enc](https://github.com/SOURO/polarity-denoising-sentiment-transfer/blob/main/pretrnd_enc/README.md) experiment as a pre-trained encoder.
    
* Noisy Back-Translation

    This process uses noisy WMT data prepared by [this](https://github.com/SOURO/polarity-denoising-sentiment-transfer/blob/main/prepare_noisy_data/README.md) process. Encoder trained in this process used in the [Polarity-Aware Denoising](https://github.com/SOURO/polarity-denoising-sentiment-transfer/blob/main/polarity_aware_noising/README.md) experiment as a pre-trained encoder. This experiment uses the 'SRC' and 'TRG' [fields](https://torchtext.readthedocs.io/en/latest/data.html#fields) saved from 'Vanila Back-Translation' experiment.
    
## Instructions

- Please create a directory and mention the same in the 'dir' option in [config](https://github.com/SOURO/polarity-denoising-sentiment-transfer/blob/main/back-translation/config.json) file. This directory will contain the 'SRC' and 'TRG' [fields](https://torchtext.readthedocs.io/en/latest/data.html#fields) the vocabulary along with the checkpoint.
- Please download the WMT14 English-German (en-de) dataset.
- Follow the data pre-processing [BPE process](https://github.com/SOURO/polarity-denoising-sentiment-transfer/blob/main/data/README.md) and mention the paths in 'train_data_path', 'valid_data_path', 'test_data_path'options in [config](https://github.com/SOURO/polarity-denoising-sentiment-transfer/blob/main/back-translation/config.json) file.
- Other options in the [config](https://github.com/SOURO/polarity-denoising-sentiment-transfer/blob/main/back-translation/config.json) file is filled with the dafault values used in our experiments. Please adjust according to the requirements if required.
- Fill the options in [config](https://github.com/SOURO/polarity-denoising-sentiment-transfer/blob/main/back-translation/config.json) for 'Vanila Back-Translation' and 'Noisy Back-Translation' experiments accordingly and execute the notebooks respectively.  
    
