# Polarity aware noisy dataset preperation

## Overview
- To introduce noise, i.e. delete or mask a certain proportion of words in the intermediate German input to the back-translation step.
- To decide which words get deleted or masked, we use automatically obtained sentiment polarity labels.
- We apply three different approaches: deleting or masking (1) general words (i.e., all the words uniformly), (2) polarity words (i.e., only high-polarity words according to a lexicon, or (3) both general and polarity words (each with a different probability).

## Instructions
- To use different data for each epoch, please run [this](https://github.com/SOURO/polarity-denoising-sentiment-transfer/blob/main/prepare_noisy_data/prepare_noisy_data.ipynb) process beforehand for <no_of_epoch> times. We prepared 50 sets (no of epochs was 50) beforehand and put it in the subdirectories named 1,2,...,50 under a directory and used that directory path details in the config file where required.
    
