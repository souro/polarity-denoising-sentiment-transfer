# Dataset

## Overview
- We derived a new English dataset for sentiment transfer, based on the [Amazon Review Dataset]().
- For data processing we have used [TORCHTEXT](https://pytorch.org/text/stable/index.html) in our code.
- Apply the byte-pair encoding on the data before use. Please do the following


        Git clone the [repo](https://github.com/rsennrich/subword-nmt.git)
        cd subword-nmt/subword_nmt/
        ./apply-bpe -c {codes_file} < file_on_which_bpe_process_will_take_place > > output_file
  For 'codes_file', We have used it from the TORCHTEXT [WMT14](https://pytorch.org/text/0.8.1/datasets.html#wmt14) En-De dataset.
- Once done, please mention the paths in every model variants' config file.