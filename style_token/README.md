# Style Token

## Overview

Is a back-translation model with added sentiment identifiers (\<pos\> or \<neg\>) as output starting tokens. At the time of sentiment transfer, we decode the output with a changed sentiment identifier (\<pos\> → \<neg\>, \<neg\> → \<pos\>).
    
## Instructions

- Follow the data pre-processing [BPE process]() and mention the paths in 'train_pos_data_path', 'train_neg_data_path', 'valid_pos_data_path, 'valid_neg_data_path', 'test_pos_data_path', 'test_neg_data_path' options in [config]() file.
- Execute the [notebook]().