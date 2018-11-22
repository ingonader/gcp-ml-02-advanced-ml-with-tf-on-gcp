## ######################################################################### ##
## Code Snippets for GCP Coursera Course
## ######################################################################### ##

## ========================================================================= ## 
## import libraries
## ========================================================================= ##

import os
import numpy as np
import pandas as pd
import collections   ## for Counters (used in frequency tables, for example)


## ========================================================================= ## 
## global variables and options
## ========================================================================= ## 

path_raw =  "/Users/ingonader/data-um-sync/training/coursera-work/" + \
            "gcp-ml-02-advanced-ml-with-tf-on-gcp"
path_dat =  os.path.join(path_raw, "01-end-to-end-ml-with-tf-on-gcp/" + \
            "lab-solutions")

# print(path_raw)
# print(path_dat)

CSV_COLUMNS = 'weight_pounds,is_male,mother_age,plurality,gestation_weeks,key'.split(',')
LABEL_COLUMN = 'weight_pounds'
KEY_COLUMN = 'key'

colnames_dict = { CSV_COLUMNS[i] : i for i in range(0, len(CSV_COLUMNS)) }
print(colnames_dict)

## ========================================================================= ## 
## read data
## ========================================================================= ## 

dat_train = pd.read_csv(os.path.join(path_dat, "train.csv"))
dat_train.head()


## ========================================================================= ## 
## inspect data
## ========================================================================= ## 

## make frequency table:
tmp = collections.Counter(dat_train.iloc[:, colnames_dict['is_male']])
print(tmp)
print(tmp.values())
print(tmp.keys())
print(tmp.most_common(3))

tmp = collections.Counter(dat_train.iloc[:, colnames_dict['plurality']])
print(tmp.keys())



np.arange(15, 45, 1)
np.arange(15, 45, 1).tolist()
