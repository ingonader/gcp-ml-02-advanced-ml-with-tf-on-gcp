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
import tensorflow as tf

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

## ========================================================================= ## 
## tensorflow snippets
## ========================================================================= ## 


## For small problems like this, it's easy to make a tf.data.Dataset by slicing the pandas.DataFrame:
def easy_input_function(df, label_key, num_epochs, shuffle, batch_size):
  label = df[label_key]
  ds = tf.data.Dataset.from_tensor_slices((dict(df),label))

  if shuffle:
    ds = ds.shuffle(10000)

  ds = ds.batch(batch_size).repeat(num_epochs)

  return ds

df = dat_train
df.columns = CSV_COLUMNS
label_key = LABEL_COLUMN

## trying out some small stuff from the function definition above:
dict(df)
dict(df)['weight_pounds']
type(dict(df)['weight_pounds'])  ## pandas.core.series.Series
