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


## ========================================================================= ## 
## Apache Beam Lab
## ========================================================================= ## 

#import apache_beam as beam
import datetime, os

"""
This function seems to be a generator, hence using the yield keyword instead of return. 
It seems to do the preprocessing of a single input row read from bigquery (probably).
Function will be called in Apache Beam's `.FlatMap` function.

Explanation of `yield`:
To master yield, you must understand that when you call the function, the code you have written 
in the function body does not run. The function only returns the generator object; your code 
will be run each time the `for` uses the generator.
"""
def to_csv(rowdict):
  import hashlib
  import copy

  # TODO #1:
  # Pull columns from BQ and create line(s) of CSV input
  CSV_COLUMNS = 'weight_pounds,is_male,mother_age,plurality,gestation_weeks'.split(',')
    
  # Create synthetic data where we assume that no ultrasound has been performed
  # and so we don't know sex of the baby. Let's assume that we can tell the difference
  # between single and multiple, but that the errors rates in determining exact number
  # is difficult in the absence of an ultrasound.
  no_ultrasound = copy.deepcopy(rowdict)
  w_ultrasound = copy.deepcopy(rowdict)

  no_ultrasound['is_male'] = 'Unknown'
  if rowdict['plurality'] > 1:
    no_ultrasound['plurality'] = 'Multiple(2+)'
  else:
    no_ultrasound['plurality'] = 'Single(1)'

  # Change the plurality column to strings
  w_ultrasound['plurality'] = ['Single(1)', 
                               'Twins(2)', 
                               'Triplets(3)', 
                               'Quadruplets(4)', 
                               'Quintuplets(5)'][rowdict['plurality'] - 1]

  # Write out two rows for each input row, one with ultrasound and one without
  for result in [no_ultrasound, w_ultrasound]:
    data = ','.join([str(result[k]) if k in result else 'None' for k in CSV_COLUMNS])
    key = hashlib.sha224(data).hexdigest()  # hash the columns to form a key
    yield str('{},{}'.format(data, key))
  
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## now for some trial stuff in terms of what this function does:
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

rowdict = None ## [[todo]]
ultrasound = None ## [[todo]]
data = ','.join([str(result[k]) if k in result else 'None' for k in CSV_COLUMNS])
    key = hashlib.sha224(data).hexdigest()  # hash the columns to form a key
  