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
## some handy code snippets
## ========================================================================= ## 

## execute system command:
import os
output = os.popen("whoami").readlines()
print(output)
output[0][:-1]

## parsing argument:
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
        '--bucket',
        help = 'GCS path to data. We assume that data is in gs://BUCKET/babyweight/preproc/',
        type = str,
        default = "Some string",
        required = False
)
parser.add_argument(
        '-f'
)
## parse all arguments
args = parser.parse_args()
arguments = args.__dict__

## -------------------------------------------------------------- ##
## making pandas data frames
## -------------------------------------------------------------- ##

## make pandas dataframe from numpy array:
n = 10
npa = np.zeros(n)
npa.shape  # (n,)
type(npa)


dat_tmp = pd.DataFrame(npa, npa)
dat_tmp.shape  # (n, 1)

dat_tmp = pd.DataFrame({'col1' : npa, 'col2' : npa})
dat_tmp.head
dat_tmp.shape # (n, 2)


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
## Lab 4: Apache Beam Lab
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

import copy
import hashlib

CSV_COLUMNS = 'weight_pounds,is_male,mother_age,plurality,gestation_weeks'.split(',')

rowdict = {u'hashmonth': 1525201076796226340, u'gestation_weeks': 40, u'is_male': False, u'weight_pounds': 8.75014717878, u'plurality': 1, u'mother_age': 34}
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
print(w_ultrasound)
# {'hashmonth': 1525201076796226340,
#  'gestation_weeks': 40,
#  'is_male': False,
#  'weight_pounds': 8.75014717878,
#  'plurality': 'Single(1)',
#  'mother_age': 34}

# Write out two rows for each input row, one with ultrasound and one without
# result = no_ultrasound
for result in [no_ultrasound, w_ultrasound]:
  data = ','.join([str(result[k]) if k in result else 'None' for k in CSV_COLUMNS])
  data = data.encode('utf-8')
  key = hashlib.sha224(data).hexdigest()  # hash the columns to form a key
  print(str('{},{}'.format(data, key)))
  
# data:  '8.75014717878,Unknown,34,Single(1),40'   ## originally, without explicit encoding
# data: b'8.75014717878,Unknown,34,Single(1),40'   ## with additinal encoding
# key:   'e96a75f2fe27ffdf5594cbc98f29ef808c32e9e1ad18741b28a6fbad'

## complete output:
# b'8.75014717878,Unknown,34,Single(1),40',e96a75f2fe27ffdf5594cbc98f29ef808c32e9e1ad18741b28a6fbad
# b'8.75014717878,False,34,Single(1),40',15a66154127a3150f29026a8ccc8e18adc68b47af1fd0c5d11374247


## ========================================================================= ## 
## Lab 5: Cloud ML Engine
## ========================================================================= ## 

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## file: task.py
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##


import argparse
import json
import os

from . import model

import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_runner

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bucket',
        help = 'GCS path to data. We assume that data is in gs://BUCKET/babyweight/preproc/',
        required = True
    )
    parser.add_argument(
        '--output_dir',
        help = 'GCS location to write checkpoints and export models',
        required = True
    )
    parser.add_argument(
        '--batch_size',
        help = 'Number of examples to compute gradient over.',
        type = int,
        default = 512
    )
    parser.add_argument(
        '--job-dir',
        help = 'this model ignores this field, but it is required by gcloud',
        default = 'junk'
    )
    parser.add_argument(
        '--nnsize',
        help = 'Hidden layer sizes to use for DNN feature columns -- provide space-separated layers',
        nargs = '+',
        type = int,
        default=[128, 32, 4]
    )
    parser.add_argument(
        '--nembeds',
        help = 'Embedding size of a cross of n key real-valued parameters',
        type = int,
        default = 3
    )
    parser.add_argument(
        '--train_examples',
        help = 'Number of examples used to train the model',
        type = int,
        default = 5000
    )

    ## TODO 1: add the new arguments here 
    parser.add_argument(
        '--eval_steps',
        help = 'Number of steps used for evaluation of the model',
        type = int,
        default = None
    )
    parser.add_argument(
        '--pattern',
        help = 'File pattern for data files',
        type = str,
        default = 'of'
    )    

    ## parse all arguments
    args = parser.parse_args()
    arguments = args.__dict__

    # unused args provided by service
    arguments.pop('job_dir', None)
    arguments.pop('job-dir', None)

    ## assign the arguments to the model variables
    output_dir = arguments.pop('output_dir')
    model.BUCKET     = arguments.pop('bucket')
    model.BATCH_SIZE = arguments.pop('batch_size')
    model.TRAIN_STEPS = (arguments.pop('train_examples') * 1000) / model.BATCH_SIZE
    model.EVAL_STEPS = arguments.pop('eval_steps')    
    print ("Will train for {} steps using batch_size={}".format(model.TRAIN_STEPS, model.BATCH_SIZE))
    model.PATTERN = arguments.pop('pattern')
    model.NEMBEDS= arguments.pop('nembeds')
    model.NNSIZE = arguments.pop('nnsize')
    print ("Will use DNN size of {}".format(model.NNSIZE))

    # Append trial_id to path if we are doing hptuning
    # This code can be removed if you are not using hyperparameter tuning
    output_dir = os.path.join(
        output_dir,
        json.loads(
            os.environ.get('TF_CONFIG', '{}')
        ).get('task', {}).get('trial', '')
    )

    # Run the training job
    model.train_and_evaluate(output_dir)


