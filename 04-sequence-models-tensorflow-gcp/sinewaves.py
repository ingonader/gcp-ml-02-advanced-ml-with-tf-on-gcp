
# coding: utf-8

# <h1> Time series prediction, end-to-end </h1>
# 
# This notebook illustrates several models to find the next value of a time-series:
# <ol>
# <li> Linear
# <li> DNN
# <li> CNN 
# <li> RNN
# </ol>

# In[ ]:


# change these to try this notebook out
BUCKET = 'cloud-training-demos-ml'
PROJECT = 'cloud-training-demos'
REGION = 'us-central1'
SEQ_LEN = 50


# In[ ]:


import os
os.environ['BUCKET'] = BUCKET
os.environ['PROJECT'] = PROJECT
os.environ['REGION'] = REGION
os.environ['SEQ_LEN'] = str(SEQ_LEN)
os.environ['TFVERSION'] = '1.8'


# <h3> Simulate some time-series data </h3>
# 
# Essentially a set of sinusoids with random amplitudes and frequencies.

# In[ ]:


import tensorflow as tf
print(tf.__version__)


# In[ ]:


import numpy as np
import seaborn as sns

def create_time_series():
  freq = (np.random.random()*0.5) + 0.1  # 0.1 to 0.6
  ampl = np.random.random() + 0.5  # 0.5 to 1.5
  noise = [np.random.random()*0.3 for i in range(SEQ_LEN)] # -0.3 to +0.3 uniformly distributed
  x = np.sin(np.arange(0,SEQ_LEN) * freq) * ampl + noise
  return x

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
for i in range(0, 5):
  sns.tsplot( create_time_series(), color=flatui[i%len(flatui)] );  # 5 series


# In[ ]:


def to_csv(filename, N):
  with open(filename, 'w') as ofp:
    for lineno in range(0, N):
      seq = create_time_series()
      line = ",".join(map(str, seq))
      ofp.write(line + '\n')

import os
try:
  os.makedirs('data/sines/')
except OSError:
  pass

np.random.seed(1) # makes data generation reproducible

to_csv('data/sines/train-1.csv', 1000)  # 1000 sequences
to_csv('data/sines/valid-1.csv', 250)


# In[ ]:


get_ipython().system('head -5 data/sines/*-1.csv')


# <h3> Train model locally </h3>
# 
# Make sure the code works as intended.

# In[ ]:


get_ipython().run_line_magic('bash', '')
DATADIR=$(pwd)/data/sines
OUTDIR=$(pwd)/trained/sines
rm -rf $OUTDIR
gcloud ml-engine local train    --module-name=sinemodel.task    --package-path=${PWD}/sinemodel    --    --train_data_path="${DATADIR}/train-1.csv"    --eval_data_path="${DATADIR}/valid-1.csv"     --output_dir=${OUTDIR}    --model=linear --train_steps=10 --sequence_length=$SEQ_LEN


# <h3> Cloud ML Engine </h3>
# 
# Now to train on Cloud ML Engine with more data.

# In[ ]:


import shutil
shutil.rmtree('data/sines', ignore_errors=True)
os.makedirs('data/sines/')
np.random.seed(1) # makes data generation reproducible
for i in range(0,10):
  to_csv('data/sines/train-{}.csv'.format(i), 1000)  # 1000 sequences
  to_csv('data/sines/valid-{}.csv'.format(i), 250)


# In[ ]:


get_ipython().run_line_magic('bash', '')
gsutil -m rm -rf gs://${BUCKET}/sines/*
gsutil -m cp data/sines/*.csv gs://${BUCKET}/sines


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'for MODEL in linear dnn cnn rnn rnn2 rnnN; do\n  OUTDIR=gs://${BUCKET}/sinewaves/${MODEL}\n  JOBNAME=sines_${MODEL}_$(date -u +%y%m%d_%H%M%S)\n  gsutil -m rm -rf $OUTDIR\n  gcloud ml-engine jobs submit training $JOBNAME \\\n     --region=$REGION \\\n     --module-name=sinemodel.task \\\n     --package-path=${PWD}/sinemodel \\\n     --job-dir=$OUTDIR \\\n     --scale-tier=BASIC \\\n     --runtime-version=$TFVERSION \\\n     -- \\\n     --train_data_path="gs://${BUCKET}/sines/train*.csv" \\\n     --eval_data_path="gs://${BUCKET}/sines/valid*.csv"  \\\n     --output_dir=$OUTDIR \\\n     --train_steps=3000 --sequence_length=$SEQ_LEN --model=$MODEL\ndone')


# ## Monitor training with TensorBoard
# 
# Use this cell to launch tensorboard. If tensorboard appears blank try refreshing after 5 minutes

# In[ ]:


from google.datalab.ml import TensorBoard
TensorBoard().start('gs://{}/sinewaves'.format(BUCKET))


# In[ ]:


for pid in TensorBoard.list()['pid']:
  TensorBoard().stop(pid)
  print('Stopped TensorBoard with pid {}'.format(pid))


# ## Results
# 
# When I ran it, these were the RMSEs that I got for different models:
# 
# | Model | Sequence length | # of steps | Minutes | RMSE |
# | --- | ----| --- | --- | --- | 
# | linear | 50 | 3000 | 10 min | 0.150 |
# | dnn | 50 | 3000 | 10 min | 0.101 |
# | cnn | 50 | 3000 | 10 min | 0.105 |
# | rnn | 50 | 3000 | 11 min | 0.100 |
# | rnn2 | 50 | 3000 | 14 min |0.105 |
# | rnnN | 50 | 3000 | 15 min | 0.097 |
# 
# ### Analysis
# You can see there is a significant improvement when switching from the linear model to non-linear models. But within the the non-linear models (DNN/CNN/RNN) performance for all is pretty similar. 
# 
# Perhaps it's because this is too simple of a problem to require advanced deep learning models. In the next lab we'll deal with a problem where an RNN is more appropriate.

# Copyright 2017 Google Inc. Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License