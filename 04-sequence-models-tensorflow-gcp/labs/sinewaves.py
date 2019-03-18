
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

# In[2]:


# change these to try this notebook out
BUCKET = 'cloud-training-demos-ml'
PROJECT = 'cloud-training-demos'
REGION = 'us-central1'
SEQ_LEN = 50


# In[3]:


import os
os.environ['BUCKET'] = BUCKET
os.environ['PROJECT'] = PROJECT
os.environ['REGION'] = REGION
os.environ['SEQ_LEN'] = str(SEQ_LEN)
os.environ['TFVERSION'] = '1.8'


# In[4]:


import os
output = os.popen("gcloud config get-value project").readlines()
project_name = output[0][:-1]

# change these to try this notebook out
PROJECT = project_name
BUCKET = project_name
#BUCKET = BUCKET.replace("qwiklabs-gcp-", "inna-bckt-")
REGION = 'europe-west1'  ## note: Cloud ML Engine not availabe in europe-west3!

# set environment variables:
os.environ['BUCKET'] = BUCKET
os.environ['PROJECT'] = PROJECT
os.environ['REGION'] = REGION

print(PROJECT)
print(BUCKET)
print(REGION)


# <h3> Simulate some time-series data </h3>
# 
# Essentially a set of sinusoids with random amplitudes and frequencies.

# In[5]:


import tensorflow as tf
print(tf.__version__)


# In[6]:


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


# In[7]:


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
to_csv('data/sines/train-1.csv', 1000)  # 1000 sequences
to_csv('data/sines/valid-1.csv', 250)


# In[8]:


get_ipython().system('head -2 data/sines/*-1.csv')


# <h3> Train model locally </h3>
# 
# Make sure the code works as intended.
# 
# The `model.py` and `task.py` containing the model code is in <a href="sinemodel">sinemodel/</a>
# 
# **Complete the TODOs in `model.py` before proceeding!**
# 
# Once you've completed the TODOs, set `--model` below to the appropriate model (linear,dnn,cnn,rnn,rnn2 or rnnN) and run it locally for a few steps to test the code.

# In[14]:


get_ipython().run_line_magic('bash', '')
DATADIR=$(pwd)/data/sines
OUTDIR=$(pwd)/trained/sines
MODELNAME=rnn2

rm -rf $OUTDIR

gcloud ml-engine local train    --module-name=sinemodel.task    --package-path=${PWD}/sinemodel    --    --train_data_path="${DATADIR}/train-1.csv"    --eval_data_path="${DATADIR}/valid-1.csv"     --output_dir=${OUTDIR}    --train_steps=10 --sequence_length=$SEQ_LEN    --model=$MODELNAME
   ##--model=linear


# <h3> Cloud ML Engine </h3>
# 
# Now to train on Cloud ML Engine with more data.

# In[15]:


import shutil
shutil.rmtree('data/sines', ignore_errors=True)
os.makedirs('data/sines/')
for i in range(0,10):
  to_csv('data/sines/train-{}.csv'.format(i), 1000)  # 1000 sequences
  to_csv('data/sines/valid-{}.csv'.format(i), 250)


# In[16]:


get_ipython().run_line_magic('bash', '')
gsutil -m rm -rf gs://${BUCKET}/sines/*
gsutil -m cp data/sines/*.csv gs://${BUCKET}/sines


# In[20]:


get_ipython().run_cell_magic('bash', '', '## note: using --scale-tier=BASIC_GPU?\n#for MODEL in linear dnn cnn rnn rnn2 rnnN; do\nfor MODEL in rnn2; do\n  OUTDIR=gs://${BUCKET}/sinewaves/${MODEL}\n  JOBNAME=sines_${MODEL}_$(date -u +%y%m%d_%H%M%S)\n  gsutil -m rm -rf $OUTDIR\n  gcloud ml-engine jobs submit training $JOBNAME \\\n     --region=$REGION \\\n     --module-name=sinemodel.task \\\n     --package-path=${PWD}/sinemodel \\\n     --job-dir=$OUTDIR \\\n     --staging-bucket=gs://$BUCKET \\\n     --scale-tier=BASIC_GPU \\\n     --runtime-version=$TFVERSION \\\n     -- \\\n     --train_data_path="gs://${BUCKET}/sines/train*.csv" \\\n     --eval_data_path="gs://${BUCKET}/sines/valid*.csv"  \\\n     --output_dir=$OUTDIR \\\n     --train_steps=3000 --sequence_length=$SEQ_LEN --model=$MODEL \\\n     --eval_delay_secs=10 --min_eval_frequency=20      ## currently NOT hard-coded any longer\ndone')


# ## Monitor training with TensorBoard
# 
# Use this cell to launch tensorboard. If tensorboard appears blank try refreshing after 5 minutes

# In[21]:


from google.datalab.ml import TensorBoard
TensorBoard().start('gs://{}/sinewaves'.format(BUCKET))


# In[19]:


for pid in TensorBoard.list()['pid']:
  TensorBoard().stop(pid)
  print('Stopped TensorBoard with pid {}'.format(pid))


# ## Results
# 
# Complete the below table with your own results! Then compare your results to the results in the solution notebook.
# 
# | Model  | Sequence length | # of steps | Training time   | RMSE |
# | ------ | --------------- | ---------- | -------         | ---- | 
# | linear | 50              | 3000       |  13 mins 03 sec |   0.146422580 |
# | dnn    | 50              | 3000       |   9 mins 57 sec |   0.106585300 |
# | cnn    | 50              | 3000       |               - |    - |
# | rnn    | 50              | 3000       |  14 mins 46 sec |   0.104014740 |
# | rnn2   | 50              | 3000       |  18 mins 32 sec |   0.100508325 |
# | rnnN   | 50              | 3000       |               - |    - |
# 
# * Linear: Saving dict for global step 2939: RMSE = 0.14642258, RMSE_same_as_last = 0.30533966, global_step = 2939, loss = 0.021291882
# * Linear: Saving dict for global step 3000: RMSE = 0.1525632, RMSE_same_as_last = 0.30533966, global_step = 3000, loss = 0.023260117
# * DNN: Saving dict for global step 3000: RMSE = 0.1065853, RMSE_same_as_last = 0.31858918, global_step = 3000, loss = 0.011395868
# * CNN (trial001): Saving dict for global step 376: RMSE = 0.111769035, RMSE_same_as_last = 0.29965696, global_step = 376, loss = 0.012379423
# * RNN (trial001): Didn't quite finish training; Similar to CNN trial above, training took very long. Single steps took 30secs, and for every steps a checkpoint was written to disk. Didn't find anything in the code; Suspicion (not confirmed): Due to limited resources in the quicklab training accounts, GPUs got pre-empted and killed, hence the graph had to be rebuilt very, very often.
# * RNN (trial002): Ran perfectly fine. With no changes in the code, as far as I can reproduce. Saving dict for global step 3000: RMSE = 0.10401474, RMSE_same_as_last = 0.30925706, global_step = 3000, loss = 0.010779689
# * RNN2: Saving dict for global step 3000: RMSE = 0.100508325, RMSE_same_as_last = 0.30925706, global_step = 3000, loss = 0.010189519

# Copyright 2017 Google Inc. Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License