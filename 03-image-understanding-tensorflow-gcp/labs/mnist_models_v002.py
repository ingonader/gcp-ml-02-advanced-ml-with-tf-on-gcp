
# coding: utf-8

# # MNIST Image Classification with TensorFlow on Cloud ML Engine
# 
# This notebook demonstrates how to implement different image models on MNIST using Estimator. 
# 
# Note the MODEL_TYPE; change it to try out different models

# In[5]:


import os
PROJECT = 'cloud-training-demos' # REPLACE WITH YOUR PROJECT ID
BUCKET = 'cloud-training-demos-ml' # REPLACE WITH YOUR BUCKET NAME
REGION = 'us-central1' # REPLACE WITH YOUR BUCKET REGION e.g. us-central1
MODEL_TYPE='dnn_dropout'  # 'linear', 'dnn', 'dnn_dropout', or 'cnn'

# do not change these
os.environ['PROJECT'] = PROJECT
os.environ['BUCKET'] = BUCKET
os.environ['REGION'] = REGION
os.environ['MODEL_TYPE'] = MODEL_TYPE
os.environ['TFVERSION'] = '1.8'  # Tensorflow version


# In[6]:


import os
output = os.popen("gcloud config get-value project").readlines()
project_name = output[0][:-1]

# change these to try this notebook out
PROJECT = project_name
BUCKET = project_name
BUCKET = BUCKET.replace("qwiklabs-gcp-", "inna-bckt-")
REGION = 'europe-west1'  ## note: Cloud ML Engine not availabe in europe-west3!

# set environment variables:
os.environ['BUCKET'] = BUCKET
os.environ['PROJECT'] = PROJECT
os.environ['REGION'] = REGION

print(PROJECT)
print(BUCKET)
print("gsutil mb -l {0} gs://{1}".format(REGION, BUCKET))

## set config for gcp config: [[?]]
print(os.popen("gcloud config set project $PROJECT").readlines())
print(os.popen("gcloud config set compute/region $REGION").readlines())


# In[7]:


get_ipython().run_line_magic('bash', '')
gcloud config set project $PROJECT
gcloud config set compute/region $REGION


# ## Run as a Python module
# 
# In the previous notebook (mnist_linear.ipynb) we ran our code directly from the notebook.
# 
# Now since we want to run our code on Cloud ML Engine, we've packaged it as a python module.
# 
# The `model.py` and `task.py` containing the model code is in <a href="mnistmodel/trainer">mnistmodel/trainer</a>
# 
# **Complete the TODOs in `model.py` before proceeding!**
# 
# Once you've completed the TODOs, set MODEL_TYPE and run it locally for a few steps to test the code.

# In[8]:


get_ipython().system('echo $MODEL_TYPE')


# In[9]:


get_ipython().run_line_magic('bash', '')
rm -rf mnistmodel.tar.gz mnist_trained
gcloud ml-engine local train    --module-name=trainer.task    --package-path=${PWD}/mnistmodel/trainer    --    --output_dir=${PWD}/mnist_trained    --train_steps=100    --learning_rate=0.01    --model=$MODEL_TYPE


# **Now, let's do it on Cloud ML Engine so we can train on GPU:** `--scale-tier=BASIC_GPU`
# 
# Note the GPU speed up depends on the model type. You'll notice the more complex CNN model trains significantly faster on GPU, however the speed up on the simpler models is not as pronounced.

# In[10]:


get_ipython().run_line_magic('bash', '')
OUTDIR=gs://${BUCKET}/mnist/trained_${MODEL_TYPE}
JOBNAME=mnist_${MODEL_TYPE}_$(date -u +%y%m%d_%H%M%S)
echo $OUTDIR $REGION $JOBNAME
gsutil -m rm -rf $OUTDIR
gcloud ml-engine jobs submit training $JOBNAME    --region=$REGION    --module-name=trainer.task    --package-path=${PWD}/mnistmodel/trainer    --job-dir=$OUTDIR    --staging-bucket=gs://$BUCKET    --scale-tier=BASIC_GPU    --runtime-version=$TFVERSION    --    --output_dir=$OUTDIR    --train_steps=10000 --learning_rate=0.01 --train_batch_size=512    --model=$MODEL_TYPE --batch_norm


# ## Monitoring training with TensorBoard
# 
# Use this cell to launch tensorboard

# In[11]:


from google.datalab.ml import TensorBoard
TensorBoard().start('gs://{}/mnist/trained_{}'.format(BUCKET, MODEL_TYPE))


# In[11]:


for pid in TensorBoard.list()['pid']:
  TensorBoard().stop(pid)
  print('Stopped TensorBoard with pid {}'.format(pid))


# Here are my results:
# 
# ```
# Model | Accuracy | Time taken | Model description | Run time parameters
# --- | :---: | ---
# linear | 91.53 | 3 min | linear | 100 steps, LR=0.01, Batch=512
# linear | 92.73 | 8 min | linear | 1000 steps, LR=0.01, Batch=512
# linear | 92.29 | 18 min | linear | 10000 steps, LR=0.01, Batch=512
# dnn | 98.14 | 15 min | 300-100-30 nodes fully connected | 10000 steps, LR=0.01, Batch=512
# dnn | 97.99 | 48 min | 300-100-30 nodes fully connected | 100000 steps, LR=0.01, Batch=512
# dnn_dropout | 97.84 | 29 min | 300-100-30-DL(0.1)- nodes | 20000 steps, LR=0.01, Batch=512
# cnn | 98.97 | 35 min | maxpool(10 5x5 cnn, 2)-maxpool(20 5x5 cnn, 2)-300-DL(0.25) | 20000 steps, LR=0.01, Batch=512
# cnn | 98.93 | 35 min | maxpool(10 11x11 cnn, 2)-maxpool(20 3x3 cnn, 2)-300-DL(0.25) | 20000 steps, LR=0.01, Batch=512
# cnn | 99.17 | 35 min | maxpool(10 11x11 cnn, 2)-maxpool(20 3x3 cnn, 2)-300-DL(0.25), batch_norm (logits only) | 20000 steps, LR=0.01, Batch=512
# cnn | 99.27 | 35 min | maxpool(10 11x11 cnn, 2)-maxpool(20 3x3 cnn, 2)-300-DL(0.25), batch_norm (logits, deep) | 10000 steps, LR=0.01, Batch=512
# cnn | 99.48 | 12 hr | as-above but nfil1=20, nfil2=27, dprob=0.1, lr=0.001, batchsize=233 | (hyperparameter optimization)
# ```
# 
# Create a table to keep track of your own results as you experiment with model type and hyperparameters!

# From logs in Cloud ML GCP Page:
# 
# ```
# | Model            | Loss for final step |
# | ---------------- | -----------------  |
# | DNN              | 0.0024385566        |
# | DNN with dropout | 0.08717914          | Saving dict for global step 10000: accuracy = 0.9714, global_step = 10000, loss = 0.19375572
# ```

# ## Deploying and predicting with model
# 
# Deploy the model:

# In[12]:


get_ipython().run_line_magic('bash', '')
MODEL_NAME="mnist"
MODEL_VERSION=${MODEL_TYPE}
MODEL_LOCATION=$(gsutil ls gs://${BUCKET}/mnist/trained_${MODEL_TYPE}/export/exporter | tail -1)
echo "Deleting and deploying $MODEL_NAME $MODEL_VERSION from $MODEL_LOCATION ... this will take a few minutes"
#gcloud ml-engine versions delete ${MODEL_VERSION} --model ${MODEL_NAME}
#gcloud ml-engine models delete ${MODEL_NAME}
gcloud ml-engine models create ${MODEL_NAME} --regions $REGION
gcloud ml-engine versions create ${MODEL_VERSION} --model ${MODEL_NAME} --origin ${MODEL_LOCATION} --runtime-version=$TFVERSION


# To predict with the model, let's take one of the example images.

# In[13]:


import json, codecs
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

HEIGHT=28
WIDTH=28

mnist = input_data.read_data_sets('mnist/data', one_hot=True, reshape=False)
IMGNO=5 #CHANGE THIS to get different images
jsondata = {'image': mnist.test.images[IMGNO].reshape(HEIGHT, WIDTH).tolist()}
json.dump(jsondata, codecs.open('test.json', 'w', encoding='utf-8'))
plt.imshow(mnist.test.images[IMGNO].reshape(HEIGHT, WIDTH));


# Send it to the prediction service

# In[14]:


get_ipython().run_line_magic('bash', '')
gcloud ml-engine predict    --model=mnist    --version=${MODEL_TYPE}    --json-instances=./test.json


# In[31]:


get_ipython().run_line_magic('bash', '')
## download model to local file system:

## not:
##inna-bckt-120695964d64ec72/mnist_dnn_dropout_190221_141035/9cf3b9b97d342bdc3693fa2db71bb3f23181dcebbe3abca3cb67b16374b7bcb8
#gsutil cp -r gs://$BUCKET/mnist_dnn_dropout_190221_141035/9cf3b9b97d342bdc3693fa2db71bb3f23181dcebbe3abca3cb67b16374b7bcb8/* .

##inna-bckt-120695964d64ec72/mnist/trained_dnn_dropout
gsutil cp -r gs://$BUCKET/mnist/trained_dnn_dropout .


# In[32]:


ls -l


# In[35]:


get_ipython().run_line_magic('bash', '')
ls -l ./trained_dnn_dropout/export/exporter


# In[37]:


## find lates model on local disk:
from pathlib import Path

export_dir = 'trained_dnn_dropout/export/exporter'
subdirs = [x for x in Path(export_dir).iterdir()
           if x.is_dir() and 'temp' not in str(x)]
latest = str(sorted(subdirs)[-1])

print(latest)


# In[38]:


## load model from local disk and make predictions:
from tensorflow.contrib import predictor

predict_fn = predictor.from_saved_model(latest)

pred = predict_fn({'image': mnist.test.images[0:2, :, :, 0]})
print(pred)
print(pred['classes'])


# In[61]:


## make pandas dataframe from predictions:
import pandas as pd
import numpy as np

pred = predict_fn({'image': mnist.test.images[:, :, :, 0]})['classes']

dat_pred = pd.DataFrame({'label': np.argmax(mnist.test.labels, axis = 1),
                         'pred' : pred})
dat_pred['correct'] = (dat_pred['label'] == dat_pred['pred'])
print(dat_pred.head(n = 5))


acc = dat_pred['correct'].mean()
print('Accuracy = ', acc)

print('\n')
print('Accuracy by label:')
print(dat_pred.groupby(['label'])[['correct']].mean())


# <pre>
# # Copyright 2017 Google Inc. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #      http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# </pre>