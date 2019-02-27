
# coding: utf-8

# # Flowers Image Classification with TensorFlow on Cloud ML Engine
# 
# This notebook demonstrates how to do image classification from scratch on a flowers dataset using the Estimator API.

# In[1]:


import os
PROJECT = 'cloud-training-demos' # REPLACE WITH YOUR PROJECT ID
BUCKET = 'cloud-training-demos-ml' # REPLACE WITH YOUR BUCKET NAME
REGION = 'us-central1' # REPLACE WITH YOUR BUCKET REGION e.g. us-central1
MODEL_TYPE = 'cnn'

# do not change these
os.environ['PROJECT'] = PROJECT
os.environ['BUCKET'] = BUCKET
os.environ['REGION'] = REGION
os.environ['MODEL_TYPE'] = MODEL_TYPE
os.environ['TFVERSION'] = '1.8'  # Tensorflow version


# In[2]:


## in datalabvm cloud datalab notebook:
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
print("gsutil mb -l {0} gs://{1}".format(REGION, BUCKET))


# In[3]:


get_ipython().run_line_magic('bash', '')
gcloud config set project $PROJECT
gcloud config set compute/region $REGION


# ## Input functions to read JPEG images
# 
# The key difference between this notebook and [the MNIST one](./mnist_models.ipynb) is in the input function.
# In the input function here, we are doing the following:
# * Reading JPEG images, rather than 2D integer arrays.
# * Reading in batches of batch_size images rather than slicing our in-memory structure to be batch_size images.
# * Resizing the images to the expected HEIGHT, WIDTH. Because this is a real-world dataset, the images are of different sizes. We need to preprocess the data to, at the very least, resize them to constant size.

# ## Run as a Python module
# 
# Since we want to run our code on Cloud ML Engine, we've packaged it as a python module.
# 
# The `model.py` and `task.py` containing the model code is in <a href="flowersmodel">flowersmodel</a>
# 
# **Complete the TODOs in `model.py` before proceeding!**
# 
# Once you've completed the TODOs, run it locally for a few steps to test the code.

# In[7]:


get_ipython().run_line_magic('bash', '')
rm -rf flowersmodel.tar.gz flowers_trained
gcloud ml-engine local train    --module-name=flowersmodel.task    --package-path=${PWD}/flowersmodel    --    --output_dir=${PWD}/flowers_trained    --train_steps=5    --learning_rate=0.01    --batch_size=2    --model=$MODEL_TYPE    --augment    --train_data_path=gs://cloud-ml-data/img/flower_photos/train_set.csv    --eval_data_path=gs://cloud-ml-data/img/flower_photos/eval_set.csv


# In[13]:


get_ipython().system('gsutil cat -r 0-480 gs://cloud-ml-data/img/flower_photos/train_set.csv')


# In[15]:


get_ipython().system('gsutil cat -r 0-500 gs://cloud-ml-data/img/flower_photos/eval_set.csv')


# Now, let's do it on ML Engine. Note the --model parameter

# In[16]:


get_ipython().run_line_magic('bash', '')
OUTDIR=gs://${BUCKET}/flowers/trained_${MODEL_TYPE}
JOBNAME=flowers_${MODEL_TYPE}_$(date -u +%y%m%d_%H%M%S)
echo $OUTDIR $REGION $JOBNAME
gsutil -m rm -rf $OUTDIR
gcloud ml-engine jobs submit training $JOBNAME    --region=$REGION    --module-name=flowersmodel.task    --package-path=${PWD}/flowersmodel    --job-dir=$OUTDIR    --staging-bucket=gs://$BUCKET    --scale-tier=BASIC_GPU    --runtime-version=$TFVERSION    --    --output_dir=$OUTDIR    --train_steps=1000    --learning_rate=0.01    --batch_size=40    --model=$MODEL_TYPE    --augment    --batch_norm    --train_data_path=gs://cloud-ml-data/img/flower_photos/train_set.csv    --eval_data_path=gs://cloud-ml-data/img/flower_photos/eval_set.csv


# ## Monitoring training with TensorBoard
# 
# Use this cell to launch tensorboard

# In[17]:


from google.datalab.ml import TensorBoard
TensorBoard().start('gs://{}/flowers/trained_{}'.format(BUCKET, MODEL_TYPE))


# In[ ]:


for pid in TensorBoard.list()['pid']:
  TensorBoard().stop(pid)
  print('Stopped TensorBoard with pid {}'.format(pid))


# Here are my results:
# 
# 
# Model | Accuracy | Time taken | Run time parameters
# --- | :---: | --- | ---
# cnn with batch-norm | 0.582 | 47 min | 1000 steps, LR=0.01, Batch=40
# as above, plus augment | 0.615 | 3 hr | 5000 steps, LR=0.01, Batch=40
# 
# 
# What was your accuracy?

# Model | Accuracy | Time taken | Run time parameters
# --- | :---: | --- | ---
# cnn with batch-norm and augment | NA |  NA | 1000 steps, LR=0.01, Batch=40
# 
# 

# ## Deploying and predicting with model
# 
# Deploy the model:

# In[ ]:


get_ipython().run_line_magic('bash', '')
MODEL_NAME="flowers"
MODEL_VERSION=${MODEL_TYPE}
MODEL_LOCATION=$(gsutil ls gs://${BUCKET}/flowers/trained_${MODEL_TYPE}/export/exporter | tail -1)
echo "Deleting and deploying $MODEL_NAME $MODEL_VERSION from $MODEL_LOCATION ... this will take a few minutes"
#gcloud ml-engine versions delete --quiet ${MODEL_VERSION} --model ${MODEL_NAME}
#gcloud ml-engine models delete ${MODEL_NAME}
gcloud ml-engine models create ${MODEL_NAME} --regions $REGION
gcloud ml-engine versions create ${MODEL_VERSION} --model ${MODEL_NAME} --origin ${MODEL_LOCATION} --runtime-version=$TFVERSION


# To predict with the model, let's take one of the example images that is available on Google Cloud Storage <img src="http://storage.googleapis.com/cloud-ml-data/img/flower_photos/sunflowers/1022552002_2b93faf9e7_n.jpg" />

# The online prediction service expects images to be base64 encoded as described [here](https://cloud.google.com/ml-engine/docs/tensorflow/online-predict#binary_data_in_prediction_input).

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'IMAGE_URL=gs://cloud-ml-data/img/flower_photos/sunflowers/1022552002_2b93faf9e7_n.jpg\n\n# Copy the image to local disk.\ngsutil cp $IMAGE_URL flower.jpg\n\n# Base64 encode and create request message in json format.\npython -c \'import base64, sys, json; img = base64.b64encode(open("flower.jpg", "rb").read()).decode(); print(json.dumps({"image_bytes":{"b64": img}}))\' &> request.json')


# Send it to the prediction service

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'gcloud ml-engine predict \\\n  --model=flowers \\\n  --version=${MODEL_TYPE} \\\n  --json-instances=./request.json')


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