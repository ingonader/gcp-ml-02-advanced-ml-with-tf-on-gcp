
# coding: utf-8

# # Flowers Image Classification with TensorFlow on Cloud ML Engine TPU
# 
# This notebook demonstrates how to do image classification from scratch on a flowers dataset using the Estimator API. Unlike [flowers_fromscratch.ipynb](the flowers_fromscratch notebook), here we do it on a TPU.
# 
# Therefore, this will work only if you have quota for TPUs (not in Qwiklabs). It will cost about $3 if you want to try it out.

# In[ ]:


get_ipython().run_line_magic('bash', '')
pip install apache-beam[gcp]


# After doing a pip install, click on Reset Session so that the Python environment picks up the new package

# In[155]:


import os
PROJECT = 'cloud-training-demos' # REPLACE WITH YOUR PROJECT ID
BUCKET = 'cloud-training-demos-ml' # REPLACE WITH YOUR BUCKET NAME
REGION = 'us-central1' # REPLACE WITH YOUR BUCKET REGION e.g. us-central1
MODEL_TYPE = 'tpu'

# do not change these
os.environ['PROJECT'] = PROJECT
os.environ['BUCKET'] = BUCKET
os.environ['REGION'] = REGION
os.environ['MODEL_TYPE'] = MODEL_TYPE
os.environ['TFVERSION'] = '1.8'  # Tensorflow version


# In[127]:


get_ipython().run_line_magic('bash', '')
gcloud config set project $PROJECT
gcloud config set compute/region $REGION


# ## Preprocess JPEG images to TF Records
# 
# While using a GPU, it is okay to read the JPEGS directly from our input_fn. However, TPUs are too fast and it will be very wasteful to have the TPUs wait on I/O. Therefore, we'll preprocess the JPEGs into TF Records.
# 
# This runs on Cloud Dataflow and will take <b> 15-20 minutes </b>

# In[3]:


get_ipython().run_line_magic('bash', '')
gsutil cat gs://cloud-ml-data/img/flower_photos/train_set.csv  | sed 's/,/ /g' | awk '{print $2}' | sort | uniq > /tmp/labels.txt


# In[37]:


get_ipython().run_line_magic('bash', '')
gsutil cat gs://cloud-ml-data/img/flower_photos/train_set.csv | wc -l
gsutil cat gs://cloud-ml-data/img/flower_photos/eval_set.csv | wc -l


# In[ ]:


get_ipython().run_line_magic('bash', '')
export PYTHONPATH=${PYTHONPATH}:${PWD}/flowersmodeltpu
gsutil -m rm -rf gs://${BUCKET}/tpu/flowers/data
python -m trainer.preprocess        --train_csv gs://cloud-ml-data/img/flower_photos/train_set.csv        --validation_csv gs://cloud-ml-data/img/flower_photos/eval_set.csv        --labels_file /tmp/labels.txt        --project_id $PROJECT        --output_dir gs://${BUCKET}/tpu/flowers/data


# In[6]:


get_ipython().run_line_magic('bash', '')
gsutil ls gs://${BUCKET}/tpu/flowers/data/


# ## Run as a Python module
# 
# First run locally without --use_tpu -- don't be concerned if the process gets killed for using too much memory.

# In[ ]:


get_ipython().run_line_magic('bash', '')
WITHOUT_TPU="--train_batch_size=2  --train_steps=5"
OUTDIR=./flowers_trained
rm -rf $OUTDIR
export PYTHONPATH=${PYTHONPATH}:${PWD}/flowersmodeltpu
python -m flowersmodeltpu.task    --output_dir=$OUTDIR    --num_train_images=3300    --num_eval_images=370    $WITHOUT_TPU    --learning_rate=0.01    --project=${PROJECT}    --train_data_path=gs://${BUCKET}/tpu/flowers/data/train*    --eval_data_path=gs://${BUCKET}/tpu/flowers/data/validation*


# Then, run it on Cloud ML Engine with --use_tpu

# In[153]:


get_ipython().run_line_magic('bash', '')
WITH_TPU="--train_batch_size=256  --train_steps=3000 --batch_norm --use_tpu"
WITHOUT_TPU="--train_batch_size=2  --train_steps=5"
OUTDIR=gs://${BUCKET}/flowers/trained_${MODEL_TYPE}_delete
JOBNAME=flowers_${MODEL_TYPE}_$(date -u +%y%m%d_%H%M%S)
echo $OUTDIR $REGION $JOBNAME
gsutil -m rm -rf $OUTDIR
gcloud ml-engine jobs submit training $JOBNAME    --region=$REGION    --module-name=flowersmodeltpu.task    --package-path=${PWD}/flowersmodeltpu    --job-dir=$OUTDIR    --staging-bucket=gs://$BUCKET    --scale-tier=BASIC_TPU    --runtime-version=$TFVERSION    --    --output_dir=$OUTDIR    --num_train_images=3300    --num_eval_images=370    $WITH_TPU    --learning_rate=0.01    --project=${PROJECT}    --train_data_path=gs://${BUCKET}/tpu/flowers/data/train-*    --eval_data_path=gs://${BUCKET}/tpu/flowers/data/validation-*


# In[154]:


get_ipython().run_line_magic('bash', '')
MODEL_LOCATION=$(gsutil ls gs://${BUCKET}/flowers/trained_${MODEL_TYPE}/export/exporter | tail -1)
saved_model_cli show --dir $MODEL_LOCATION --all


# ## Monitoring training with TensorBoard
# 
# Use this cell to launch tensorboard

# In[ ]:


from google.datalab.ml import TensorBoard
TensorBoard().start('gs://{}/flowers/trained_{}'.format(BUCKET, MODEL_TYPE))


# In[ ]:


for pid in TensorBoard.list()['pid']:
  TensorBoard().stop(pid)
  print 'Stopped TensorBoard with pid {}'.format(pid)


# ## Deploying and predicting with model
# 
# Deploy the model:

# In[158]:


get_ipython().run_line_magic('bash', '')
MODEL_NAME="flowers"
MODEL_VERSION=${MODEL_TYPE}
MODEL_LOCATION=$(gsutil ls gs://${BUCKET}/flowers/trained_${MODEL_TYPE}/export/exporter | tail -1)
echo "Deleting and deploying $MODEL_NAME $MODEL_VERSION from $MODEL_LOCATION ... this will take a few minutes"
#gcloud ml-engine versions delete --quiet ${MODEL_VERSION} --model ${MODEL_NAME}
#gcloud ml-engine models delete ${MODEL_NAME}
#gcloud ml-engine models create ${MODEL_NAME} --regions $REGION
gcloud alpha ml-engine versions create ${MODEL_VERSION} --machine-type mls1-c4-m4 --model ${MODEL_NAME} --origin ${MODEL_LOCATION} --runtime-version=$TFVERSION


# To predict with the model, let's take one of the example images that is available on Google Cloud Storage <img src="http://storage.googleapis.com/cloud-ml-data/img/flower_photos/sunflowers/1022552002_2b93faf9e7_n.jpg" />

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'gcloud alpha ml-engine models list')


# The online prediction service expects images to be base64 encoded as described [here](https://cloud.google.com/ml-engine/docs/tensorflow/online-predict#binary_data_in_prediction_input).

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'IMAGE_URL=gs://cloud-ml-data/img/flower_photos/sunflowers/1022552002_2b93faf9e7_n.jpg\n\n# Copy the image to local disk.\ngsutil cp $IMAGE_URL flower.jpg\n\n# Base64 encode and create request message in json format.\npython -c \'import base64, sys, json; img = base64.b64encode(open("flower.jpg", "rb").read()).decode(); print(json.dumps({"image_bytes":{"b64": img}}))\' &> request.json')


# Send it to the prediction service

# In[162]:


get_ipython().run_cell_magic('bash', '', 'gcloud ml-engine predict \\\n  --model=flowers2 \\\n  --version=${MODEL_TYPE} \\\n  --json-instances=./request.json')


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