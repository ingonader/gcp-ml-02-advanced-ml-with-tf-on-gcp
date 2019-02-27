
# coding: utf-8

# # Image Classification from scratch with TPUs on Cloud ML Engine using ResNet
# 
# This notebook demonstrates how to do image classification from scratch on a flowers dataset using TPUs and the resnet trainer.

# In[ ]:


import os
PROJECT = 'cloud-training-demos' # REPLACE WITH YOUR PROJECT ID
BUCKET = 'cloud-training-demos-ml' # REPLACE WITH YOUR BUCKET NAME
REGION = 'us-central1' # REPLACE WITH YOUR BUCKET REGION e.g. us-central1

# do not change these
os.environ['PROJECT'] = PROJECT
os.environ['BUCKET'] = BUCKET
os.environ['REGION'] = REGION
os.environ['TFVERSION'] = '1.9'


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'gcloud config set project $PROJECT\ngcloud config set compute/region $REGION')


# ## Convert JPEG images to TensorFlow Records
# 
# My dataset consists of JPEG images in Google Cloud Storage. I have two CSV files that are formatted as follows:
#    image-name, category
# 
# Instead of reading the images from JPEG each time, we'll convert the JPEG data and store it as TF Records.
# 

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'gsutil cat gs://cloud-ml-data/img/flower_photos/train_set.csv | head -5 > /tmp/input.csv\ncat /tmp/input.csv')


# In[ ]:


get_ipython().run_cell_magic('bash', '', "gsutil cat gs://cloud-ml-data/img/flower_photos/train_set.csv  | sed 's/,/ /g' | awk '{print $2}' | sort | uniq > /tmp/labels.txt\ncat /tmp/labels.txt")


# ## Clone the TPU repo
# 
# Let's git clone the repo and get the preprocessing and model files. The model code has imports of the form:
# <pre>
# import resnet_model as model_lib
# </pre>
# We will need to change this to:
# <pre>
# from . import resnet_model as model_lib
# </pre>
# 

# In[ ]:


get_ipython().run_cell_magic('writefile', 'copy_resnet_files.sh', '#!/bin/bash\nrm -rf tpu\ngit clone https://github.com/tensorflow/tpu\ncd tpu\nTFVERSION=$1\necho "Switching to version r$TFVERSION"\ngit checkout r$TFVERSION\ncd ..\n  \nMODELCODE=tpu/models/official/resnet\nOUTDIR=mymodel\nrm -rf $OUTDIR\n\n# preprocessing\ncp -r imgclass $OUTDIR   # brings in setup.py and __init__.py\ncp tpu/tools/datasets/jpeg_to_tf_record.py $OUTDIR/trainer/preprocess.py\n\n# model: fix imports\nfor FILE in $(ls -p $MODELCODE | grep -v /); do\n    CMD="cat $MODELCODE/$FILE "\n    for f2 in $(ls -p $MODELCODE | grep -v /); do\n        MODULE=`echo $f2 | sed \'s/.py//g\'`\n        CMD="$CMD | sed \'s/^import ${MODULE}/from . import ${MODULE}/g\' "\n    done\n    CMD="$CMD > $OUTDIR/trainer/$FILE"\n    eval $CMD\ndone\nfind $OUTDIR\necho "Finished copying files into $OUTDIR"')


# In[ ]:


get_ipython().system('bash ./copy_resnet_files.sh $TFVERSION')


# ## Enable TPU service account
# 
# Allow Cloud ML Engine to access the TPU and bill to your project

# In[ ]:


get_ipython().run_cell_magic('writefile', 'enable_tpu_mlengine.sh', 'SVC_ACCOUNT=$(curl -H "Authorization: Bearer $(gcloud auth print-access-token)"  \\\n    https://ml.googleapis.com/v1/projects/${PROJECT}:getConfig \\\n              | grep tpuServiceAccount | tr \'"\' \' \' | awk \'{print $3}\' )\necho "Enabling TPU service account $SVC_ACCOUNT to act as Cloud ML Service Agent"\ngcloud projects add-iam-policy-binding $PROJECT \\\n    --member serviceAccount:$SVC_ACCOUNT --role roles/ml.serviceAgent\necho "Done"')


# In[ ]:


get_ipython().system('bash ./enable_tpu_mlengine.sh')


# ## Try preprocessing locally

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'export PYTHONPATH=${PYTHONPATH}:${PWD}/mymodel\n  \nrm -rf /tmp/out\npython -m trainer.preprocess \\\n       --train_csv /tmp/input.csv \\\n       --validation_csv /tmp/input.csv \\\n       --labels_file /tmp/labels.txt \\\n       --project_id $PROJECT \\\n       --output_dir /tmp/out --runner=DirectRunner')


# In[ ]:


get_ipython().system('ls -l /tmp/out')


# Now run it over full training and evaluation datasets.  This will happen in Cloud Dataflow.

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'export PYTHONPATH=${PYTHONPATH}:${PWD}/mymodel\ngsutil -m rm -rf gs://${BUCKET}/tpu/resnet/data\npython -m trainer.preprocess \\\n       --train_csv gs://cloud-ml-data/img/flower_photos/train_set.csv \\\n       --validation_csv gs://cloud-ml-data/img/flower_photos/eval_set.csv \\\n       --labels_file /tmp/labels.txt \\\n       --project_id $PROJECT \\\n       --output_dir gs://${BUCKET}/tpu/resnet/data')


# The above preprocessing step will take <b>15-20 minutes</b>. Wait for the job to finish before you proceed. Navigate to [Cloud Dataflow section of GCP web console](https://console.cloud.google.com/dataflow) to monitor job progress. You will see something like this <img src="dataflow.png" />

# Alternately, you can simply copy my already preprocessed files and proceed to the next step:
# <pre>
# gsutil -m cp gs://cloud-training-demos/tpu/resnet/data/* gs://${BUCKET}/tpu/resnet/copied_data
# </pre>

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'gsutil ls gs://${BUCKET}/tpu/resnet/data')


# ## Train on the Cloud

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'echo -n "--num_train_images=$(gsutil cat gs://cloud-ml-data/img/flower_photos/train_set.csv | wc -l)  "\necho -n "--num_eval_images=$(gsutil cat gs://cloud-ml-data/img/flower_photos/eval_set.csv | wc -l)  "\necho "--num_label_classes=$(cat /tmp/labels.txt | wc -l)"')


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'TOPDIR=gs://${BUCKET}/tpu/resnet\nOUTDIR=${TOPDIR}/trained\nJOBNAME=imgclass_$(date -u +%y%m%d_%H%M%S)\necho $OUTDIR $REGION $JOBNAME\ngsutil -m rm -rf $OUTDIR  # Comment out this line to continue training from the last time\ngcloud ml-engine jobs submit training $JOBNAME \\\n  --region=$REGION \\\n  --module-name=trainer.resnet_main \\\n  --package-path=$(pwd)/mymodel/trainer \\\n  --job-dir=$OUTDIR \\\n  --staging-bucket=gs://$BUCKET \\\n  --scale-tier=BASIC_TPU \\\n  --runtime-version=$TFVERSION --python-version=3.5 \\\n  -- \\\n  --data_dir=${TOPDIR}/data \\\n  --model_dir=${OUTDIR} \\\n  --resnet_depth=18 \\\n  --train_batch_size=128 --eval_batch_size=32 --skip_host_call=True \\\n  --steps_per_eval=250 --train_steps=1000 \\\n  --num_train_images=3300  --num_eval_images=370  --num_label_classes=5 \\\n  --export_dir=${OUTDIR}/export')


# The above training job will take 15-20 minutes. 
# Wait for the job to finish before you proceed. 
# Navigate to [Cloud ML Engine section of GCP web console](https://console.cloud.google.com/mlengine) 
# to monitor job progress.
# 
# The model should finish with a 80-83% accuracy (results will vary):
# ```
# Eval results: {'global_step': 1000, 'loss': 0.7359053, 'top_1_accuracy': 0.82954544, 'top_5_accuracy': 1.0}
# ```

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'gsutil ls gs://${BUCKET}/tpu/resnet/trained/export/')


# You can look at the training charts with TensorBoard:

# In[ ]:


OUTDIR = 'gs://{}/tpu/resnet/trained/'.format(BUCKET)
from google.datalab.ml import TensorBoard
TensorBoard().start(OUTDIR)


# In[ ]:


TensorBoard().stop(11531)
print("Stopped Tensorboard")


# These were the charts I got (I set smoothing to be zero):
# <img src="resnet_traineval.png" height="50"/>
# As you can see, the final blue dot (eval) is quite close to the lowest training loss, indicating that the model hasn't overfit.  The top_1 accuracy on the evaluation dataset, however, is 80% which isn't that great. More data would help.
# <img src="resnet_accuracy.png" height="50"/>

# ## Deploying and predicting with model
# 
# Deploy the model:

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'MODEL_NAME="flowers"\nMODEL_VERSION=resnet\nMODEL_LOCATION=$(gsutil ls gs://${BUCKET}/tpu/resnet/trained/export/ | tail -1)\necho "Deleting/deploying $MODEL_NAME $MODEL_VERSION from $MODEL_LOCATION ... this will take a few minutes"\n\n# comment/uncomment the appropriate line to run. The first time around, you will need only the two create calls\n# But during development, you might need to replace a version by deleting the version and creating it again\n\n#gcloud ml-engine versions delete --quiet ${MODEL_VERSION} --model ${MODEL_NAME}\n#gcloud ml-engine models delete ${MODEL_NAME}\ngcloud ml-engine models create ${MODEL_NAME} --regions $REGION\ngcloud ml-engine versions create ${MODEL_VERSION} --model ${MODEL_NAME} --origin ${MODEL_LOCATION} --runtime-version=$TFVERSION')


# We can use saved_model_cli to find out what inputs the model expects:

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'saved_model_cli show --dir $(gsutil ls gs://${BUCKET}/tpu/resnet/trained/export/ | tail -1) --tag_set serve --signature_def serving_default')


# As you can see, the model expects image_bytes.  This is typically base64 encoded

# To predict with the model, let's take one of the example images that is available on Google Cloud Storage <img src="http://storage.googleapis.com/cloud-ml-data/img/flower_photos/sunflowers/1022552002_2b93faf9e7_n.jpg" /> and convert it to a base64-encoded array

# In[ ]:


import base64, sys, json
import tensorflow as tf
import io
with tf.gfile.GFile('gs://cloud-ml-data/img/flower_photos/sunflowers/1022552002_2b93faf9e7_n.jpg', 'rb') as ifp:
  with io.open('test.json', 'w') as ofp:
    image_data = ifp.read()
    img = base64.b64encode(image_data).decode('utf-8')
    json.dump({"image_bytes": {"b64": img}}, ofp)


# In[ ]:


get_ipython().system('ls -l test.json')


# Send it to the prediction service

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'gcloud ml-engine predict --model=flowers --version=resnet --json-instances=./test.json')


# What does CLASS no. 3 correspond to? (remember that classes is 0-based)

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'head -4 /tmp/labels.txt | tail -1')


# Here's how you would invoke those predictions without using gcloud

# In[ ]:


from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
import base64, sys, json
import tensorflow as tf

with tf.gfile.GFile('gs://cloud-ml-data/img/flower_photos/sunflowers/1022552002_2b93faf9e7_n.jpg', 'rb') as ifp:
  credentials = GoogleCredentials.get_application_default()
  api = discovery.build('ml', 'v1', credentials=credentials,
            discoveryServiceUrl='https://storage.googleapis.com/cloud-ml/discovery/ml_v1_discovery.json')
  
  request_data = {'instances':
  [
      {"image_bytes": {"b64": base64.b64encode(ifp.read()).decode('utf-8')}}
  ]}

  parent = 'projects/%s/models/%s/versions/%s' % (PROJECT, 'flowers', 'resnet')
  response = api.projects().predict(body=request_data, name=parent).execute()
  print("response={0}".format(response))


# <pre>
# # Copyright 2018 Google Inc. All Rights Reserved.
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