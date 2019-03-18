
# coding: utf-8

# <h1> Text Classification using TensorFlow/Keras on Cloud ML Engine </h1>
# 
# This notebook illustrates:
# <ol>
# <li> Creating datasets for Machine Learning using BigQuery
# <li> Creating a text classification model using the Estimator API with a Keras model
# <li> Training on Cloud ML Engine
# <li> Deploying the model
# <li> Predicting with model
# <li> Rerun with pre-trained embedding
# </ol>

# In[1]:


# change these to try this notebook out
BUCKET = 'cloud-training-demos-ml'
PROJECT = 'cloud-training-demos'
REGION = 'us-central1'


# In[2]:


import os
os.environ['BUCKET'] = BUCKET
os.environ['PROJECT'] = PROJECT
os.environ['REGION'] = REGION
os.environ['TFVERSION'] = '1.8'


# In[5]:


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


# In[4]:


import tensorflow as tf
print(tf.__version__)


# We will look at the titles of articles and figure out whether the article came from the New York Times, TechCrunch or GitHub. 
# 
# We will use [hacker news](https://news.ycombinator.com/) as our data source. It is an aggregator that displays tech related headlines from various  sources.

# ### Creating Dataset from BigQuery 
# 
# Hacker news headlines are available as a BigQuery public dataset. The [dataset](https://bigquery.cloud.google.com/table/bigquery-public-data:hacker_news.stories?tab=details) contains all headlines from the sites inception in October 2006 until October 2015. 
# 
# Here is a sample of the dataset:

# In[7]:


import google.datalab.bigquery as bq
import pandas as pd

## set pandas display options to display all content 
## in each field, adding line-breaks as needed:
pd.set_option('display.max_colwidth', -1)

query="""
SELECT
  url, title, score
FROM
  `bigquery-public-data.hacker_news.stories`
WHERE
  LENGTH(title) > 10
  AND score > 10
LIMIT 10
"""
df = bq.Query(query).execute().result().to_dataframe()
df


# Let's do some regular expression parsing in BigQuery to get the source of the newspaper article from the URL. For example, if the url is http://mobile.nytimes.com/...., I want to be left with <i>nytimes</i>

# In[8]:


query="""
SELECT
  ARRAY_REVERSE(SPLIT(REGEXP_EXTRACT(url, '.*://(.[^/]+)/'), '.'))[OFFSET(1)] AS source,
  COUNT(title) AS num_articles
FROM
  `bigquery-public-data.hacker_news.stories`
WHERE
  REGEXP_CONTAINS(REGEXP_EXTRACT(url, '.*://(.[^/]+)/'), '.com$')
  AND LENGTH(title) > 10
GROUP BY
  source
ORDER BY num_articles DESC
LIMIT 10
"""
df = bq.Query(query).execute().result().to_dataframe()
df


# Now that we have good parsing of the URL to get the source, let's put together a dataset of source and titles. This will be our labeled dataset for machine learning.

# In[9]:


query="""
SELECT source, LOWER(REGEXP_REPLACE(title, '[^a-zA-Z0-9 $.-]', ' ')) AS title FROM
  (SELECT
    ARRAY_REVERSE(SPLIT(REGEXP_EXTRACT(url, '.*://(.[^/]+)/'), '.'))[OFFSET(1)] AS source,
    title
  FROM
    `bigquery-public-data.hacker_news.stories`
  WHERE
    REGEXP_CONTAINS(REGEXP_EXTRACT(url, '.*://(.[^/]+)/'), '.com$')
    AND LENGTH(title) > 10
  )
WHERE (source = 'github' OR source = 'nytimes' OR source = 'techcrunch')
"""

df = bq.Query(query + " LIMIT 10").execute().result().to_dataframe()
df.head()


# For ML training, we will need to split our dataset into training and evaluation datasets (and perhaps an independent test dataset if we are going to do model or feature selection based on the evaluation dataset).  
# 
# A simple, repeatable way to do this is to use the hash of a well-distributed column in our data (See https://www.oreilly.com/learning/repeatable-sampling-of-data-sets-in-bigquery-for-machine-learning).

# In[10]:


traindf = bq.Query(query + " AND MOD(ABS(FARM_FINGERPRINT(title)),4) > 0").execute().result().to_dataframe()
evaldf  = bq.Query(query + " AND MOD(ABS(FARM_FINGERPRINT(title)),4) = 0").execute().result().to_dataframe()


# Below we can see that roughly 75% of the data is used for training, and 25% for evaluation. 
# 
# We can also see that within each dataset, the classes are roughly balanced.

# In[11]:


print(traindf.shape)
print(evaldf.shape)


# In[14]:


traindf['source'].value_counts(normalize = True)


# In[15]:


evaldf['source'].value_counts(normalize = True)


# Finally we will save our data, which is currently in-memory, to disk.

# In[16]:


import os, shutil
DATADIR='data/txtcls'
shutil.rmtree(DATADIR, ignore_errors=True)
os.makedirs(DATADIR)
traindf.to_csv( os.path.join(DATADIR,'train.tsv'), header=False, index=False, encoding='utf-8', sep='\t')
evaldf.to_csv( os.path.join(DATADIR,'eval.tsv'), header=False, index=False, encoding='utf-8', sep='\t')


# In[17]:


get_ipython().system('head -3 data/txtcls/train.tsv')


# In[18]:


get_ipython().system('wc -l data/txtcls/*.tsv')


# ### TensorFlow/Keras Code
# 
# Please explore the code in this <a href="txtclsmodel/trainer">directory</a>: `model.py` contains the TensorFlow model and `task.py` parses command line arguments and launches off the training job.
# 
# There are some TODOs in the `model.py`, **make sure to complete the TODOs before proceeding!**

# In[19]:


get_ipython().system("grep -rnwi ./txtclsmodel/trainer/*.py -e 'todo'")


# ### Run Locally
# Let's make sure the code compiles by running locally for a fraction of an epoch

# In[21]:


get_ipython().run_cell_magic('bash', '', '## Make sure we have the latest version of Google Cloud Storage package\npip install --upgrade google-cloud-storage\nrm -rf txtcls_trained\ngcloud ml-engine local train \\\n   --module-name=trainer.task \\\n   --package-path=${PWD}/txtclsmodel/trainer \\\n   -- \\\n   --output_dir=${PWD}/txtcls_trained \\\n   --train_data_path=${PWD}/data/txtcls/train.tsv \\\n   --eval_data_path=${PWD}/data/txtcls/eval.tsv \\\n   --num_epochs=0.1')


# ### Train on the Cloud
# 
# Let's first copy our training data to the cloud:

# In[22]:


get_ipython().run_cell_magic('bash', '', 'gsutil cp data/txtcls/*.tsv gs://${BUCKET}/txtcls/')


# In[23]:


get_ipython().run_cell_magic('bash', '', 'OUTDIR=gs://${BUCKET}/txtcls/trained_fromscratch\nJOBNAME=txtcls_$(date -u +%y%m%d_%H%M%S)\ngsutil -m rm -rf $OUTDIR\ngcloud ml-engine jobs submit training $JOBNAME \\\n --region=$REGION \\\n --module-name=trainer.task \\\n --package-path=${PWD}/txtclsmodel/trainer \\\n --job-dir=$OUTDIR \\\n --scale-tier=BASIC_GPU \\\n --runtime-version=$TFVERSION \\\n -- \\\n --output_dir=$OUTDIR \\\n --train_data_path=gs://${BUCKET}/txtcls/train.tsv \\\n --eval_data_path=gs://${BUCKET}/txtcls/eval.tsv \\\n --num_epochs=5')


# ### Monitor training with TensorBoard
# If tensorboard appears blank try refreshing after 10 minutes

# In[24]:


from google.datalab.ml import TensorBoard
TensorBoard().start('gs://{}/txtcls/trained_fromscratch'.format(BUCKET))


# In[ ]:


for pid in TensorBoard.list()['pid']:
  TensorBoard().stop(pid)
  print('Stopped TensorBoard with pid {}'.format(pid))


# ### Results
# What accuracy did you get?

# Logs:
# 
# ```
# Saving dict for global step 2819: acc = 0.81215173, global_step = 2819, loss = 0.45492265
# ``` 

# ### Deploy trained model 
# 
# Once your training completes you will see your exported models in the output directory specified in Google Cloud Storage. 
# 
# You should see one model for each training checkpoint (default is every 1000 steps).

# In[25]:


get_ipython().run_cell_magic('bash', '', 'gsutil ls gs://${BUCKET}/txtcls/trained_fromscratch/export/exporter/')


# We will take the last export and deploy it as a REST API using Google Cloud Machine Learning Engine 

# In[26]:


get_ipython().run_cell_magic('bash', '', 'MODEL_NAME="txtcls"\nMODEL_VERSION="v1_fromscratch"\nMODEL_LOCATION=$(gsutil ls gs://${BUCKET}/txtcls/trained_fromscratch/export/exporter/ | tail -1)\n#gcloud ml-engine versions delete ${MODEL_VERSION} --model ${MODEL_NAME} --quiet\n#gcloud ml-engine models delete ${MODEL_NAME}\ngcloud ml-engine models create ${MODEL_NAME} --regions $REGION\ngcloud ml-engine versions create ${MODEL_VERSION} --model ${MODEL_NAME} --origin ${MODEL_LOCATION} --runtime-version=$TFVERSION')


# ### Get Predictions
# 
# Here are some actual hacker news headlines gathered from July 2018. These titles were not part of the training or evaluation datasets.

# In[27]:


techcrunch=[
  'Uber shuts down self-driving trucks unit',
  'Grover raises €37M Series A to offer latest tech products as a subscription',
  'Tech companies can now bid on the Pentagon’s $10B cloud contract'
]
nytimes=[
  '‘Lopping,’ ‘Tips’ and the ‘Z-List’: Bias Lawsuit Explores Harvard’s Admissions',
  'A $3B Plan to Turn Hoover Dam into a Giant Battery',
  'A MeToo Reckoning in China’s Workplace Amid Wave of Accusations'
]
github=[
  'Show HN: Moon – 3kb JavaScript UI compiler',
  'Show HN: Hello, a CLI tool for managing social media',
  'Firefox Nightly added support for time-travel debugging'
]


# Our serving input function expects the already tokenized representations of the headlines, so we do that pre-processing in the code before calling the REST API.
# 
# Note: Ideally we would do these transformation in the tensorflow graph directly instead of relying on separate client pre-processing code (see: [training-serving skew](https://developers.google.com/machine-learning/guides/rules-of-ml/#training_serving_skew)), howevever the pre-processing functions we're using are python functions so cannot be embedded in a tensorflow graph. 
# 
# See the <a href="../text_classification_native.ipynb">text_classification_native</a> notebook for a solution to this.

# In[28]:


import pickle
from tensorflow.python.keras.preprocessing import sequence
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
import json

requests = techcrunch+nytimes+github

# Tokenize and pad sentences using same mapping used in the deployed model
tokenizer = pickle.load( open( "txtclsmodel/tokenizer.pickled", "rb" ) )

requests_tokenized = tokenizer.texts_to_sequences(requests)
requests_tokenized = sequence.pad_sequences(requests_tokenized,maxlen=50)

# JSON format the requests
request_data = {'instances':requests_tokenized.tolist()}

# Authenticate and call CMLE prediction API 
credentials = GoogleCredentials.get_application_default()
api = discovery.build('ml', 'v1', credentials=credentials,
            discoveryServiceUrl='https://storage.googleapis.com/cloud-ml/discovery/ml_v1_discovery.json')

parent = 'projects/%s/models/%s' % (PROJECT, 'txtcls') #version is not specified so uses default
response = api.projects().predict(body=request_data, name=parent).execute()

# Format and print response
for i in range(len(requests)):
  print('\n{}'.format(requests[i]))
  print(' github    : {}'.format(response['predictions'][i]['dense_1'][0]))
  print(' nytimes   : {}'.format(response['predictions'][i]['dense_1'][1]))
  print(' techcrunch: {}'.format(response['predictions'][i]['dense_1'][2]))


# How many of your predictions were correct?

# ### Rerun with Pre-trained Embedding
# 
# In the previous model we trained our word embedding from scratch. Often times we get better performance and/or converge faster by leveraging a pre-trained embedding. This is a similar concept to transfer learning during image classification.
# 
# We will use the popular GloVe embedding which is trained on Wikipedia as well as various news sources like the New York Times.
# 
# You can read more about Glove at the project homepage: https://nlp.stanford.edu/projects/glove/
# 
# You can download the embedding files directly from the stanford.edu site, but we've rehosted it in a GCS bucket for faster download speed.

# In[29]:


get_ipython().system('gsutil cp gs://cloud-training-demos/courses/machine_learning/deepdive/09_sequence/text_classification/glove.6B.200d.txt gs://$BUCKET/txtcls/')


# Once the embedding is downloaded re-run your cloud training job with the added command line argument: 
# 
# ` --embedding_path=gs://${BUCKET}/txtcls/glove.6B.200d.txt`
# 
# Be sure to change your OUTDIR so it doesn't overwrite the previous model.
# 
# While the final accuracy may not change significantly, you should notice the model is able to converge to it much more quickly because it no longer has to learn an embedding from scratch.

# In[30]:


get_ipython().run_cell_magic('bash', '', 'OUTDIR=gs://${BUCKET}/txtcls/trained_withembeddings\nJOBNAME=txtcls_$(date -u +%y%m%d_%H%M%S)\ngsutil -m rm -rf $OUTDIR\ngcloud ml-engine jobs submit training $JOBNAME \\\n --region=$REGION \\\n --module-name=trainer.task \\\n --package-path=${PWD}/txtclsmodel/trainer \\\n --job-dir=$OUTDIR \\\n --scale-tier=BASIC_GPU \\\n --runtime-version=$TFVERSION \\\n -- \\\n --output_dir=$OUTDIR \\\n --train_data_path=gs://${BUCKET}/txtcls/train.tsv \\\n --eval_data_path=gs://${BUCKET}/txtcls/eval.tsv \\\n --embedding_path=gs://${BUCKET}/txtcls/glove.6B.200d.txt \\\n --num_epochs=5')


# Finals step:
# 
# ``` 
# Saving dict for global step 2819: acc = 0.8063949, global_step = 2819, loss = 0.50821364
# ``` 
# 

# #### References
# - This implementation is based on code from: https://github.com/google/eng-edu/tree/master/ml/guides/text_classification.
# - See the full text classification tutorial at: https://developers.google.com/machine-learning/guides/text-classification/

# Copyright 2017 Google Inc. Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License