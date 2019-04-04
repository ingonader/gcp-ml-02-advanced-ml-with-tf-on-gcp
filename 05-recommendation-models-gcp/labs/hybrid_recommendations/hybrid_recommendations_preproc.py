
# coding: utf-8

# # Neural network hybrid recommendation system on Google Analytics data preprocessing
# 
# This notebook demonstrates how to implement a hybrid recommendation system using a neural network to combine content-based and collaborative filtering recommendation models using Google Analytics data. We are going to use the learned user embeddings from [wals.ipynb](../wals.ipynb) and combine that with our previous content-based features from [content_based_using_neural_networks.ipynb](../content_based_using_neural_networks.ipynb)
# 
# First we are going to preprocess our data using BigQuery and Cloud Dataflow to be used in our later neural network hybrid recommendation model.

# Apache Beam only works in Python 2 at the moment, so we're going to switch to the Python 2 kernel. In the above menu, click the dropdown arrow and select `python2`.

# In[1]:


get_ipython().run_cell_magic('bash', '', 'source activate py2env\npip uninstall -y google-cloud-dataflow\nconda install -y pytz==2018.4\npip install apache-beam[gcp]')


# In[2]:


get_ipython().run_cell_magic('bash', '', 'conda update -n base -c defaults conda')


# In[3]:


get_ipython().run_cell_magic('bash', '', 'pip install google-cloud-dataflow')


# In[4]:


get_ipython().run_cell_magic('bash', '', 'source activate py2env\npip uninstall -y google-cloud-dataflow\nconda install -y pytz==2018.4\npip install apache-beam[gcp]')


# Now restart notebook's session kernel!

# In[1]:


# Import helpful libraries and setup our project, bucket, and region
import os

output = os.popen("gcloud config get-value project").readlines()
project_name = output[0][:-1]

# change these to try this notebook out
PROJECT = project_name
BUCKET = project_name
#BUCKET = BUCKET.replace("qwiklabs-gcp-", "inna-bckt-")
REGION = 'europe-west1'  ## note: Cloud ML Engine not availabe in europe-west3!

print(PROJECT)
print(BUCKET)
print(REGION)

# do not change these
os.environ['PROJECT'] = PROJECT
os.environ['BUCKET'] = BUCKET
os.environ['REGION'] = REGION
os.environ['TFVERSION'] = '1.8'


# In[2]:


get_ipython().run_cell_magic('bash', '', 'gcloud config set project $PROJECT\ngcloud config set compute/region $REGION')


# <h2> Create ML dataset using Dataflow </h2>
# Let's use Cloud Dataflow to read in the BigQuery data, do some preprocessing, and write it out as CSV files.
# 
# First, let's create our hybrid dataset query that we will use in our Cloud Dataflow pipeline. This will combine some content-based features and the user and item embeddings learned from our WALS Matrix Factorization Collaborative filtering lab that we extracted from our trained WALSMatrixFactorization Estimator and uploaded to BigQuery.

# In[3]:


query_hybrid_dataset = """
WITH CTE_site_history AS (
  SELECT
      fullVisitorId as visitor_id,
      (SELECT MAX(IF(index = 10, value, NULL)) FROM UNNEST(hits.customDimensions)) AS content_id,
      (SELECT MAX(IF(index = 7, value, NULL)) FROM UNNEST(hits.customDimensions)) AS category, 
      (SELECT MAX(IF(index = 6, value, NULL)) FROM UNNEST(hits.customDimensions)) AS title,
      (SELECT MAX(IF(index = 2, value, NULL)) FROM UNNEST(hits.customDimensions)) AS author_list,
      SPLIT(RPAD((SELECT MAX(IF(index = 4, value, NULL)) FROM UNNEST(hits.customDimensions)), 7), '.') AS year_month_array,
      LEAD(hits.customDimensions, 1) OVER (PARTITION BY fullVisitorId ORDER BY hits.time ASC) AS nextCustomDimensions
  FROM 
    `cloud-training-demos.GA360_test.ga_sessions_sample`,   
     UNNEST(hits) AS hits
   WHERE 
     # only include hits on pages
      hits.type = "PAGE"
      AND
      fullVisitorId IS NOT NULL
      AND
      hits.time != 0
      AND
      hits.time IS NOT NULL
      AND
      (SELECT MAX(IF(index = 10, value, NULL)) FROM UNNEST(hits.customDimensions)) IS NOT NULL
),
CTE_training_dataset AS (
SELECT
  (SELECT MAX(IF(index=10, value, NULL)) FROM UNNEST(nextCustomDimensions)) AS next_content_id,
  
  visitor_id,
  content_id,
  category,
  REGEXP_REPLACE(title, r",", "") AS title,
  REGEXP_EXTRACT(author_list, r"^[^,]+") AS author,
  DATE_DIFF(DATE(CAST(year_month_array[OFFSET(0)] AS INT64), CAST(year_month_array[OFFSET(1)] AS INT64), 1), DATE(1970, 1, 1), MONTH) AS months_since_epoch
FROM
  CTE_site_history
WHERE (SELECT MAX(IF(index=10, value, NULL)) FROM UNNEST(nextCustomDimensions)) IS NOT NULL)

SELECT
  CAST(next_content_id AS STRING) AS next_content_id,
  
  CAST(training_dataset.visitor_id AS STRING) AS visitor_id,
  CAST(training_dataset.content_id AS STRING) AS content_id,
  CAST(IFNULL(category, 'None') AS STRING) AS category,
  CONCAT("\\"", REPLACE(TRIM(CAST(IFNULL(title, 'None') AS STRING)), "\\"",""), "\\"") AS title,
  CAST(IFNULL(author, 'None') AS STRING) AS author,
  CAST(months_since_epoch AS STRING) AS months_since_epoch,
  
  IFNULL(user_factors._0, 0.0) AS user_factor_0,
  IFNULL(user_factors._1, 0.0) AS user_factor_1,
  IFNULL(user_factors._2, 0.0) AS user_factor_2,
  IFNULL(user_factors._3, 0.0) AS user_factor_3,
  IFNULL(user_factors._4, 0.0) AS user_factor_4,
  IFNULL(user_factors._5, 0.0) AS user_factor_5,
  IFNULL(user_factors._6, 0.0) AS user_factor_6,
  IFNULL(user_factors._7, 0.0) AS user_factor_7,
  IFNULL(user_factors._8, 0.0) AS user_factor_8,
  IFNULL(user_factors._9, 0.0) AS user_factor_9,
  
  IFNULL(item_factors._0, 0.0) AS item_factor_0,
  IFNULL(item_factors._1, 0.0) AS item_factor_1,
  IFNULL(item_factors._2, 0.0) AS item_factor_2,
  IFNULL(item_factors._3, 0.0) AS item_factor_3,
  IFNULL(item_factors._4, 0.0) AS item_factor_4,
  IFNULL(item_factors._5, 0.0) AS item_factor_5,
  IFNULL(item_factors._6, 0.0) AS item_factor_6,
  IFNULL(item_factors._7, 0.0) AS item_factor_7,
  IFNULL(item_factors._8, 0.0) AS item_factor_8,
  IFNULL(item_factors._9, 0.0) AS item_factor_9,
  
  FARM_FINGERPRINT(CONCAT(CAST(visitor_id AS STRING), CAST(content_id AS STRING))) AS hash_id
FROM CTE_training_dataset AS training_dataset
LEFT JOIN `cloud-training-demos.GA360_test.user_factors` AS user_factors
  ON CAST(training_dataset.visitor_id AS FLOAT64) = CAST(user_factors.user_id AS FLOAT64)
LEFT JOIN `cloud-training-demos.GA360_test.item_factors` AS item_factors
  ON CAST(training_dataset.content_id AS STRING) = CAST(item_factors.item_id AS STRING)
"""


# Let's pull a sample of our data into a dataframe to see what it looks like.

# In[4]:


import google.datalab.bigquery as bq
df_hybrid_dataset = bq.Query(query_hybrid_dataset + "LIMIT 100").execute().result().to_dataframe()
df_hybrid_dataset.head()


# In[5]:


df_hybrid_dataset.describe()


# In[10]:


import apache_beam as beam
import datetime, os

def to_csv(rowdict):
  # Pull columns from BQ and create a line
  import hashlib
  import copy
  CSV_COLUMNS = 'next_content_id,visitor_id,content_id,category,title,author,months_since_epoch'.split(',')
  FACTOR_COLUMNS = ["user_factor_{}".format(i) for i in range(10)] + ["item_factor_{}".format(i) for i in range(10)]
    
  # Write out rows for each input row for each column in rowdict
  data = ','.join(['None' if k not in rowdict else (rowdict[k].encode('utf-8') if rowdict[k] is not None else 'None') for k in CSV_COLUMNS])
  data += ','
  data += ','.join([str(rowdict[k]) if k in rowdict else 'None' for k in FACTOR_COLUMNS])
  yield ('{}'.format(data))
  
def preprocess(in_test_mode):
  import shutil, os, subprocess
  job_name = 'preprocess-hybrid-recommendation-features' + '-' + datetime.datetime.now().strftime('%y%m%d-%H%M%S')

  if in_test_mode:
      print('Launching local job ... hang on')
      OUTPUT_DIR = './preproc/features'
      shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
      os.makedirs(OUTPUT_DIR)
  else:
      print('Launching Dataflow job {} ... hang on'.format(job_name))
      OUTPUT_DIR = 'gs://{0}/hybrid_recommendation/preproc/features/'.format(BUCKET)
      try:
        subprocess.check_call('gsutil -m rm -r {}'.format(OUTPUT_DIR).split())
      except:
        pass

  options = {
      'staging_location': os.path.join(OUTPUT_DIR, 'tmp', 'staging'),
      'temp_location': os.path.join(OUTPUT_DIR, 'tmp'),
      'job_name': job_name,
      'project': PROJECT,
      'teardown_policy': 'TEARDOWN_ALWAYS',
      'no_save_main_session': True
  }
  opts = beam.pipeline.PipelineOptions(flags = [], **options)
  if in_test_mode:
    RUNNER = 'DirectRunner'
  else:
    RUNNER = 'DataflowRunner'
  p = beam.Pipeline(RUNNER, options = opts)
  
  query = query_hybrid_dataset

  if in_test_mode:
    query = query + ' LIMIT 100' 

  for step in ['train', 'eval']:
    if step == 'train':
      selquery = 'SELECT * FROM ({}) WHERE MOD(ABS(hash_id), 10) < 9'.format(query)
    else:
      selquery = 'SELECT * FROM ({}) WHERE MOD(ABS(hash_id), 10) = 9'.format(query)

    (p 
     | '{}_read'.format(step) >> beam.io.Read(beam.io.BigQuerySource(query = selquery, use_standard_sql = True))
     | '{}_csv'.format(step) >> beam.FlatMap(to_csv)
     | '{}_out'.format(step) >> beam.io.Write(beam.io.WriteToText(os.path.join(OUTPUT_DIR, '{}.csv'.format(step))))
    )

  job = p.run()
  if in_test_mode:
    job.wait_until_finish()
    print("Done!")
    
preprocess(in_test_mode = False)


# Let's check our files to make sure everything went as expected

# In[23]:


get_ipython().run_cell_magic('bash', '', 'rm -rf features\nmkdir features')


# In[24]:


get_ipython().system('gsutil -m cp -r gs://{BUCKET}/hybrid_recommendation/preproc/features/*.csv* features/')


# In[25]:


get_ipython().system('head -3 features/*')


# <h2> Create vocabularies using Dataflow </h2>
# 
# Let's use Cloud Dataflow to read in the BigQuery data, do some preprocessing, and write it out as CSV files.
# 
# Now we'll create our vocabulary files for our categorical features.

# In[7]:


query_vocabularies = """
SELECT
  CAST((SELECT MAX(IF(index = index_value, value, NULL)) FROM UNNEST(hits.customDimensions)) AS STRING) AS grouped_by
FROM `cloud-training-demos.GA360_test.ga_sessions_sample`,
  UNNEST(hits) AS hits
WHERE
  # only include hits on pages
  hits.type = "PAGE"
  AND (SELECT MAX(IF(index = index_value, value, NULL)) FROM UNNEST(hits.customDimensions)) IS NOT NULL
GROUP BY
  grouped_by
"""


# In[8]:


import apache_beam as beam
import datetime, os

def to_txt(rowdict):
  # Pull columns from BQ and create a line

  # Write out rows for each input row for grouped by column in rowdict
  return '{}'.format(rowdict['grouped_by'].encode('utf-8'))
  
def preprocess(in_test_mode):
  import shutil, os, subprocess
  job_name = 'preprocess-hybrid-recommendation-vocab-lists' + '-' + datetime.datetime.now().strftime('%y%m%d-%H%M%S')

  if in_test_mode:
      print('Launching local job ... hang on')
      OUTPUT_DIR = './preproc/vocabs'
      shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
      os.makedirs(OUTPUT_DIR)
  else:
      print('Launching Dataflow job {} ... hang on'.format(job_name))
      OUTPUT_DIR = 'gs://{0}/hybrid_recommendation/preproc/vocabs/'.format(BUCKET)
      try:
        subprocess.check_call('gsutil -m rm -r {}'.format(OUTPUT_DIR).split())
      except:
        pass

  options = {
      'staging_location': os.path.join(OUTPUT_DIR, 'tmp', 'staging'),
      'temp_location': os.path.join(OUTPUT_DIR, 'tmp'),
      'job_name': job_name,
      'project': PROJECT,
      'teardown_policy': 'TEARDOWN_ALWAYS',
      'no_save_main_session': True
  }
  opts = beam.pipeline.PipelineOptions(flags = [], **options)
  if in_test_mode:
      RUNNER = 'DirectRunner'
  else:
      RUNNER = 'DataflowRunner'
      
  p = beam.Pipeline(RUNNER, options = opts)
  
  def vocab_list(index, name):
    query = query_vocabularies.replace("index_value", "{}".format(index))

    (p 
     | '{}_read'.format(name) >> beam.io.Read(beam.io.BigQuerySource(query = query, use_standard_sql = True))
     | '{}_txt'.format(name) >> beam.Map(to_txt)
     | '{}_out'.format(name) >> beam.io.Write(beam.io.WriteToText(os.path.join(OUTPUT_DIR, '{0}_vocab.txt'.format(name))))
    )

  # Call vocab_list function for each
  vocab_list(10, 'content_id') # content_id
  vocab_list(7, 'category') # category
  vocab_list(2, 'author') # author
  
  job = p.run()
  if in_test_mode:
    job.wait_until_finish()
    print("Done!")
    
preprocess(in_test_mode = False)


# Also get vocab counts from the length of the vocabularies

# In[9]:


import apache_beam as beam
import datetime, os

def count_to_txt(rowdict):
  # Pull columns from BQ and create a line

  # Write out count
  return '{}'.format(rowdict['count_number'])
  
def mean_to_txt(rowdict):
  # Pull columns from BQ and create a line

  # Write out mean
  return '{}'.format(rowdict['mean_value'])
  
def preprocess(in_test_mode):
  import shutil, os, subprocess
  job_name = 'preprocess-hybrid-recommendation-vocab-counts' + '-' + datetime.datetime.now().strftime('%y%m%d-%H%M%S')

  if in_test_mode:
      print('Launching local job ... hang on')
      OUTPUT_DIR = './preproc/vocab_counts'
      shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
      os.makedirs(OUTPUT_DIR)
  else:
      print('Launching Dataflow job {} ... hang on'.format(job_name))
      OUTPUT_DIR = 'gs://{0}/hybrid_recommendation/preproc/vocab_counts/'.format(BUCKET)
      try:
        subprocess.check_call('gsutil -m rm -r {}'.format(OUTPUT_DIR).split())
      except:
        pass

  options = {
      'staging_location': os.path.join(OUTPUT_DIR, 'tmp', 'staging'),
      'temp_location': os.path.join(OUTPUT_DIR, 'tmp'),
      'job_name': job_name,
      'project': PROJECT,
      'teardown_policy': 'TEARDOWN_ALWAYS',
      'no_save_main_session': True
  }
  opts = beam.pipeline.PipelineOptions(flags = [], **options)
  if in_test_mode:
      RUNNER = 'DirectRunner'
  else:
      RUNNER = 'DataflowRunner'
      
  p = beam.Pipeline(RUNNER, options = opts)
  
  def vocab_count(index, column_name):
    query = """
SELECT
  COUNT(*) AS count_number
FROM ({})
""".format(query_vocabularies.replace("index_value", "{}".format(index)))

    (p 
     | '{}_read'.format(column_name) >> beam.io.Read(beam.io.BigQuerySource(query = query, use_standard_sql = True))
     | '{}_txt'.format(column_name) >> beam.Map(count_to_txt)
     | '{}_out'.format(column_name) >> beam.io.Write(beam.io.WriteToText(os.path.join(OUTPUT_DIR, '{0}_vocab_count.txt'.format(column_name))))
    )
    
  def global_column_mean(column_name):
    query = """
SELECT
  AVG(CAST({1} AS FLOAT64)) AS mean_value
FROM ({0})
""".format(query_hybrid_dataset, column_name)
    
    (p 
     | '{}_read'.format(column_name) >> beam.io.Read(beam.io.BigQuerySource(query = query, use_standard_sql = True))
     | '{}_txt'.format(column_name) >> beam.Map(mean_to_txt)
     | '{}_out'.format(column_name) >> beam.io.Write(beam.io.WriteToText(os.path.join(OUTPUT_DIR, '{0}_mean.txt'.format(column_name))))
    )
    
  # Call vocab_count function for each column we want the vocabulary count for
  vocab_count(10, 'content_id') # content_id
  vocab_count(7, 'category') # category
  vocab_count(2, 'author') # author
  
  # Call global_column_mean function for each column we want the mean for
  global_column_mean('months_since_epoch') # months_since_epoch
  
  job = p.run()
  if in_test_mode:
    job.wait_until_finish()
    print("Done!")
    
preprocess(in_test_mode = False)


# Let's check our files to make sure everything went as expected

# In[14]:


get_ipython().run_cell_magic('bash', '', 'rm -rf vocabs\nmkdir vocabs')


# In[15]:


get_ipython().system('gsutil -m cp -r gs://{BUCKET}/hybrid_recommendation/preproc/vocabs/*.txt* vocabs/')


# In[16]:


get_ipython().system('head -3 vocabs/*')


# In[20]:


get_ipython().run_cell_magic('bash', '', 'rm -rf vocab_counts\nmkdir vocab_counts')


# In[21]:


get_ipython().system('gsutil -m cp -r gs://{BUCKET}/hybrid_recommendation/preproc/vocab_counts/*.txt* vocab_counts/')


# In[22]:


get_ipython().system('head -3 vocab_counts/*')
