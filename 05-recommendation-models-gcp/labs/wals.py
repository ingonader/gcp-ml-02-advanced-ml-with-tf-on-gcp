
# coding: utf-8

# # Collaborative filtering on Google Analytics data
# 
# This notebook demonstrates how to implement a WALS matrix refactorization approach to do collaborative filtering.

# In[1]:


import os
output = os.popen("gcloud config get-value project").readlines()
project_name = output[0][:-1]

# change these to try this notebook out
PROJECT = project_name
BUCKET = project_name
#BUCKET = BUCKET.replace("qwiklabs-gcp-", "inna-bckt-")
REGION = 'europe-west1'  ## note: Cloud ML Engine not availabe in europe-west3!

# do not change these
os.environ['PROJECT'] = PROJECT
os.environ['BUCKET'] = BUCKET
os.environ['REGION'] = REGION
os.environ['TFVERSION'] = '1.8'

print(PROJECT)
print(BUCKET)
print(REGION)


# In[2]:


get_ipython().run_cell_magic('bash', '', 'gcloud config set project $PROJECT\ngcloud config set compute/region $REGION')


# In[3]:


import tensorflow as tf
print(tf.__version__)


# ## Create raw dataset
# <p>
# For collaborative filtering, we don't need to know anything about either the users or the content. Essentially, all we need to know is userId, itemId, and rating that the particular user gave the particular item.
# <p>
# In this case, we are working with newspaper articles. The company doesn't ask their users to rate the articles. However, we can use the time-spent on the page as a proxy for rating.
# <p>
# Normally, we would also add a time filter to this ("latest 7 days"), but our dataset is itself limited to a few days.

# In[4]:


import google.datalab.bigquery as bq

sql="""
#standardSQL
WITH visitor_page_content AS (

   SELECT  
     fullVisitorID,
     (SELECT MAX(IF(index=10, value, NULL)) FROM UNNEST(hits.customDimensions)) AS latestContentId,  
     (LEAD(hits.time, 1) OVER (PARTITION BY fullVisitorId ORDER BY hits.time ASC) - hits.time) AS session_duration 
   FROM `cloud-training-demos.GA360_test.ga_sessions_sample`,   
     UNNEST(hits) AS hits
   WHERE 
     # only include hits on pages
      hits.type = "PAGE"

   GROUP BY   
     fullVisitorId, latestContentId, hits.time
     )

# aggregate web stats
SELECT   
  fullVisitorID as visitorId,
  latestContentId as contentId,
  SUM(session_duration) AS session_duration 
 
FROM visitor_page_content
  WHERE latestContentId IS NOT NULL 
  GROUP BY fullVisitorID, latestContentId
  HAVING session_duration > 0
  ORDER BY latestContentId 
"""

df = bq.Query(sql).execute().result().to_dataframe()
df.head()


# In[7]:


stats = df.describe()
stats.round(2)


# In[9]:


# the rating is the session_duration scaled to be in the range 0-1.  This will help with training.
df['rating'] = 0.3 * (1 + (df['session_duration'] - stats.loc['50%', 'session_duration'])/stats.loc['50%', 'session_duration'])
df.loc[df['rating'] > 1, 'rating'] = 1
df.describe().round(3)


# In[10]:


del df['session_duration']


# In[11]:


get_ipython().run_cell_magic('bash', '', 'rm -rf data\nmkdir data')


# In[12]:


df.to_csv('data/collab_raw.csv', index=False, header=False)


# In[13]:


get_ipython().system('head data/collab_raw.csv')


# ## Create dataset for WALS
# <p>
# The raw dataset (above) won't work for WALS:
# <ol>
# <li> The userId and itemId have to be 0,1,2 ... so we need to create a mapping from visitorId (in the raw data) to userId and contentId (in the raw data) to itemId.
# <li> We will need to save the above mapping to a file because at prediction time, we'll need to know how to map the contentId in the table above to the itemId.
# <li> We'll need two files: a "rows" dataset where all the items for a particular user are listed; and a "columns" dataset where all the users for a particular item are listed.
# </ol>
# 
# <p>
# 
# ### Mapping

# In[14]:


import pandas as pd
import numpy as np
def create_mapping(values, filename):
  with open(filename, 'w') as ofp:
    value_to_id = {value:idx for idx, value in enumerate(values.unique())}
    for value, idx in value_to_id.items():
      ofp.write('{},{}\n'.format(value, idx))
  return value_to_id

df = pd.read_csv('data/collab_raw.csv',
                 header=None,
                 names=['visitorId', 'contentId', 'rating'],
                dtype={'visitorId': str, 'contentId': str, 'rating': np.float})
df.to_csv('data/collab_raw.csv', index=False, header=False)
user_mapping = create_mapping(df['visitorId'], 'data/users.csv')
item_mapping = create_mapping(df['contentId'], 'data/items.csv')


# In[15]:


get_ipython().system('head -3 data/*.csv')


# In[16]:


df['userId'] = df['visitorId'].map(user_mapping.get)
df['itemId'] = df['contentId'].map(item_mapping.get)


# In[17]:


mapped_df = df[['userId', 'itemId', 'rating']]
mapped_df.to_csv('data/collab_mapped.csv', index=False, header=False)
mapped_df.head()


# ### Creating rows and columns datasets

# In[18]:


import pandas as pd
import numpy as np
mapped_df = pd.read_csv('data/collab_mapped.csv', header=None, names=['userId', 'itemId', 'rating'])
mapped_df.head()


# In[21]:


NITEMS = np.max(mapped_df['itemId']) + 1
NUSERS = np.max(mapped_df['userId']) + 1
mapped_df['rating'] = np.round(mapped_df['rating'].values, 2)
print('{} items, {} users, {} interactions'.format( NITEMS, NUSERS, len(mapped_df) ))
mapped_df.head()


# In[22]:


grouped_by_items = mapped_df.groupby('itemId')
iter = 0
for item, grouped in grouped_by_items:
  print(item, grouped['userId'].values, grouped['rating'].values)
  iter = iter + 1
  if iter > 5:
    break


# In[23]:


import tensorflow as tf
grouped_by_items = mapped_df.groupby('itemId')
with tf.python_io.TFRecordWriter('data/users_for_item') as ofp:
  for item, grouped in grouped_by_items:
    example = tf.train.Example(features=tf.train.Features(feature={
          'key': tf.train.Feature(int64_list=tf.train.Int64List(value=[item])),
          'indices': tf.train.Feature(int64_list=tf.train.Int64List(value=grouped['userId'].values)),
          'values': tf.train.Feature(float_list=tf.train.FloatList(value=grouped['rating'].values))
        }))
    ofp.write(example.SerializeToString())      


# In[24]:


grouped_by_users = mapped_df.groupby('userId')
with tf.python_io.TFRecordWriter('data/items_for_user') as ofp:
  for user, grouped in grouped_by_users:
    example = tf.train.Example(features=tf.train.Features(feature={
          'key': tf.train.Feature(int64_list=tf.train.Int64List(value=[user])),
          'indices': tf.train.Feature(int64_list=tf.train.Int64List(value=grouped['itemId'].values)),
          'values': tf.train.Feature(float_list=tf.train.FloatList(value=grouped['rating'].values))
        }))
    ofp.write(example.SerializeToString())      


# In[25]:


grouped_by_users = mapped_df.groupby('userId')
N = 0
with tf.python_io.TFRecordWriter('data/items_for_user_subset') as ofp:
  for user, grouped in grouped_by_users:
    example = tf.train.Example(features=tf.train.Features(feature={
          'key': tf.train.Feature(int64_list=tf.train.Int64List(value=[user])),
          'indices': tf.train.Feature(int64_list=tf.train.Int64List(value=grouped['itemId'].values)),
          'values': tf.train.Feature(float_list=tf.train.FloatList(value=grouped['rating'].values))
        }))
    ofp.write(example.SerializeToString())    
    N = N + 1
    if N > 20:
      break


# In[26]:


get_ipython().system('ls -lrt data')


# To summarize, we created the following data files from collab_raw.csv:
# 
# * `collab_mapped.csv` is essentially the same data as in `collab_raw.csv` except that `visitorId` and `contentId` which are business-specific have been mapped to `userId` and `itemId` which are enumerated in 0,1,2,....  The mappings themselves are stored in `items.csv` and `users.csv` so that they can be used during inference.
# * `users_for_item` contains all the users/ratings for each item in TFExample format
# * `items_for_user` contains all the items/ratings for each user in TFExample format
# 

# ## Train with WALS
# 
# Once you have the dataset, do matrix factorization with WALS using the [WALSMatrixFactorization](https://www.tensorflow.org/versions/master/api_docs/python/tf/contrib/factorization/WALSMatrixFactorization) in the contrib directory.
# This is an estimator model, so it should be relatively familiar.
# <p>
# As usual, we write an input_fn to provide the data to the model, and then create the Estimator to do train_and_evaluate.
# Because it is in contrib and hasn't moved over to tf.estimator yet, we use tf.contrib.learn.Experiment to handle the training loop.

# In[27]:


import os
import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow.contrib.factorization import WALSMatrixFactorization
  
def read_dataset(mode, args):
  def decode_example(protos, vocab_size):
    #TODO
    features = {'key': tf.FixedLenFeature([1], tf.int64),   ## one user (or one item)
                'indices': tf.VarLenFeature(dtype = tf.int64),  ## multiple items (or users) that the user (item) has interacted with
                'values': tf.VarLenFeature(dtype = tf.float32)  ## ratings of these items (or users) in the above interactions
               }
    ## read tf-record file examples in the given format above:
    parsed_features = tf.parse_single_example(protos, features)
    ## convert into sparse tensor as needed for WALS:
    values = tf.sparse_merge(
      parsed_features['indices'], 
      parsed_features['values'],
      vocab_size = vocab_size
    )
    # Save key to remap after batching
    # This is a temporary workaround to assign correct row numbers in each batch.
    # You can ignore details of this part and remap_keys().
    key = parsed_features['key']
    decoded_sparse_tensor = tf.SparseTensor(
      indices = tf.concat([values.indices, [key]], axis = 0),
      values = tf.concat([values.values, [0.0]], axis = 0),
      dense_shape = values.dense_shape
    )
    return decoded_sparse_tensor
  
  
  def remap_keys(sparse_tensor):
    # Current indices of our SparseTensor that we need to fix
    bad_indices = sparse_tensor.indices
    # Current values of our SparseTensor that we need to fix
    bad_values = sparse_tensor.values 
  
    # Group by the batch_indices and get the count for each  
    size = tf.segment_sum(data = tf.ones_like(bad_indices[:,0], dtype = tf.int64), segment_ids = bad_indices[:,0]) - 1
    # The number of batch_indices (this should be batch_size unless it is a partially full batch)
    length = tf.shape(size, out_type = tf.int64)[0]
    # Finds the cumulative sum which we can use for indexing later
    cum = tf.cumsum(size)
    # The offsets between each example in the batch due to our concatentation of the keys in the decode_example method
    length_range = tf.range(start = 0, limit = length, delta = 1, dtype = tf.int64)
    # Indices of the SparseTensor's indices member of the rows we added by the concatentation of our keys in the decode_example method
    cum_range = cum + length_range

    # The keys that we have extracted back out of our concatentated SparseTensor
    gathered_indices = tf.squeeze(tf.gather(bad_indices, cum_range)[:,1])

    # The enumerated row indices of the SparseTensor's indices member
    sparse_indices_range = tf.range(tf.shape(bad_indices, out_type = tf.int64)[0], dtype = tf.int64)

    # We want to find here the row indices of the SparseTensor's indices member that are of our actual data and not the concatentated rows
    # So we want to find the intersection of the two sets and then take the opposite of that
    x = sparse_indices_range
    s = cum_range

    # Number of multiples we are going to tile x, which is our sparse_indices_range
    tile_multiples = tf.concat([tf.ones(tf.shape(tf.shape(x)), dtype=tf.int64), tf.shape(s, out_type = tf.int64)], axis = 0)
    # Expands x, our sparse_indices_range, into a rank 2 tensor and then multiplies the rows by 1 (no copying) and the columns by the number of examples in the batch
    x_tile = tf.tile(tf.expand_dims(x, -1), tile_multiples)
    # Essentially a vectorized logical or, that we then negate
    x_not_in_s = ~tf.reduce_any(tf.equal(x_tile, s), -1)

    # The SparseTensor's indices that are our actual data by using the boolean_mask we just made above applied to the entire indices member of our SparseTensor
    selected_indices = tf.boolean_mask(tensor = bad_indices, mask = x_not_in_s, axis = 0)
    # Apply the same boolean_mask to the entire values member of our SparseTensor to get the actual values data
    selected_values = tf.boolean_mask(tensor = bad_values, mask = x_not_in_s, axis = 0)

    # Need to replace the first column of our selected_indices with keys, so we first need to tile our gathered_indices
    tiling = tf.tile(input = tf.expand_dims(gathered_indices[0], -1), multiples = tf.expand_dims(size[0] , -1))
    
    # We have to repeatedly apply the tiling to each example in the batch
    # Since it is jagged we cannot use tf.map_fn due to the stacking of the TensorArray, so we have to create our own custom version
    def loop_body(i, tensor_grow):
      return i + 1, tf.concat(values = [tensor_grow, tf.tile(input = tf.expand_dims(gathered_indices[i], -1), multiples = tf.expand_dims(size[i] , -1))], axis = 0)

    _, result = tf.while_loop(lambda i, tensor_grow: i < length, loop_body, [tf.constant(1, dtype = tf.int64), tiling])
    
    # Concatenate tiled keys with the 2nd column of selected_indices
    selected_indices_fixed = tf.concat([tf.expand_dims(result, -1), tf.expand_dims(selected_indices[:, 1], -1)], axis = 1)
    
    # Combine everything together back into a SparseTensor
    remapped_sparse_tensor = tf.SparseTensor(indices = selected_indices_fixed, values = selected_values, dense_shape = sparse_tensor.dense_shape)
    return remapped_sparse_tensor

    
  def parse_tfrecords(filename, vocab_size):
    if mode == tf.estimator.ModeKeys.TRAIN:
        num_epochs = None # indefinitely
    else:
        num_epochs = 1 # end-of-input after this
    
    files = tf.gfile.Glob(os.path.join(args['input_path'], filename))
    
    # Create dataset from file list
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(lambda x: decode_example(x, vocab_size))
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(args['batch_size'])
    dataset = dataset.map(lambda x: remap_keys(x))
    return dataset.make_one_shot_iterator().get_next()
  
  def _input_fn():
    features = {
      WALSMatrixFactorization.INPUT_ROWS: parse_tfrecords('items_for_user', args['nitems']),
      WALSMatrixFactorization.INPUT_COLS: parse_tfrecords('users_for_item', args['nusers']),
      WALSMatrixFactorization.PROJECT_ROW: tf.constant(True)
    }
    return features, None
  
  # just for developing line by line. You don't need this in production
  def _input_fn_subset():
    features = {
      WALSMatrixFactorization.INPUT_ROWS: parse_tfrecords('items_for_user_subset', args['nitems']),
      WALSMatrixFactorization.PROJECT_ROW: tf.constant(True)
    }
    return features, None
  
  def input_cols():
    return parse_tfrecords('users_for_item', args['nusers'])
  
  return _input_fn#_subset


# This code is helpful in developing the input function. You don't need it in production.

# In[28]:


def try_out():
  with tf.Session() as sess:
    fn = read_dataset(tf.estimator.ModeKeys.EVAL, 
                    {'input_path': 'data', 'batch_size': 8, 'nitems': 5668, 'nusers': 82802})
    feats, _ = fn()
    print(feats['input_rows'].eval())

try_out()


# In[29]:


def find_top_k(user, item_factors, k):
  all_items = tf.matmul(tf.expand_dims(user, 0), tf.transpose(item_factors))
  topk = tf.nn.top_k(all_items, k=k)
  return tf.cast(topk.indices, dtype=tf.int64)
    
def batch_predict(args):
  import numpy as np
  with tf.Session() as sess:
    #TODO
    estimator = tf.contrib.factorization.WALSMatrixFactorization(
      num_rows = args['nusers'],
      num_cols = args['nitems'],
      embedding_dimension = args['n_embeds'],
      model_dir = args['output_dir']
    )
    # this is how you would get the row factors for out-of-vocab user data
    #row_factors = list(estimator.get_projections(input_fn=read_dataset(tf.estimator.ModeKeys.EVAL, args)))
    #user_factors = tf.convert_to_tensor(np.array(row_factors))
    
    # but for in-vocab data, the row factors are already in the checkpoint
    user_factors = tf.convert_to_tensor(estimator.get_row_factors()[0]) # (nusers, nembeds)
    # in either case, we have to assume catalog doesn't change, so col_factors are read in
    item_factors = tf.convert_to_tensor(estimator.get_col_factors()[0])# (nitems, nembeds)
    
    # for each user, find the top K items
    topk = tf.squeeze(tf.map_fn(lambda user: find_top_k(user, item_factors, args['topk']), user_factors, dtype=tf.int64))
    with file_io.FileIO(os.path.join(args['output_dir'], 'batch_pred.txt'), mode='w') as f:
      for best_items_for_user in topk.eval():
        f.write(','.join(str(x) for x in best_items_for_user) + '\n')

# online prediction returns row and column factors as needed
def create_serving_input_fn(args):
  def for_user_embeddings(userId):
      # all items for this user (for user_embeddings)
      items = tf.range(args['nitems'], dtype=tf.int64)
      users = userId * tf.ones([args['nitems']], dtype=tf.int64)
      ratings = 0.1 * tf.ones_like(users, dtype=tf.float32)
      return items, users, ratings, tf.constant(True)
    
  def for_item_embeddings(itemId):
      # all users for this item (for item_embeddings)
      users = tf.range(args['nusers'], dtype=tf.int64)
      items = itemId * tf.ones([args['nusers']], dtype=tf.int64)
      ratings = 0.1 * tf.ones_like(users, dtype=tf.float32)
      return items, users, ratings, tf.constant(False)
    
  def serving_input_fn():
    feature_ph = {
        'userId': tf.placeholder(tf.int64, 1),
        'itemId': tf.placeholder(tf.int64, 1)
    }

    (items, users, ratings, project_row) =                   tf.cond(feature_ph['userId'][0] < tf.constant(0, dtype=tf.int64),
                          lambda: for_item_embeddings(feature_ph['itemId']),
                          lambda: for_user_embeddings(feature_ph['userId']))
    rows = tf.stack( [users, items], axis=1 )
    cols = tf.stack( [items, users], axis=1 )
    input_rows = tf.SparseTensor(rows, ratings, (args['nusers'], args['nitems']))
    input_cols = tf.SparseTensor(cols, ratings, (args['nusers'], args['nitems']))
    
    features = {
      WALSMatrixFactorization.INPUT_ROWS: input_rows,
      WALSMatrixFactorization.INPUT_COLS: input_cols,
      WALSMatrixFactorization.PROJECT_ROW: project_row
    }
    return tf.contrib.learn.InputFnOps(features, None, feature_ph)
  return serving_input_fn
        
def train_and_evaluate(args):
    train_steps = int(0.5 + (1.0 * args['num_epochs'] * args['nusers']) / args['batch_size'])
    steps_in_epoch = int(0.5 + args['nusers'] / args['batch_size'])
    print('Will train for {} steps, evaluating once every {} steps'.format(train_steps, steps_in_epoch))
    def experiment_fn(output_dir):
        return tf.contrib.learn.Experiment(
            tf.contrib.factorization.WALSMatrixFactorization(
                         num_rows=args['nusers'], num_cols=args['nitems'],
                         embedding_dimension=args['n_embeds'],
                         model_dir=args['output_dir']),
            train_input_fn=read_dataset(tf.estimator.ModeKeys.TRAIN, args),
            eval_input_fn=read_dataset(tf.estimator.ModeKeys.EVAL, args),
            train_steps=train_steps,
            eval_steps=1,
            min_eval_frequency=steps_in_epoch,
            export_strategies=tf.contrib.learn.utils.saved_model_export_utils.make_export_strategy(serving_input_fn=create_serving_input_fn(args))
        )

    from tensorflow.contrib.learn.python.learn import learn_runner
    learn_runner.run(experiment_fn, args['output_dir'])
    
    batch_predict(args)


# In[30]:


import shutil
shutil.rmtree('wals_trained', ignore_errors=True)
train_and_evaluate({
    'output_dir': 'wals_trained',
    'input_path': 'data/',
    'num_epochs': 0.05,
    'nitems': 5668,
    'nusers': 82802,

    'batch_size': 512,
    'n_embeds': 10,
    'topk': 3
  })


# In[31]:


get_ipython().system('ls wals_trained')


# In[32]:


get_ipython().system('head wals_trained/batch_pred.txt')


# ## Run as a Python module
# 
# Let's run it as Python module for just a few steps.

# In[33]:


get_ipython().run_cell_magic('bash', '', 'rm -rf wals.tar.gz wals_trained\ngcloud ml-engine local train \\\n   --module-name=walsmodel.task \\\n   --package-path=${PWD}/walsmodel \\\n   -- \\\n   --output_dir=${PWD}/wals_trained \\\n   --input_path=${PWD}/data \\\n   --num_epochs=0.01 --nitems=5668 --nusers=82802 \\\n   --job-dir=./tmp')


# ## Get row and column factors
# 
# Once you have a trained WALS model, you can get row and column factors (user and item embeddings) using the serving input function that we exported.  We'll look at how to use these in the section on building a recommendation system using deep neural networks.

# In[34]:


get_ipython().run_line_magic('writefile', 'data/input.json')
{"userId": 4, "itemId": -1}


# In[35]:


get_ipython().run_cell_magic('bash', '', 'MODEL_DIR=$(ls wals_trained/export/Servo | tail -1)\ngcloud ml-engine local predict --model-dir=wals_trained/export/Servo/$MODEL_DIR --json-instances=data/input.json')


# In[36]:


get_ipython().run_line_magic('writefile', 'data/input.json')
{"userId": -1, "itemId": 4}


# In[37]:


get_ipython().run_cell_magic('bash', '', 'MODEL_DIR=$(ls wals_trained/export/Servo | tail -1)\ngcloud ml-engine local predict --model-dir=wals_trained/export/Servo/$MODEL_DIR --json-instances=data/input.json')


# ## Run on Cloud

# In[38]:


get_ipython().run_cell_magic('bash', '', 'gsutil -m cp data/* gs://${BUCKET}/wals/data')


# In[39]:


get_ipython().run_cell_magic('bash', '', 'OUTDIR=gs://${BUCKET}/wals/model_trained\nJOBNAME=wals_$(date -u +%y%m%d_%H%M%S)\necho $OUTDIR $REGION $JOBNAME\ngsutil -m rm -rf $OUTDIR\ngcloud ml-engine jobs submit training $JOBNAME \\\n   --region=$REGION \\\n   --module-name=walsmodel.task \\\n   --package-path=${PWD}/walsmodel \\\n   --job-dir=$OUTDIR \\\n   --staging-bucket=gs://$BUCKET \\\n   --scale-tier=BASIC_GPU \\\n   --runtime-version=$TFVERSION \\\n   -- \\\n   --output_dir=$OUTDIR \\\n   --input_path=gs://${BUCKET}/wals/data \\\n   --num_epochs=10 --nitems=5668 --nusers=82802 ')


# This took <b>10 minutes</b> for me.

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