
# coding: utf-8

# ## Content-Based Filtering Using Neural Networks

# This lab relies on files created in the [content_based_preproc.ipynb](./content_based_preproc.ipynb) notebook. Be sure to complete the TODOs in that notebook and run the code there before completing this lab.  
# Also, we'll be using the **python3** kernel from here on out so don't forget to change the kernel if it's still python2.

# This lab illustrates:
# 1. how to build feature columns for a model using tf.feature_column
# 2. how to create custom evaluation metrics and add them to Tensorboard
# 3. how to train a model and make predictions with the saved model

# Tensorflow Hub should already be installed. You can check using pip freeze.

# In[1]:


get_ipython().run_cell_magic('bash', '', 'pip freeze | grep tensor')


# If 'tensorflow-hub' isn't one of the outputs above, then you'll need to install it. Uncomment the cell below and execute the commands. After doing the pip install, click **"Reset Session"** on the notebook so that the Python environment picks up the new packages.

# In[2]:


get_ipython().run_cell_magic('bash', '', 'pip install tensorflow-hub')


# In[1]:


import os
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import shutil

output = os.popen("gcloud config get-value project").readlines()
project_name = output[0][:-1]

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


# ### Build the feature columns for the model.

# To start, we'll load the list of categories, authors and article ids we created in the previous **Create Datasets** notebook.

# In[3]:


categories_list = open("categories.txt").read().splitlines()
authors_list = open("authors.txt").read().splitlines()
content_ids_list = open("content_ids.txt").read().splitlines()
mean_months_since_epoch = 523


# In the cell below we'll define the feature columns to use in our model. If necessary, remind yourself the [various feature columns](https://www.tensorflow.org/api_docs/python/tf/feature_column) to use.  
# For the embedded_title_column feature column, use a Tensorflow Hub Module to create an embedding of the article title. Since the articles and titles are in German, you'll want to use a German language embedding module.  
# Explore the text embedding Tensorflow Hub modules [available here](https://alpha.tfhub.dev/). Filter by setting the language to 'German'. The 50 dimensional embedding should be sufficient for our purposes. 

# In[7]:


#TODO (done): use a Tensorflow Hub module to create a text embeddding column for the article "title". 
# Use the module available at https://alpha.tfhub.dev/ filtering by German language.
embedded_title_column = hub.text_embedding_column(
    key = "title",
    module_spec = "https://tfhub.dev/google/nnlm-de-dim50/1",
    trainable = False
)

#TODO (done): create an embedded categorical feature column for the article id; i.e. "content_id".
content_id_column = tf.feature_column.categorical_column_with_hash_bucket(
    key = "content_id",
    hash_bucket_size = len(content_ids_list) + 1
)
embedded_content_column = tf.feature_column.embedding_column(
    categorical_column = content_id_column,
    dimension = 10
)

#TODO (done): create an embedded categorical feature column for the article "author"
author_column = tf.feature_column.categorical_column_with_hash_bucket(
    key = "author",
    hash_bucket_size = len(authors_list) + 1
)
embedded_author_column = tf.feature_column.embedding_column(
    categorical_column = author_column,
    dimension = 3
)

#TODO (done): create a categorical feature column for the article "category"
category_column_categorical = tf.feature_column.categorical_column_with_vocabulary_list(
    key = "category",
    vocabulary_list = categories_list,
    num_oov_buckets = 1
)
category_column = tf.feature_column.indicator_column(category_column_categorical)
## note: indicator_column creates a multi-hot-encoded column.

#TODO (done): create a bucketized numeric feature column of values for the "months since epoch"
months_since_epoch_boundaries = list(range(400,700,20))
months_since_epochs_numeric = tf.feature_column.numeric_column(key = "months_since_epoch")
months_since_epoch_bucketized = tf.feature_column.bucketized_column(
  source_column = months_since_epochs_numeric,
  boundaries = months_since_epoch_boundaries
)

#TODO (done): create a crossed feature column using the "category" and "months since epoch" values
crossed_months_since_category_column = tf.feature_column.indicator_column(
  tf.feature_column.crossed_column(
    keys = [category_column_categorical, months_since_epoch_bucketized], 
    hash_bucket_size = len(months_since_epoch_boundaries * (len(categories_list) + 1))
  )
)

feature_columns = [embedded_content_column,
                   embedded_author_column,
                   category_column,
                   embedded_title_column,
                   crossed_months_since_category_column] 


# ### Create the input function.
# 
# Next we'll create the input function for our model. This input function reads the data from the csv files we created in the previous labs. 

# In[8]:


record_defaults = [["Unknown"], ["Unknown"],["Unknown"],["Unknown"],["Unknown"],[mean_months_since_epoch],["Unknown"]]
column_keys = ["visitor_id", "content_id", "category", "title", "author", "months_since_epoch", "next_content_id"]
label_key = "next_content_id"
def read_dataset(filename, mode, batch_size = 512):
  def _input_fn():
      def decode_csv(value_column):
          columns = tf.decode_csv(value_column,record_defaults=record_defaults)
          features = dict(zip(column_keys, columns))          
          label = features.pop(label_key)         
          return features, label

      # Create list of files that match pattern
      file_list = tf.gfile.Glob(filename)

      # Create dataset from file list
      dataset = tf.data.TextLineDataset(file_list).map(decode_csv)

      if mode == tf.estimator.ModeKeys.TRAIN:
          num_epochs = None # indefinitely
          dataset = dataset.shuffle(buffer_size = 10 * batch_size)
      else:
          num_epochs = 1 # end-of-input after this

      dataset = dataset.repeat(num_epochs).batch(batch_size)
      return dataset.make_one_shot_iterator().get_next()
  return _input_fn


# ### Create the model and train/evaluate
# 
# 
# Next, we'll build our model which recommends an article for a visitor to the Kurier.at website. Look through the code below. We use the input_layer feature column to create the dense input layer to our network. This is just a sigle layer network where we can adjust the number of hidden units as a parameter.
# 
# Currently, we compute the accuracy between our predicted 'next article' and the actual 'next article' read next by the visitor. Resolve the TODOs in the cell below by adding additional performance metrics to assess our model. You will need to 
# * use the [tf.metrics library](https://www.tensorflow.org/api_docs/python/tf/metrics) to compute an additional performance metric
# * add this additional metric to the metrics dictionary, and 
# * include it in the tf.summary that is sent to Tensorboard.

# In[13]:


def model_fn(features, labels, mode, params):
  net = tf.feature_column.input_layer(features, params['feature_columns'])
  for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
   # Compute logits (1 per class).
  logits = tf.layers.dense(net, params['n_classes'], activation=None) 

  predicted_classes = tf.argmax(logits, 1)
  from tensorflow.python.lib.io import file_io
    
  with file_io.FileIO('content_ids.txt', mode='r') as ifp:
    content = tf.constant([x.rstrip() for x in ifp])
  predicted_class_names = tf.gather(content, predicted_classes)
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'class_ids': predicted_classes[:, tf.newaxis],
        'class_names' : predicted_class_names[:, tf.newaxis],
        'probabilities': tf.nn.softmax(logits),
        'logits': logits,
    }
    return tf.estimator.EstimatorSpec(mode, predictions=predictions)
  table = tf.contrib.lookup.index_table_from_file(vocabulary_file="content_ids.txt")
  labels = table.lookup(labels)
  # Compute loss.
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Compute evaluation metrics.
  accuracy = tf.metrics.accuracy(labels=labels,
                                 predictions=predicted_classes,
                                 name='acc_op')
  #TODO (done): Compute the top_10 accuracy, using the tf.nn.in_top_k and tf.metrics.mean functions in Tensorflow
  top_10_accuracy = tf.metrics.mean(
    tf.nn.in_top_k(predictions = logits, targets = labels, k = 10)
  )
  
  metrics = {
    'accuracy': accuracy,
    #TODO (done): Add top_10_accuracy to the metrics dictionary
    'top_10_accuracy': top_10_accuracy
  }
  
  ## note: second element [1] is the `update_op`, which is the updated metric after each batch...
  tf.summary.scalar('accuracy', accuracy[1])
  #TODO (done): Add the top_10_accuracy metric to the Tensorboard summary
  tf.summary.scalar('top_10_accuracy', top_10_accuracy[1])

  if mode == tf.estimator.ModeKeys.EVAL:
      return tf.estimator.EstimatorSpec(
          mode, loss=loss, eval_metric_ops=metrics)

  # Create training op.
  assert mode == tf.estimator.ModeKeys.TRAIN

  optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
  train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
  return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


# ### Train and Evaluate

# In[14]:


outdir = 'content_based_model_trained'
shutil.rmtree(outdir, ignore_errors = True) # start fresh each time
tf.summary.FileWriterCache.clear() # ensure filewriter cache is clear for TensorBoard events file
estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir = outdir,
    params={
     'feature_columns': feature_columns,
      'hidden_units': [200, 100, 50],
      'n_classes': len(content_ids_list)
    })

train_spec = tf.estimator.TrainSpec(
    input_fn = read_dataset("training_set.csv", tf.estimator.ModeKeys.TRAIN),
    max_steps = 200)

eval_spec = tf.estimator.EvalSpec(
    input_fn = read_dataset("test_set.csv", tf.estimator.ModeKeys.EVAL),
    steps = None,
    start_delay_secs = 30,
    throttle_secs = 60)

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


# ### Make predictions with the trained model. 
# 
# With the model now trained, we can make predictions by calling the predict method on the estimator. Let's look at how our model predicts on the first five examples of the training set.  
# To start, we'll create a new file 'first_5.csv' which contains the first five elements of our training set. We'll also save the target values to a file 'first_5_content_ids' so we can compare our results. 

# In[15]:


get_ipython().run_cell_magic('bash', '', 'head -5 training_set.csv > first_5.csv\nhead first_5.csv\nawk -F "\\"*,\\"*" \'{print $2}\' first_5.csv > first_5_content_ids')


# Recall, to make predictions on the trained model we pass a list of examples through the input function. Complete the code below to make predicitons on the examples contained in the "first_5.csv" file we created above. 

# In[16]:


#TODO: Use the predict method on our trained model to find the predictions for the examples contained in "first_5.csv".
output = list(
  estimator.predict(
    input_fn = read_dataset(filename = "first_5.csv", mode = tf.estimator.ModeKeys.PREDICT)
  )
)
print(output)


# In[17]:


import numpy as np
recommended_content_ids = [np.asscalar(d["class_names"]).decode('UTF-8') for d in output]
content_ids = open("first_5_content_ids").read().splitlines()


# Finally, we'll map the content id back to the article title. We can then compare our model's recommendation for the first of our examples. This can all be done in BigQuery. Look through the query below and make sure it is clear what is being returned.

# In[18]:


import google.datalab.bigquery as bq
recommended_title_sql="""
#standardSQL
SELECT
(SELECT MAX(IF(index=6, value, NULL)) FROM UNNEST(hits.customDimensions)) AS title
FROM `cloud-training-demos.GA360_test.ga_sessions_sample`,   
  UNNEST(hits) AS hits
WHERE 
  # only include hits on pages
  hits.type = "PAGE"
  AND (SELECT MAX(IF(index=10, value, NULL)) FROM UNNEST(hits.customDimensions)) = \"{}\"
LIMIT 1""".format(recommended_content_ids[0])

current_title_sql="""
#standardSQL
SELECT
(SELECT MAX(IF(index=6, value, NULL)) FROM UNNEST(hits.customDimensions)) AS title
FROM `cloud-training-demos.GA360_test.ga_sessions_sample`,   
  UNNEST(hits) AS hits
WHERE 
  # only include hits on pages
  hits.type = "PAGE"
  AND (SELECT MAX(IF(index=10, value, NULL)) FROM UNNEST(hits.customDimensions)) = \"{}\"
LIMIT 1""".format(content_ids[0])
recommended_title = bq.Query(recommended_title_sql).execute().result().to_dataframe()['title'].tolist()[0]
current_title = bq.Query(current_title_sql).execute().result().to_dataframe()['title'].tolist()[0]
print("Current title: {} ".format(current_title))
print("Recommended title: {}".format(recommended_title))


# ### Tensorboard
# 
# As usual, we can monitor the performance of our training job using Tensorboard. 

# In[20]:


from google.datalab.ml import TensorBoard
TensorBoard().start('content_based_model_trained')


# In[21]:


for pid in TensorBoard.list()['pid']:
  TensorBoard().stop(pid)
  print("Stopped TensorBoard with pid {}".format(pid))
