
# coding: utf-8

# # Neural network hybrid recommendation system on Google Analytics data model and training
# 
# This notebook demonstrates how to implement a hybrid recommendation system using a neural network to combine content-based and collaborative filtering recommendation models using Google Analytics data. We are going to use the learned user embeddings from [wals.ipynb](../wals.ipynb) and combine that with our previous content-based features from [content_based_using_neural_networks.ipynb](../content_based_using_neural_networks.ipynb)
# 
# Now that we have our data preprocessed from BigQuery and Cloud Dataflow, we can build our neural network hybrid recommendation model to our preprocessed data. Then we can train locally to make sure everything works and then use the power of Google Cloud ML Engine to scale it out.

# We're going to use TensorFlow Hub to use trained text embeddings, so let's first pip install that and reset our session.

# In[1]:


get_ipython().system('pip install tensorflow_hub')


# Now reset the notebook's session kernel! Since we're no longer using Cloud Dataflow, we'll be using the python3 kernel from here on out so don't forget to change the kernel if it's still python2.

# In[1]:


# Import helpful libraries and setup our project, bucket, and region
import os
import tensorflow as tf
import tensorflow_hub as hub

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


# In[3]:


get_ipython().run_cell_magic('bash', '', "if ! gsutil ls | grep -q gs://${BUCKET}/hybrid_recommendation/preproc; then\n  gsutil mb -l ${REGION} gs://${BUCKET}\n  # copy canonical set of preprocessed files if you didn't do preprocessing notebook\n  gsutil -m cp -R gs://cloud-training-demos/courses/machine_learning/deepdive/10_recommendation/hybrid_recommendation gs://${BUCKET}\nfi")


# In[12]:


get_ipython().run_cell_magic('bash', '', 'gsutil cat -r 0-1024 gs://cloud-training-demos/courses/machine_learning/deepdive/10_recommendation/hybrid_recommendation/preproc/features/train.csv-00000-of-00004')


# Let's first get some of our aggregate information that we will use in the model from some of our preprocessed files we saved in Google Cloud Storage.

# In[4]:


from tensorflow.python.lib.io import file_io


# In[5]:


# Get number of content ids from text file in Google Cloud Storage
with file_io.FileIO(tf.gfile.Glob(filename = "gs://{}/hybrid_recommendation/preproc/vocab_counts/content_id_vocab_count.txt*".format(BUCKET))[0], mode = 'r') as ifp:
  number_of_content_ids = int([x for x in ifp][0])
print("number_of_content_ids = {}".format(number_of_content_ids))


# In[6]:


# Get number of categories from text file in Google Cloud Storage
with file_io.FileIO(tf.gfile.Glob(filename = "gs://{}/hybrid_recommendation/preproc/vocab_counts/category_vocab_count.txt*".format(BUCKET))[0], mode = 'r') as ifp:
  number_of_categories = int([x for x in ifp][0])
print("number_of_categories = {}".format(number_of_categories))


# In[7]:


# Get number of authors from text file in Google Cloud Storage
with file_io.FileIO(tf.gfile.Glob(filename = "gs://{}/hybrid_recommendation/preproc/vocab_counts/author_vocab_count.txt*".format(BUCKET))[0], mode = 'r') as ifp:
  number_of_authors = int([x for x in ifp][0])
print("number_of_authors = {}".format(number_of_authors))


# In[8]:


# Get mean months since epoch from text file in Google Cloud Storage
with file_io.FileIO(tf.gfile.Glob(filename = "gs://{}/hybrid_recommendation/preproc/vocab_counts/months_since_epoch_mean.txt*".format(BUCKET))[0], mode = 'r') as ifp:
  mean_months_since_epoch = float([x for x in ifp][0])
print("mean_months_since_epoch = {}".format(mean_months_since_epoch))


# In[9]:


# Determine CSV and label columns
NON_FACTOR_COLUMNS = 'next_content_id,visitor_id,content_id,category,title,author,months_since_epoch'.split(',')
FACTOR_COLUMNS = ["user_factor_{}".format(i) for i in range(10)] + ["item_factor_{}".format(i) for i in range(10)]
CSV_COLUMNS = NON_FACTOR_COLUMNS + FACTOR_COLUMNS
LABEL_COLUMN = 'next_content_id'

# Set default values for each CSV column
NON_FACTOR_DEFAULTS = [["Unknown"],["Unknown"],["Unknown"],["Unknown"],["Unknown"],["Unknown"],[mean_months_since_epoch]]
FACTOR_DEFAULTS = [[0.0] for i in range(10)] + [[0.0] for i in range(10)] # user and item
DEFAULTS = NON_FACTOR_DEFAULTS + FACTOR_DEFAULTS


# Create input function for training and evaluation to read from our preprocessed CSV files.

# In[13]:


# Create input function for train and eval
def read_dataset(filename, mode, batch_size = 512):
  def _input_fn():
    def decode_csv(value_column):
      columns = tf.decode_csv(records = value_column, record_defaults = DEFAULTS)
      features = dict(zip(CSV_COLUMNS, columns))          
      label = features.pop(LABEL_COLUMN)         
      return features, label

    # Create list of files that match pattern
    file_list = tf.gfile.Glob(filename = filename)

    # Create dataset from file list
    dataset = tf.data.TextLineDataset(filenames = file_list).map(map_func = decode_csv)

    if mode == tf.estimator.ModeKeys.TRAIN:
      num_epochs = None # indefinitely
      dataset = dataset.shuffle(buffer_size = 10 * batch_size)
    else:
      num_epochs = 1 # end-of-input after this

    dataset = dataset.repeat(count = num_epochs).batch(batch_size = batch_size)
    return dataset.make_one_shot_iterator().get_next()
  return _input_fn


# Next, we will create our feature columns using our read in features.

# In[16]:


# Create feature columns to be used in model
def create_feature_columns(args):
  # Create content_id feature column
  content_id_column = tf.feature_column.categorical_column_with_hash_bucket(
    key = "content_id",
    hash_bucket_size = number_of_content_ids)

  # Embed content id into a lower dimensional representation
  embedded_content_column = tf.feature_column.embedding_column(
    categorical_column = content_id_column,
    dimension = args['content_id_embedding_dimensions'])

  # Create category feature column
  categorical_category_column = tf.feature_column.categorical_column_with_vocabulary_file(
    key = "category",
    vocabulary_file = tf.gfile.Glob(
      filename = "gs://{}/hybrid_recommendation/preproc/vocabs/category_vocab.txt*".format(args['bucket']))[0],
    num_oov_buckets = 1)

  # Convert categorical category column into indicator column so that it can be used in a DNN
  indicator_category_column = tf.feature_column.indicator_column(categorical_column = categorical_category_column)

  # Create title feature column using TF Hub
  embedded_title_column = hub.text_embedding_column(
    key = "title", 
    module_spec = "https://tfhub.dev/google/nnlm-de-dim50-with-normalization/1",
    trainable = False)

  # Create author feature column
  author_column = tf.feature_column.categorical_column_with_hash_bucket(
    key = "author",
    hash_bucket_size = number_of_authors + 1)

  # Embed author into a lower dimensional representation
  embedded_author_column = tf.feature_column.embedding_column(
    categorical_column = author_column,
    dimension = args['author_embedding_dimensions'])

  # Create months since epoch boundaries list for our binning
  months_since_epoch_boundaries = list(range(400, 700, 20))

  # Create months_since_epoch feature column using raw data
  months_since_epoch_column = tf.feature_column.numeric_column(
    key = "months_since_epoch")

  # Create bucketized months_since_epoch feature column using our boundaries
  months_since_epoch_bucketized = tf.feature_column.bucketized_column(
    source_column = months_since_epoch_column,
    boundaries = months_since_epoch_boundaries)

  # Cross our categorical category column and bucketized months since epoch column
  crossed_months_since_category_column = tf.feature_column.crossed_column(
    keys = [categorical_category_column, months_since_epoch_bucketized],
    hash_bucket_size = len(months_since_epoch_boundaries) * (number_of_categories + 1))

  # Convert crossed categorical category and bucketized months since epoch column into indicator column so that it can be used in a DNN
  indicator_crossed_months_since_category_column = tf.feature_column.indicator_column(
    categorical_column = crossed_months_since_category_column
  )

  # Create user and item factor feature columns from our trained WALS model
  user_factors = [tf.feature_column.numeric_column(key = "user_factor_" + str(i)) for i in range(10)]
  item_factors =  [tf.feature_column.numeric_column(key = "item_factor_" + str(i)) for i in range(10)]

  # Create list of feature columns
  feature_columns = [embedded_content_column,
                     embedded_author_column,
                     indicator_category_column,
                     embedded_title_column,
                     indicator_crossed_months_since_category_column] + user_factors + item_factors

  return feature_columns


# Now we'll create our model function

# In[17]:


# Create custom model function for our custom estimator
def model_fn(features, labels, mode, params):
  # TODO: Create neural network input layer using our feature columns defined above
  net = tf.feature_column.input_layer(
    features = features,
    feature_columns = params['feature_columns']
  )

  # TODO: Create hidden layers by looping through hidden unit list
  for units in params['hidden_units']:
    net = tf.layers.dense(
      inputs = net,
      units = units,
      activation = tf.nn.relu
    )

  # TODO: Compute logits (1 per class) using the output of our last hidden layer
  logits = tf.layers.dense(
    inputs = net,
    units = params['n_classes'],
    activation = None
  )

  # TODO: Find the predicted class indices based on the highest logit (which will result in the highest probability)
  predicted_classes = tf.argmax(input = logits, axis = 1) 

  # Read in the content id vocabulary so we can tie the predicted class indices to their respective content ids
  with file_io.FileIO(
    tf.gfile.Glob(filename = "gs://{}/hybrid_recommendation/preproc/vocabs/content_id_vocab.txt*".format(BUCKET))[0], mode = 'r'
  ) as ifp:
    content_id_names = tf.constant(value = [x.rstrip() for x in ifp])

  # Gather predicted class names based predicted class indices
  predicted_class_names = tf.gather(params = content_id_names, indices = predicted_classes)

  # If the mode is prediction
  if mode == tf.estimator.ModeKeys.PREDICT:
    # Create predictions dict
    predictions_dict = {
        'class_ids': tf.expand_dims(input = predicted_classes, axis = -1),
        'class_names' : tf.expand_dims(input = predicted_class_names, axis = -1),
        'probabilities': tf.nn.softmax(logits = logits),
        'logits': logits
    }

    # Create export outputs
    export_outputs = {
      "predict_export_outputs": tf.estimator.export.PredictOutput(outputs = predictions_dict)
    }

    return tf.estimator.EstimatorSpec( # return early since we're done with what we need for prediction mode
      mode = mode,
      predictions = predictions_dict,
      loss = None,
      train_op = None,
      eval_metric_ops = None,
      export_outputs = export_outputs)

  # Continue on with training and evaluation modes

  # Create lookup table using our content id vocabulary
  table = tf.contrib.lookup.index_table_from_file(
    vocabulary_file = tf.gfile.Glob(
      filename = "gs://{}/hybrid_recommendation/preproc/vocabs/content_id_vocab.txt*".format(BUCKET)
    )[0]
  )

  # Look up labels from vocabulary table
  labels = table.lookup(keys = labels)

  # TODO: Compute loss using the correct type of softmax cross entropy since this is classification and 
  # our labels (content id indices) and probabilities are mutually exclusive
  loss = tf.losses.sparse_softmax_cross_entropy(
    labels = labels,
    logits = logits
  )

  # Compute evaluation metrics of total accuracy and the accuracy of the top k classes
  accuracy = tf.metrics.accuracy(labels = labels, predictions = predicted_classes, name = 'acc_op')
  top_k_accuracy = tf.metrics.mean(values = tf.nn.in_top_k(predictions = logits, targets = labels, k = params['top_k']))
  map_at_k = tf.metrics.average_precision_at_k(labels = labels, predictions = predicted_classes, k = params['top_k'])

  # Put eval metrics into a dictionary
  eval_metrics = {
    'accuracy': accuracy,
    'top_k_accuracy': top_k_accuracy,
    'map_at_k': map_at_k}

  # Create scalar summaries to see in TensorBoard
  tf.summary.scalar(name = 'accuracy', tensor = accuracy[1])
  tf.summary.scalar(name = 'top_k_accuracy', tensor = top_k_accuracy[1])
  tf.summary.scalar(name = 'map_at_k', tensor = map_at_k[1])

  # Create scalar summaries to see in TensorBoard
  #tf.summary.scalar(name = 'accuracy', tensor = accuracy[1])
  #tf.summary.scalar(name = 'top_k_accuracy', tensor = top_k_accuracy[1])

  # If the mode is evaluation
  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec( # return early since we're done with what we need for evaluation mode
        mode = mode,
        predictions = None,
        loss = loss,
        train_op = None,
        eval_metric_ops = eval_metrics,
        export_outputs = None)

  # Continue on with training mode

  # If the mode is training
  assert mode == tf.estimator.ModeKeys.TRAIN

  # Create a custom optimizer
  optimizer = tf.train.AdagradOptimizer(learning_rate = params['learning_rate'])

  # Create train op
  train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step())

  return tf.estimator.EstimatorSpec( # final return since we're done with what we need for training mode
    mode = mode,
    predictions = None,
    loss = loss,
    train_op = train_op,
    eval_metric_ops = None,
    export_outputs = None)


# Now create a serving input function

# In[18]:


# Create serving input function
def serving_input_fn():  
  feature_placeholders = {
    colname : tf.placeholder(dtype = tf.string, shape = [None]) \
    for colname in NON_FACTOR_COLUMNS[1:-1]
  }
  feature_placeholders['months_since_epoch'] = tf.placeholder(dtype = tf.float32, shape = [None])
  
  for colname in FACTOR_COLUMNS:
    feature_placeholders[colname] = tf.placeholder(dtype = tf.float32, shape = [None])

  features = {
    key: tf.expand_dims(tensor, -1) \
    for key, tensor in feature_placeholders.items()
  }
    
  return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)


# Now that all of the pieces are assembled let's create and run our train and evaluate loop

# In[19]:


# Create train and evaluate loop to combine all of the pieces together.
tf.logging.set_verbosity(tf.logging.INFO)
def train_and_evaluate(args):
  estimator = tf.estimator.Estimator(
    model_fn = model_fn,
    model_dir = args['output_dir'],
    params={
      'feature_columns': create_feature_columns(args),
      'hidden_units': args['hidden_units'],
      'n_classes': number_of_content_ids,
      'learning_rate': args['learning_rate'],
      'top_k': args['top_k'],
      'bucket': args['bucket']
    })

  train_spec = tf.estimator.TrainSpec(
    input_fn = read_dataset(filename = args['train_data_paths'], mode = tf.estimator.ModeKeys.TRAIN, batch_size = args['batch_size']),
    max_steps = args['train_steps'])

  exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)

  eval_spec = tf.estimator.EvalSpec(
    input_fn = read_dataset(filename = args['eval_data_paths'], mode = tf.estimator.ModeKeys.EVAL, batch_size = args['batch_size']),
    steps = None,
    start_delay_secs = args['start_delay_secs'],
    throttle_secs = args['throttle_secs'],
    exporters = exporter)

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


# Run train_and_evaluate!

# In[20]:


# Call train and evaluate loop
import shutil

outdir = 'hybrid_recommendation_trained'
shutil.rmtree(outdir, ignore_errors = True) # start fresh each time

arguments = {
  'bucket': BUCKET,
  'train_data_paths': "gs://{}/hybrid_recommendation/preproc/features/train.csv*".format(BUCKET),
  'eval_data_paths': "gs://{}/hybrid_recommendation/preproc/features/eval.csv*".format(BUCKET),
  'output_dir': outdir,
  'batch_size': 128,
  'learning_rate': 0.1,
  'hidden_units': [256, 128, 64],
  'content_id_embedding_dimensions': 10,
  'author_embedding_dimensions': 10,
  'top_k': 10,
  'train_steps': 1000,
  'start_delay_secs': 30,
  'throttle_secs': 30
}

train_and_evaluate(arguments)


# ## Run on module locally
# 
# Now let's place our code into a python module with model.py and task.py files so that we can train using Google Cloud's ML Engine! First, let's test our module locally.

# In[16]:


get_ipython().run_line_magic('writefile', 'requirements.txt')
tensorflow_hub


# In[17]:


get_ipython().run_cell_magic('bash', '', 'echo "bucket=${BUCKET}"\nrm -rf hybrid_recommendation_trained\nexport PYTHONPATH=${PYTHONPATH}:${PWD}/hybrid_recommendations_module\npython -m trainer.task \\\n  --bucket=${BUCKET} \\\n  --train_data_paths=gs://${BUCKET}/hybrid_recommendation/preproc/features/train.csv* \\\n  --eval_data_paths=gs://${BUCKET}/hybrid_recommendation/preproc/features/eval.csv* \\\n  --output_dir=${OUTDIR} \\\n  --batch_size=128 \\\n  --learning_rate=0.1 \\\n  --hidden_units="256 128 64" \\\n  --content_id_embedding_dimensions=10 \\\n  --author_embedding_dimensions=10 \\\n  --top_k=10 \\\n  --train_steps=1000 \\\n  --start_delay_secs=30 \\\n  --throttle_secs=60')


# # Run on Google Cloud ML Engine
# If our module locally trained fine, let's now use of the power of ML Engine to scale it out on Google Cloud.

# In[18]:


get_ipython().run_cell_magic('bash', '', 'OUTDIR=gs://${BUCKET}/hybrid_recommendation/small_trained_model\nJOBNAME=hybrid_recommendation_$(date -u +%y%m%d_%H%M%S)\necho $OUTDIR $REGION $JOBNAME\ngsutil -m rm -rf $OUTDIR\ngcloud ml-engine jobs submit training $JOBNAME \\\n  --region=$REGION \\\n  --module-name=trainer.task \\\n  --package-path=$(pwd)/hybrid_recommendations_module/trainer \\\n  --job-dir=$OUTDIR \\\n  --staging-bucket=gs://$BUCKET \\\n  --scale-tier=STANDARD_1 \\\n  --runtime-version=$TFVERSION \\\n  -- \\\n  --bucket=${BUCKET} \\\n  --train_data_paths=gs://${BUCKET}/hybrid_recommendation/preproc/features/train.csv* \\\n  --eval_data_paths=gs://${BUCKET}/hybrid_recommendation/preproc/features/eval.csv* \\\n  --output_dir=${OUTDIR} \\\n  --batch_size=128 \\\n  --learning_rate=0.1 \\\n  --hidden_units="256 128 64" \\\n  --content_id_embedding_dimensions=10 \\\n  --author_embedding_dimensions=10 \\\n  --top_k=10 \\\n  --train_steps=1000 \\\n  --start_delay_secs=30 \\\n  --throttle_secs=30')


# Results:
# 
# ``` 
# Saving dict for global step 1008: accuracy = 0.02687605, global_step = 1008, loss = 5.470378, map_at_k = 0.05117718253968257, top_k_accuracy = 0.2249502
# ```

# Let's add some hyperparameter tuning!

# In[23]:


get_ipython().run_cell_magic('writefile', 'hyperparam.yaml', "trainingInput:\n  hyperparameters:\n    goal: MAXIMIZE\n    maxTrials: 5\n    maxParallelTrials: 1\n    hyperparameterMetricTag: accuracy\n    params:\n    - parameterName: batch_size\n      type: INTEGER\n      minValue: 8\n      maxValue: 64\n      scaleType: UNIT_LINEAR_SCALE\n    - parameterName: learning_rate\n      type: DOUBLE\n      minValue: 0.01\n      maxValue: 0.1\n      scaleType: UNIT_LINEAR_SCALE\n    - parameterName: hidden_units\n      type: CATEGORICAL\n      categoricalValues: ['1024 512 256', '1024 512 128', '1024 256 128', '512 256 128', '1024 512 64', '1024 256 64', '512 256 64', '1024 128 64', '512 128 64', '256 128 64', '1024 512 32', '1024 256 32', '512 256 32', '1024 128 32', '512 128 32', '256 128 32', '1024 64 32', '512 64 32', '256 64 32', '128 64 32']\n    - parameterName: content_id_embedding_dimensions\n      type: INTEGER\n      minValue: 5\n      maxValue: 250\n      scaleType: UNIT_LOG_SCALE\n    - parameterName: author_embedding_dimensions\n      type: INTEGER\n      minValue: 5\n      maxValue: 30\n      scaleType: UNIT_LINEAR_SCALE")


# In[24]:


get_ipython().run_cell_magic('bash', '', 'OUTDIR=gs://${BUCKET}/hybrid_recommendation/hypertuning\nJOBNAME=hybrid_recommendation_$(date -u +%y%m%d_%H%M%S)\necho $OUTDIR $REGION $JOBNAME\ngsutil -m rm -rf $OUTDIR\ngcloud ml-engine jobs submit training $JOBNAME \\\n  --region=$REGION \\\n  --module-name=trainer.task \\\n  --package-path=$(pwd)/hybrid_recommendations_module/trainer \\\n  --job-dir=$OUTDIR \\\n  --staging-bucket=gs://$BUCKET \\\n  --scale-tier=STANDARD_1 \\\n  --runtime-version=$TFVERSION \\\n  --config=hyperparam.yaml \\\n  -- \\\n  --bucket=${BUCKET} \\\n  --train_data_paths=gs://${BUCKET}/hybrid_recommendation/preproc/features/train.csv* \\\n  --eval_data_paths=gs://${BUCKET}/hybrid_recommendation/preproc/features/eval.csv* \\\n  --output_dir=${OUTDIR} \\\n  --batch_size=128 \\\n  --learning_rate=0.1 \\\n  --hidden_units="256 128 64" \\\n  --content_id_embedding_dimensions=10 \\\n  --author_embedding_dimensions=10 \\\n  --top_k=10 \\\n  --train_steps=1000 \\\n  --start_delay_secs=30 \\\n  --throttle_secs=30')


# Results can be found in the *Job Details* under *Training Output*:
# 
# ```
# Training output
# {
#   "completedTrialCount": "5",
#   "trials": [
#     {
#       "trialId": "5",
#       "hyperparameters": {
#         "batch_size": "40",
#         "learning_rate": "0.072136064767837529",
#         "author_embedding_dimensions": "19",
#         "hidden_units": "256 128 64",
#         "content_id_embedding_dimensions": "139"
#       },
#       "finalMetric": {
#         "trainingStep": "1007",
#         "objectiveValue": 0.0253916177899
#       }
#     },
#     {
#       "trialId": "2",
#       "hyperparameters": {
#         "hidden_units": "256 128 32",
#         "content_id_embedding_dimensions": "21",
#         "batch_size": "46",
#         "learning_rate": "0.031392376422882083",
#         "author_embedding_dimensions": "12"
#       },
#       "finalMetric": {
#         "trainingStep": "1007",
#         "objectiveValue": 0.022344622761
#       }
#     },
#     {
#       "trialId": "1",
#       "hyperparameters": {
#         "batch_size": "43",
#         "learning_rate": "0.073695758581161508",
#         "author_embedding_dimensions": "21",
#         "hidden_units": "512 256 64",
#         "content_id_embedding_dimensions": "78"
#       },
#       "finalMetric": {
#         "trainingStep": "1006",
#         "objectiveValue": 0.0157818663865
#       }
#     },
#     {
#       "trialId": "4",
#       "hyperparameters": {
#         "hidden_units": "1024 512 128",
#         "content_id_embedding_dimensions": "22",
#         "batch_size": "48",
#         "learning_rate": "0.084746223688125608",
#         "author_embedding_dimensions": "9"
#       },
#       "finalMetric": {
#         "trainingStep": "1004",
#         "objectiveValue": 0.0108988629654
#       }
#     },
#     {
#       "trialId": "3",
#       "hyperparameters": {
#         "hidden_units": "1024 512 128",
#         "content_id_embedding_dimensions": "245",
#         "batch_size": "8",
#         "learning_rate": "0.080643538236618045",
#         "author_embedding_dimensions": "8"
#       }
#     }
#   ],
#   "consumedMLUnits": 3.55,
#   "isHyperparameterTuningJob": true
# }
# ```

# Now that we know the best hyperparameters, run a big training job!

# In[21]:


get_ipython().run_cell_magic('bash', '', 'OUTDIR=gs://${BUCKET}/hybrid_recommendation/big_trained_model\nJOBNAME=hybrid_recommendation_$(date -u +%y%m%d_%H%M%S)\necho $OUTDIR $REGION $JOBNAME\ngsutil -m rm -rf $OUTDIR\ngcloud ml-engine jobs submit training $JOBNAME \\\n  --region=$REGION \\\n  --module-name=trainer.task \\\n  --package-path=$(pwd)/hybrid_recommendations_module/trainer \\\n  --job-dir=$OUTDIR \\\n  --staging-bucket=gs://$BUCKET \\\n  --scale-tier=STANDARD_1 \\\n  --runtime-version=$TFVERSION \\\n  -- \\\n  --bucket=${BUCKET} \\\n  --train_data_paths=gs://${BUCKET}/hybrid_recommendation/preproc/features/train.csv* \\\n  --eval_data_paths=gs://${BUCKET}/hybrid_recommendation/preproc/features/eval.csv* \\\n  --output_dir=${OUTDIR} \\\n  --batch_size=128 \\\n  --learning_rate=0.1 \\\n  --hidden_units="256 128 64" \\\n  --content_id_embedding_dimensions=10 \\\n  --author_embedding_dimensions=10 \\\n  --top_k=10 \\\n  --train_steps=10000 \\\n  --start_delay_secs=30 \\\n  --throttle_secs=30')


# Results:
# 
# First run:
# 
# ```
# Saving dict for global step 10008: accuracy = 0.045392398, global_step = 10008, loss = 5.0295167, map_at_k = 0.04409563492063496, top_k_accuracy = 0.27708113
# ```
# 
# Second run:
# 
# ```
# Saving dict for global step 10008: accuracy = 0.059221063, global_step = 10008, loss = 4.820803, map_at_k = 0.04631904761904759, top_k_accuracy = 0.33321613
# ```