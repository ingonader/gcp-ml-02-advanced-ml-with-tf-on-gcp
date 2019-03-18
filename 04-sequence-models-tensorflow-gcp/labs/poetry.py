
# coding: utf-8

# ## Text generation using tensor2tensor on Cloud ML Engine
# 
# This notebook illustrates using the <a href="https://github.com/tensorflow/tensor2tensor">tensor2tensor</a> library to do from-scratch, distributed training of a poetry model. Then, the trained model is used to complete new poems.
# 
# <br/>
# 
# ### Install tensor2tensor, and specify Google Cloud Platform project and bucket

# Install the necessary packages. tensor2tensor will give us the Transformer model. Project Gutenberg gives us access to historical poems.
# 
# 
# <b>p.s.</b> Note that this notebook uses Python2 because Project Gutenberg relies on BSD-DB which was deprecated in Python 3 and removed from the standard library.
# tensor2tensor itself can be used on Python 3. It's just Project Gutenberg that has this issue.

# In[1]:


get_ipython().run_cell_magic('bash', '', 'pip freeze | grep tensor')


# In[2]:


# Choose a version of TensorFlow that is supported on TPUs
TFVERSION='1.10'
import os
os.environ['TFVERSION'] = TFVERSION


# In[3]:


get_ipython().run_cell_magic('bash', '', 'pip install tensor2tensor==${TFVERSION} tensorflow==${TFVERSION} gutenberg \n#git clone https://github.com/tensorflow/tensor2tensor.git\n#cd tensor2tensor; pip install --user -e .')


# If the following cell does not reflect the version of tensorflow and tensor2tensor that you just installed, click **"Reset Session"** on the notebook so that the Python environment picks up the new packages.

# In[4]:


get_ipython().run_cell_magic('bash', '', 'pip freeze | grep tensor')


# In[5]:


import os
output = os.popen("gcloud config get-value project").readlines()
project_name = output[0][:-1]

# change these to try this notebook out
PROJECT = project_name
BUCKET = project_name
#BUCKET = BUCKET.replace("qwiklabs-gcp-", "inna-bckt-")
REGION = 'europe-west1'  ## note: Cloud ML Engine not availabe in europe-west3!


# this is what this notebook is demonstrating
PROBLEM= 'poetry_line_problem'

# for bash
os.environ['PROJECT'] = PROJECT
os.environ['BUCKET'] = BUCKET
os.environ['REGION'] = REGION
os.environ['PROBLEM'] = PROBLEM

print(PROJECT)
print(BUCKET)
print(REGION)

#os.environ['PATH'] = os.environ['PATH'] + ':' + os.getcwd() + '/tensor2tensor/tensor2tensor/bin/'


# In[6]:


get_ipython().run_cell_magic('bash', '', 'gcloud config set project $PROJECT\ngcloud config set compute/region $REGION')


# ### Download data
# 
# We will get some <a href="https://www.gutenberg.org/wiki/Poetry_(Bookshelf)">poetry anthologies</a> from Project Gutenberg.

# In[7]:


get_ipython().run_cell_magic('bash', '', 'rm -rf data/poetry\nmkdir -p data/poetry')


# In[8]:


from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
import re

books = [
  # bookid, skip N lines
  (26715, 1000, 'Victorian songs'),
  (30235, 580, 'Baldwin collection'),
  (35402, 710, 'Swinburne collection'),
  (574, 15, 'Blake'),
  (1304, 172, 'Bulchevys collection'),
  (19221, 223, 'Palgrave-Pearse collection'),
  (15553, 522, 'Knowles collection') 
]

with open('data/poetry/raw.txt', 'w') as ofp:
  lineno = 0
  for (id_nr, toskip, title) in books:
    startline = lineno
    text = strip_headers(load_etext(id_nr)).strip()
    lines = text.split('\n')[toskip:]
    # any line that is all upper case is a title or author name
    # also don't want any lines with years (numbers)
    for line in lines:
      if (len(line) > 0 
          and line.upper() != line 
          and not re.match('.*[0-9]+.*', line)
          and len(line) < 50
         ):
        cleaned = re.sub('[^a-z\'\-]+', ' ', line.strip().lower())
        ofp.write(cleaned)
        ofp.write('\n')
        lineno = lineno + 1
      else:
        ofp.write('\n')
    print('Wrote lines {} to {} from {}'.format(startline, lineno, title))


# In[9]:


get_ipython().system('wc -l data/poetry/*.txt')


# In[10]:


get_ipython().system('head data/poetry/*.txt')


# ## Create training dataset
# 
# We are going to train a machine learning model to write poetry given a starting point. We'll give it one line, and it is going to tell us the next line.  So, naturally, we will train it on real poetry. Our feature will be a line of a poem and the label will be next line of that poem.
# <p>
# Our training dataset will consist of two files.  The first file will consist of the input lines of poetry and the other file will consist of the corresponding output lines, one output line per input line.

# In[11]:


with open('data/poetry/raw.txt', 'r') as rawfp,  open('data/poetry/input.txt', 'w') as infp,  open('data/poetry/output.txt', 'w') as outfp:
    
    prev_line = ''
    for curr_line in rawfp:
        curr_line = curr_line.strip()
        # poems break at empty lines, so this ensures we train only
        # on lines of the same poem
        if len(prev_line) > 0 and len(curr_line) > 0:       
            infp.write(prev_line + '\n')
            outfp.write(curr_line + '\n')
        prev_line = curr_line      


# In[12]:


get_ipython().system('head -5 data/poetry/*.txt')


# We do not need to generate the data beforehand -- instead, we can have Tensor2Tensor create the training dataset for us. So, in the code below, I will use only data/poetry/raw.txt -- obviously, this allows us to productionize our model better.  Simply keep collecting raw data and generate the training/test data at the time of training.

# ### Set up problem
# The Problem in tensor2tensor is where you specify parameters like the size of your vocabulary and where to get the training data from.

# In[13]:


get_ipython().run_cell_magic('bash', '', 'rm -rf poetry\nmkdir -p poetry/trainer')


# In[14]:


get_ipython().run_cell_magic('writefile', 'poetry/trainer/problem.py', 'import os\nimport tensorflow as tf\nfrom tensor2tensor.utils import registry\nfrom tensor2tensor.models import transformer\nfrom tensor2tensor.data_generators import problem\nfrom tensor2tensor.data_generators import text_encoder\nfrom tensor2tensor.data_generators import text_problems\nfrom tensor2tensor.data_generators import generator_utils\n\n\n@registry.register_problem\nclass PoetryLineProblem(text_problems.Text2TextProblem):\n  """Predict next line of poetry from the last line. From Gutenberg texts."""\n\n  @property\n  def approx_vocab_size(self):\n    return 2**13  # ~8k\n\n  @property\n  def is_generate_per_split(self):\n    # generate_data will NOT shard the data into TRAIN and EVAL for us.\n    return False\n\n  @property\n  def dataset_splits(self):\n    """Splits of data to produce and number of output shards for each."""\n    # 10% evaluation data\n    return [{\n        "split": problem.DatasetSplit.TRAIN,\n        "shards": 90,\n    }, {\n        "split": problem.DatasetSplit.EVAL,\n        "shards": 10,\n    }]\n\n  def generate_samples(self, data_dir, tmp_dir, dataset_split):\n    with open(\'data/poetry/raw.txt\', \'r\') as rawfp:\n      prev_line = \'\'\n      for curr_line in rawfp:\n        curr_line = curr_line.strip()\n        # poems break at empty lines, so this ensures we train only\n        # on lines of the same poem\n        if len(prev_line) > 0 and len(curr_line) > 0:       \n            yield {\n                "inputs": prev_line,\n                "targets": curr_line\n            }\n        prev_line = curr_line          \n\n\n# Smaller than the typical translate model, and with more regularization\n@registry.register_hparams\ndef transformer_poetry():\n  hparams = transformer.transformer_base()\n  hparams.num_hidden_layers = 2\n  hparams.hidden_size = 128\n  hparams.filter_size = 512\n  hparams.num_heads = 4\n  hparams.attention_dropout = 0.6\n  hparams.layer_prepostprocess_dropout = 0.6\n  hparams.learning_rate = 0.05\n  return hparams\n\n@registry.register_hparams\ndef transformer_poetry_tpu():\n  hparams = transformer_poetry()\n  transformer.update_hparams_for_tpu(hparams)\n  return hparams\n\n# hyperparameter tuning ranges\n@registry.register_ranged_hparams\ndef transformer_poetry_range(rhp):\n  rhp.set_float("learning_rate", 0.05, 0.25, scale=rhp.LOG_SCALE)\n  rhp.set_int("num_hidden_layers", 2, 4)\n  rhp.set_discrete("hidden_size", [128, 256, 512])\n  rhp.set_float("attention_dropout", 0.4, 0.7)')


# In[15]:


get_ipython().run_cell_magic('writefile', 'poetry/trainer/__init__.py', 'from . import problem')


# In[16]:


get_ipython().run_cell_magic('writefile', 'poetry/setup.py', "from setuptools import find_packages\nfrom setuptools import setup\n\nREQUIRED_PACKAGES = [\n  'tensor2tensor'\n]\n\nsetup(\n    name='poetry',\n    version='0.1',\n    author = 'Google',\n    author_email = 'training-feedback@cloud.google.com',\n    install_requires=REQUIRED_PACKAGES,\n    packages=find_packages(),\n    include_package_data=True,\n    description='Poetry Line Problem',\n    requires=[]\n)")


# In[17]:


get_ipython().system('touch poetry/__init__.py')


# In[18]:


get_ipython().system('find poetry')


# ## Generate training data 
# 
# Our problem (translation) requires the creation of text sequences from the training dataset.  This is done using t2t-datagen and the Problem defined in the previous section.
# 
# (Ignore any runtime warnings about np.float64. they are harmless).

# In[19]:


get_ipython().run_cell_magic('bash', '', 'DATA_DIR=./t2t_data\nTMP_DIR=$DATA_DIR/tmp\nrm -rf $DATA_DIR $TMP_DIR\nmkdir -p $DATA_DIR $TMP_DIR\n# Generate data\nt2t-datagen \\\n  --t2t_usr_dir=./poetry/trainer \\\n  --problem=$PROBLEM \\\n  --data_dir=$DATA_DIR \\\n  --tmp_dir=$TMP_DIR')


# In[20]:


get_ipython().system('ls t2t_data | head')


# ## Provide Cloud ML Engine access to data
# 
# Copy the data to Google Cloud Storage, and then provide access to the data

# In[21]:


get_ipython().run_cell_magic('bash', '', 'DATA_DIR=./t2t_data\ngsutil -m rm -r gs://${BUCKET}/poetry/\ngsutil -m cp ${DATA_DIR}/${PROBLEM}* ${DATA_DIR}/vocab* gs://${BUCKET}/poetry/data')


# In[22]:


get_ipython().run_cell_magic('bash', '', 'PROJECT_ID=$PROJECT\nAUTH_TOKEN=$(gcloud auth print-access-token)\nSVC_ACCOUNT=$(curl -X GET -H "Content-Type: application/json" \\\n    -H "Authorization: Bearer $AUTH_TOKEN" \\\n    https://ml.googleapis.com/v1/projects/${PROJECT_ID}:getConfig \\\n    | python -c "import json; import sys; response = json.load(sys.stdin); \\\n    print(response[\'serviceAccount\'])")\n\necho "Authorizing the Cloud ML Service account $SVC_ACCOUNT to access files in $BUCKET"\ngsutil -m defacl ch -u $SVC_ACCOUNT:R gs://$BUCKET\ngsutil -m acl ch -u $SVC_ACCOUNT:R -r gs://$BUCKET  # error message (if bucket is empty) can be ignored\ngsutil -m acl ch -u $SVC_ACCOUNT:W gs://$BUCKET')


# ## Train model locally
# 
# Let's run it locally on a subset of the data to make sure it works.

# In[23]:


get_ipython().run_cell_magic('bash', '', 'BASE=gs://${BUCKET}/poetry/data\nOUTDIR=gs://${BUCKET}/poetry/subset\ngsutil -m rm -r $OUTDIR\ngsutil -m cp \\\n    ${BASE}/${PROBLEM}-train-0008* \\\n    ${BASE}/${PROBLEM}-dev-00000*  \\\n    ${BASE}/vocab* \\\n    $OUTDIR')


# Note: the following will work only if you are running Datalab on a reasonably powerful machine. Don't be alarmed if your process is killed.

# In[24]:


get_ipython().run_cell_magic('bash', '', 'DATA_DIR=gs://${BUCKET}/poetry/subset\nOUTDIR=./trained_model\nrm -rf $OUTDIR\nt2t-trainer \\\n  --data_dir=gs://${BUCKET}/poetry/subset \\\n  --t2t_usr_dir=./poetry/trainer \\\n  --problem=$PROBLEM \\\n  --model=transformer \\\n  --hparams_set=transformer_poetry \\\n  --output_dir=$OUTDIR --job-dir=$OUTDIR --train_steps=10')


# ## Train on Cloud ML Engine
# 
# tensor2tensor has a convenient --cloud_mlengine option to kick off the training on the managed service.
# It uses the [Python API](https://cloud.google.com/ml-engine/docs/training-jobs) mentioned in the Cloud ML Engine docs, rather than requiring you to use gcloud to submit the job.
# <p>
# Note: your project needs P100 quota in the region.
# <p>
# The echo is because t2t-trainer asks you to confirm before submitting the job to the cloud.

# In[25]:


get_ipython().run_cell_magic('bash', '', 'GPU="--train_steps=7500 --cloud_mlengine --worker_gpu=1 --hparams_set=transformer_poetry"\n\nDATADIR=gs://${BUCKET}/poetry/data\nOUTDIR=gs://${BUCKET}/poetry/model\nJOBNAME=poetry_$(date -u +%y%m%d_%H%M%S)\necho $OUTDIR $REGION $JOBNAME\ngsutil -m rm -rf $OUTDIR\necho "\'Y\'" | t2t-trainer \\\n  --data_dir=gs://${BUCKET}/poetry/subset \\\n  --t2t_usr_dir=./poetry/trainer \\\n  --problem=$PROBLEM \\\n  --model=transformer \\\n  --output_dir=$OUTDIR \\\n  ${GPU}')


# The job took about <b>25 minutes</b> for me and ended with these evaluation metrics:
# <pre>
# Saving dict for global step 8000: global_step = 8000, loss = 6.03338, 
# metrics-poetry_line_problem/accuracy = 0.138544, 
# metrics-poetry_line_problem/accuracy_per_sequence = 0.0, 
# metrics-poetry_line_problem/accuracy_top5 = 0.232037, 
# metrics-poetry_line_problem/approx_bleu_score = 0.00492648, 
# metrics-poetry_line_problem/neg_log_perplexity = -6.68994, 
# metrics-poetry_line_problem/rouge_2_fscore = 0.00256089, 
# metrics-poetry_line_problem/rouge_L_fscore = 0.128194
# </pre>
# 
# Notice that accuracy_per_sequence is 0 -- Considering that we are asking the NN to be rather creative, that doesn't surprise me. Why am I looking at accuracy_per_sequence and not the other metrics? This is because it is more appropriate for problem we are solving; metrics like Bleu score are better for translation.

# What I got doing the lab (18 Mar 2019):
# 
# ```
# Saving dict for global step 7500: global_step = 7500, loss = 7.2016287, 
# metrics-poetry_line_problem/targets/accuracy = 0.16934386, 
# metrics-poetry_line_problem/targets/accuracy_per_sequence = 0.0, 
# metrics-poetry_line_problem/targets/accuracy_top5 = 0.2730738, 
# metrics-poetry_line_problem/targets/approx_bleu_score = 0.013282104, 
# metrics-poetry_line_problem/targets/neg_log_perplexity = -7.0870624,
# metrics-poetry_line_problem/targets/rouge_2_fscore = 0.017581841, 
# metrics-poetry_line_problem/targets/rouge_L_fscore = 0.19151948
# ```

# In[26]:


get_ipython().run_cell_magic('bash', '', 'gsutil ls gs://${BUCKET}/poetry/model')


# ## Train on a directly-connected TPU
# 
# If you are running on a VM connected directly to a Cloud TPU, you can run t2t-trainer directly. Unfortunately, you won't see any output from Jupyter while the program is running.
# 
# Compare this command line to the one using GPU in the previous section.

# First, enable the Cloud TPU API, otherwise this will error out:
# 
# ```
# googleapiclient.errors.HttpError: <HttpError 403 when requesting https://tpu.googleapis.com/v1alpha1/projects/qwiklabs-gcp-36677528dc4d202e/locations/europe-west1-c/nodes/laktpu?alt=json returned "Cloud TPU API has not been used in project 1032299714650 before or it is disabled. Enable it by visiting https://console.developers.google.com/apis/api/tpu.googleapis.com/overview?project=1032299714650 then retry. If you enabled this API recently, wait a few minutes for the action to propagate to our systems and retry.">
# ```

# In[28]:


get_ipython().run_cell_magic('bash', '', '# use one of these\nTPU="--train_steps=7500 --use_tpu=True --cloud_tpu_name=laktpu --hparams_set=transformer_poetry_tpu"\n\nDATADIR=gs://${BUCKET}/poetry/data\nOUTDIR=gs://${BUCKET}/poetry/model_tpu\nJOBNAME=poetry_$(date -u +%y%m%d_%H%M%S)\necho $OUTDIR $REGION $JOBNAME\ngsutil -m rm -rf $OUTDIR\necho "\'Y\'" | t2t-trainer \\\n  --data_dir=gs://${BUCKET}/poetry/subset \\\n  --t2t_usr_dir=./poetry/trainer \\\n  --problem=$PROBLEM \\\n  --model=transformer \\\n  --output_dir=$OUTDIR \\\n  ${TPU}')


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'gsutil ls gs://${BUCKET}/poetry/model_tpu')


# The job took about <b>10 minutes</b> for me and ended with these evaluation metrics:
# <pre>
# Saving dict for global step 8000: global_step = 8000, loss = 6.03338, metrics-poetry_line_problem/accuracy = 0.138544, metrics-poetry_line_problem/accuracy_per_sequence = 0.0, metrics-poetry_line_problem/accuracy_top5 = 0.232037, metrics-poetry_line_problem/approx_bleu_score = 0.00492648, metrics-poetry_line_problem/neg_log_perplexity = -6.68994, metrics-poetry_line_problem/rouge_2_fscore = 0.00256089, metrics-poetry_line_problem/rouge_L_fscore = 0.128194
# </pre>
# Notice that accuracy_per_sequence is 0 -- Considering that we are asking the NN to be rather creative, that doesn't surprise me. Why am I looking at accuracy_per_sequence and not the other metrics? This is because it is more appropriate for problem we are solving; metrics like Bleu score are better for translation.

# ## Training longer
# 
# Let's train on 4 GPUs for 75,000 steps. Note the change in the last line of the job.

# In[ ]:


get_ipython().run_cell_magic('bash', '', '\nXXX This takes 3 hours on 4 GPUs. Remove this line if you are sure you want to do this.\n\nDATADIR=gs://${BUCKET}/poetry/data\nOUTDIR=gs://${BUCKET}/poetry/model_full2\nJOBNAME=poetry_$(date -u +%y%m%d_%H%M%S)\necho $OUTDIR $REGION $JOBNAME\ngsutil -m rm -rf $OUTDIR\necho "\'Y\'" | t2t-trainer \\\n  --data_dir=gs://${BUCKET}/poetry/subset \\\n  --t2t_usr_dir=./poetry/trainer \\\n  --problem=$PROBLEM \\\n  --model=transformer \\\n  --hparams_set=transformer_poetry \\\n  --output_dir=$OUTDIR \\\n  --train_steps=75000 --cloud_mlengine --worker_gpu=4')


# This job took <b>12 hours</b> for me and ended with these metrics:
# <pre>
# global_step = 76000, loss = 4.99763, metrics-poetry_line_problem/accuracy = 0.219792, metrics-poetry_line_problem/accuracy_per_sequence = 0.0192308, metrics-poetry_line_problem/accuracy_top5 = 0.37618, metrics-poetry_line_problem/approx_bleu_score = 0.017955, metrics-poetry_line_problem/neg_log_perplexity = -5.38725, metrics-poetry_line_problem/rouge_2_fscore = 0.0325563, metrics-poetry_line_problem/rouge_L_fscore = 0.210618
# </pre>
# At least the accuracy per sequence is no longer zero. It is now 0.0192308 ... note that we are using a relatively small dataset (12K lines) and this is *tiny* in the world of natural language problems.
# <p>
# In order that you have your expectations set correctly: a high-performing translation model needs 400-million lines of input and takes 1 whole day on a TPU pod!

# ## Batch-predict
# 
# How will our poetry model do when faced with Rumi's spiritual couplets?

# In[29]:


get_ipython().run_cell_magic('writefile', 'data/poetry/rumi.txt', 'Where did the handsome beloved go?\nI wonder, where did that tall, shapely cypress tree go?\nHe spread his light among us like a candle.\nWhere did he go? So strange, where did he go without me?\nAll day long my heart trembles like a leaf.\nAll alone at midnight, where did that beloved go?\nGo to the road, and ask any passing traveler\u2009—\u2009\nThat soul-stirring companion, where did he go?\nGo to the garden, and ask the gardener\u2009—\u2009\nThat tall, shapely rose stem, where did he go?\nGo to the rooftop, and ask the watchman\u2009—\u2009\nThat unique sultan, where did he go?\nLike a madman, I search in the meadows!\nThat deer in the meadows, where did he go?\nMy tearful eyes overflow like a river\u2009—\u2009\nThat pearl in the vast sea, where did he go?\nAll night long, I implore both moon and Venus\u2009—\u2009\nThat lovely face, like a moon, where did he go?\nIf he is mine, why is he with others?\nSince he’s not here, to what “there” did he go?\nIf his heart and soul are joined with God,\nAnd he left this realm of earth and water, where did he go?\nTell me clearly, Shams of Tabriz,\nOf whom it is said, “The sun never dies”\u2009—\u2009where did he go?')


# Let's write out the odd-numbered lines. We'll compare how close our model can get to the beauty of Rumi's second lines given his first.

# In[30]:


get_ipython().run_cell_magic('bash', '', 'awk \'NR % 2 == 1\' data/poetry/rumi.txt | tr \'[:upper:]\' \'[:lower:]\' | sed "s/[^a-z\\\'-\\ ]//g" > data/poetry/rumi_leads.txt\nhead -3 data/poetry/rumi_leads.txt')


# In[31]:


get_ipython().run_cell_magic('bash', '', '# same as the above training job ...\nTOPDIR=gs://${BUCKET}\nOUTDIR=${TOPDIR}/poetry/model #_tpu  # or ${TOPDIR}/poetry/model_full\nDATADIR=${TOPDIR}/poetry/data\nMODEL=transformer\nHPARAMS=transformer_poetry #_tpu\n\n# the file with the input lines\nDECODE_FILE=data/poetry/rumi_leads.txt\n\nBEAM_SIZE=4\nALPHA=0.6\n\nt2t-decoder \\\n  --data_dir=$DATADIR \\\n  --problem=$PROBLEM \\\n  --model=$MODEL \\\n  --hparams_set=$HPARAMS \\\n  --output_dir=$OUTDIR \\\n  --t2t_usr_dir=./poetry/trainer \\\n  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \\\n  --decode_from_file=$DECODE_FILE')


# <b> Note </b> if you get an error about "AttributeError: 'HParams' object has no attribute 'problems'" please <b>Reset Session</b>, run the cell that defines the PROBLEM and run the above cell again.

# In[32]:


get_ipython().run_cell_magic('bash', ' ', 'DECODE_FILE=data/poetry/rumi_leads.txt\ncat ${DECODE_FILE}.*.decodes')


# Some of these are still phrases and not complete sentences. This indicates that we might need to train longer or better somehow. We need to diagnose the model ...
# <p>
#     
# ### Diagnosing training run
# 
# <p>
# Let's diagnose the training run to see what we'd improve the next time around.
# (Note that this package may not be present on Jupyter -- `pip install pydatalab` if necessary)

# In[33]:


from google.datalab.ml import TensorBoard
TensorBoard().start('gs://{}/poetry/model_full'.format(BUCKET))


# In[34]:


for pid in TensorBoard.list()['pid']:
    TensorBoard().stop(pid)
    print('Stopped TensorBoard with pid {}'.format(pid))


# <table>
# <tr>
# <td><img src="diagrams/poetry_loss.png"/></td>
# <td><img src="diagrams/poetry_acc.png"/></td>
# </table>
# Looking at the loss curve, it is clear that we are overfitting (note that the orange training curve is well below the blue eval curve). Both loss curves and the accuracy-per-sequence curve, which is our key evaluation measure, plateaus after 40k. (The red curve is a faster way of computing the evaluation metric, and can be ignored). So, how do we improve the model? Well, we need to reduce overfitting and make sure the eval metrics keep going down as long as the loss is also going down.
# <p>
# What we really need to do is to get more data, but if that's not an option, we could try to reduce the NN and increase the dropout regularization. We could also do hyperparameter tuning on the dropout and network sizes.

# ## Hyperparameter tuning
# 
# tensor2tensor also supports hyperparameter tuning on Cloud ML Engine. Note the addition of the autotune flags.
# <p>
# The `transformer_poetry_range` was registered in problem.py above.

# In[ ]:


get_ipython().run_cell_magic('bash', '', '\nXXX This takes about 15 hours and consumes about 420 ML units.  Uncomment if you wish to proceed anyway\n\nDATADIR=gs://${BUCKET}/poetry/data\nOUTDIR=gs://${BUCKET}/poetry/model_hparam\nJOBNAME=poetry_$(date -u +%y%m%d_%H%M%S)\necho $OUTDIR $REGION $JOBNAME\ngsutil -m rm -rf $OUTDIR\necho "\'Y\'" | t2t-trainer \\\n  --data_dir=gs://${BUCKET}/poetry/subset \\\n  --t2t_usr_dir=./poetry/trainer \\\n  --problem=$PROBLEM \\\n  --model=transformer \\\n  --hparams_set=transformer_poetry \\\n  --output_dir=$OUTDIR \\\n  --hparams_range=transformer_poetry_range \\\n  --autotune_objective=\'metrics-poetry_line_problem/accuracy_per_sequence\' \\\n  --autotune_maximize \\\n  --autotune_max_trials=4 \\\n  --autotune_parallel_trials=4 \\\n  --train_steps=7500 --cloud_mlengine --worker_gpu=4')


# When I ran the above job, it took about 15 hours and finished with these as the best parameters:
# <pre>
# {
#       "trialId": "37",
#       "hyperparameters": {
#         "hp_num_hidden_layers": "4",
#         "hp_learning_rate": "0.026711152525921437",
#         "hp_hidden_size": "512",
#         "hp_attention_dropout": "0.60589466163419292"
#       },
#       "finalMetric": {
#         "trainingStep": "8000",
#         "objectiveValue": 0.0276162791997
#       }
# </pre>
# In other words, the accuracy per sequence achieved was 0.027 (as compared to 0.019 before hyperparameter tuning, so a <b>40% improvement!</b>) using 4 hidden layers, a learning rate of 0.0267, a hidden size of 512 and droput probability of 0.606. This is inspite of training for only 7500 steps instead of 75,000 steps ... we could train for 75k steps with these parameters, but I'll leave that as an exercise for you.
# <p>
# Instead, let's try predicting with this optimized model. Note the addition of the hp* flags in order to override the values hardcoded in the source code. (there is no need to specify learning rate and dropout because they are not used during inference). I am using 37 because I got the best result at trialId=37

# In[ ]:


get_ipython().run_cell_magic('bash', '', '# same as the above training job ...\nBEST_TRIAL=28  # CHANGE as needed.\nTOPDIR=gs://${BUCKET}\nOUTDIR=${TOPDIR}/poetry/model_hparam/$BEST_TRIAL\nDATADIR=${TOPDIR}/poetry/data\nMODEL=transformer\nHPARAMS=transformer_poetry\n\n# the file with the input lines\nDECODE_FILE=data/poetry/rumi_leads.txt\n\nBEAM_SIZE=4\nALPHA=0.6\n\nt2t-decoder \\\n  --data_dir=$DATADIR \\\n  --problem=$PROBLEM \\\n  --model=$MODEL \\\n  --hparams_set=$HPARAMS \\\n  --output_dir=$OUTDIR \\\n  --t2t_usr_dir=./poetry/trainer \\\n  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \\\n  --decode_from_file=$DECODE_FILE \\\n  --hparams="num_hidden_layers=4,hidden_size=512"')


# In[ ]:


get_ipython().run_cell_magic('bash', ' ', 'DECODE_FILE=data/poetry/rumi_leads.txt\ncat ${DECODE_FILE}.*.decodes')


# Take the first three line. I'm showing the first line of the couplet provided to the model, how the AI model that we trained complets it and how Rumi completes it:
# <p>
# INPUT: where did the handsome beloved go <br/>
# AI: where art thou worse to me than dead <br/>
# RUMI: I wonder, where did that tall, shapely cypress tree go?
# <p>
# INPUT: he spread his light among us like a candle <br/>
# AI: like the hurricane eclipse <br/>
# RUMI: Where did he go? So strange, where did he go without me? <br/>
# <p>
# INPUT: all day long my heart trembles like a leaf <br/>
# AI: and through their hollow aisles it plays <br/>
# RUMI: All alone at midnight, where did that beloved go? 
# <p>
# Oh wow. The couplets as completed are quite decent considering that:
#     
# * We trained the model on American poetry, so feeding it Rumi is a bit out of left field.
# * Rumi, of course, has a context and thread running through his lines while the AI (since it was fed only that one line) doesn't. 
# 
# <p>
# "Spreading light like a hurricane eclipse" is a metaphor I won't soon forget. And it was created by a machine learning model!

# ## Serving poetry
# 
# How would you serve these predictions? There are two ways:
# <ol>
# <li> Use [Cloud ML Engine](https://cloud.google.com/ml-engine/docs/deploying-models) -- this is serverless and you don't have to manage any infrastructure.
# <li> Use [Kubeflow](https://github.com/kubeflow/kubeflow/blob/master/user_guide.md) on Google Kubernetes Engine -- this uses clusters but will also work on-prem on your own Kubernetes cluster.
# </ol>
# <p>
# In either case, you need to export the model first and have TensorFlow serving serve the model. The model, however, expects to see *encoded* (i.e. preprocessed) data. So, we'll do that in the Python Flask application (in AppEngine Flex) that serves the user interface.

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'TOPDIR=gs://${BUCKET}\nOUTDIR=${TOPDIR}/poetry/model_full2\nDATADIR=${TOPDIR}/poetry/data\nMODEL=transformer\nHPARAMS=transformer_poetry\nBEAM_SIZE=4\nALPHA=0.6\n\nt2t-exporter \\\n  --model=$MODEL \\\n  --hparams_set=$HPARAMS \\\n  --problem=$PROBLEM \\\n  --t2t_usr_dir=./poetry/trainer \\\n  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \\\n  --data_dir=$DATADIR \\\n  --output_dir=$OUTDIR')


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'MODEL_LOCATION=$(gsutil ls gs://${BUCKET}/poetry/model_full2/export/Servo | tail -1)\necho $MODEL_LOCATION\nsaved_model_cli show --dir $MODEL_LOCATION --tag_set serve --signature_def serving_default')


# #### Cloud ML Engine

# In[ ]:


get_ipython().run_cell_magic('writefile', 'mlengine.json', "description: Poetry service on ML Engine\nautoScaling:\n    minNodes: 1  # We don't want this model to autoscale down to zero")


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'MODEL_NAME="poetry"\nMODEL_VERSION="v1"\nMODEL_LOCATION=$(gsutil ls gs://${BUCKET}/poetry/model_full2/export/Servo | tail -1)\necho "Deleting and deploying $MODEL_NAME $MODEL_VERSION from $MODEL_LOCATION ... this will take a few minutes"\ngcloud ml-engine versions delete ${MODEL_VERSION} --model ${MODEL_NAME}\n#gcloud ml-engine models delete ${MODEL_NAME}\n#gcloud ml-engine models create ${MODEL_NAME} --regions $REGION\ngcloud alpha ml-engine versions create --machine-type=mls1-highcpu-4 ${MODEL_VERSION} \\\n       --model ${MODEL_NAME} --origin ${MODEL_LOCATION} --runtime-version=1.5 --config=mlengine.json')


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'gcloud components update --quiet\ngcloud components install alpha --quiet')


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'MODEL_NAME="poetry"\nMODEL_VERSION="v1"\nMODEL_LOCATION=$(gsutil ls gs://${BUCKET}/poetry/model_full2/export/Servo | tail -1)\ngcloud alpha ml-engine versions create --machine-type=mls1-highcpu-4 ${MODEL_VERSION} \\\n       --model ${MODEL_NAME} --origin ${MODEL_LOCATION} --runtime-version=1.5 --config=mlengine.json')


# #### Kubeflow
# 
# Follow these instructions:
# * On the GCP console, launch a Google Kubernetes Engine (GKE) cluster named 'poetry' with 2 nodes, each of which is a n1-standard-2 (2 vCPUs, 7.5 GB memory) VM
# * On the GCP console, click on the Connect button for your cluster, and choose the CloudShell option
# * In CloudShell, run: 
#     ```
#     git clone https://github.com/GoogleCloudPlatform/training-data-analyst`
#     cd training-data-analyst/courses/machine_learning/deepdive/09_sequence
#     ```
# * Look at [`./setup_kubeflow.sh`](setup_kubeflow.sh) and modify as appropriate.

# ### AppEngine
# 
# What's deployed in Cloud ML Engine or Kubeflow is only the TensorFlow model. We still need a preprocessing service. That is done using AppEngine.  Edit application/app.yaml appropriately.

# In[ ]:


get_ipython().system('cat application/app.yaml')


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'cd application\n#gcloud app create  # if this is your first app\n#gcloud app deploy --quiet --stop-previous-version app.yaml')


# Now visit https://mlpoetry-dot-cloud-training-demos.appspot.com and try out the prediction app!
# 
# <img src="diagrams/poetry_app.png" width="50%"/>

# Copyright 2018 Google Inc. Licensed under the Apache License, Version 2.0 (the \"License\"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an \"AS IS\" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License