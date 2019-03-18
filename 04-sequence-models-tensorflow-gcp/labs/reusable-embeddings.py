
# coding: utf-8

# <h1>Using pre-trained embeddings with TensorFlow Hub</h1>
# 
# This notebook illustrates:
# <ol>
#     <li>How to instantiate a TensorFlow Hub module</li>
#     <li>How to find pre-trained TensorFlow Hub modules for a variety of purposes</li>
#     <li>How to examine the embeddings of a Hub module</li>
#     <li>How one Hub module composes representations of sentences from individual words</li>
#     <li>How to assess word embeddings using a semantic similarity test</li>
# </ol>

# In[14]:


# change these to try this notebook out 
#BUCKET = 'cloud-training-demos-ml'
#PROJECT = 'cloud-training-demos'
#REGION = 'us-central1'


# Install the TensorFlow Hub library

# In[15]:


get_ipython().system('pip install -q tensorflow-hub')


# In[16]:


import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import scipy
import math

#os.environ['BUCKET'] = BUCKET
#os.environ['PROJECT'] = PROJECT
#os.environ['REGION'] = REGION
#os.environ['TFVERSION'] = '1.8'


# In[17]:


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
os.environ['TFVERSION'] = '1.8'

print(PROJECT)
print(BUCKET)
print(REGION)


# In[18]:


import tensorflow as tf
print(tf.__version__)


# <h2>TensorFlow Hub Concepts</h2>
# 
# TensorFlow Hub is a library for the publication, discovery, and consumption of reusable parts of machine learning models. A module is a self-contained piece of a TensorFlow graph, along with its weights and assets, that can be reused across different tasks in a process known as transfer learning, which we covered as part of the course on Image Models.
# 
# To download and use a module, it's as easy as:
import tensorflow as tf
import tensorflow_hub as hub

with tf.Graph().as_default():
  module_url = "path/to/hub/module"
  embed = hub.Module(module_url)
  embeddings = embed(["word1", "word2", "word3"])
  # ...
# However, because modules are self-contained parts of a TensorFlow graph, in order to actually collect values from a module, you'll need to evaluate it in the context of a session.
  # .... earlier code
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    print(sess.run(embeddings))
# First, let's explore what hub modules there are. Go to [the documentation page](https://www.tensorflow.org/hub/modules) and explore a bit.
# 
# Note that TensorFlow Hub has modules for Images, Text, and Other. In this case, we're interested in a Text module, so navigate to the Text section.
# 
# Within the Text section, there are a number of modules. If you click on a link, you'll be taken to a page that describes the module and links to the original paper where the model was proposed. Click on a model in the Word2Vec section of the page.
# 
# Note the details section, which describes what the module expects as input, how it preprocesses data, what it does when it encounters a word it hasn't seen before (OOV means "out of vocabulary") and in this case, how word embeddings can be composed to form sentence embeddings.
# 
# Finally, note the URL of the page. This is the URL you can copy to instantiate your module.

# ### nnlm-en-dim50: [https://tfhub.dev/google/nnlm-en-dim50/1](https://tfhub.dev/google/nnlm-en-dim50/1)
# 
# Token based text embedding trained on English Google News 7B corpus.
# 
# * Overview: Text embedding based on feed-forward Neural-Net Language Models[1] with pre-built OOV. Maps from text to 50-dimensional embedding vectors.
# 
# Example use
# 
# ```
# embed = hub.Module("https://tfhub.dev/google/nnlm-en-dim50/1")
# embeddings = embed(["cat is on the mat", "dog is in the fog"])
# ```
# 
# * Details: Based on NNLM with two hidden layers.
# * Input: The module takes a batch of sentences in a 1-D tensor of strings as input.
# * Preprocessing: The module preprocesses its input by splitting on spaces.
# * Out of vocabulary tokens: Small fraction of the least frequent tokens and embeddings (~2.5%) are replaced by hash buckets. Each hash bucket is initialized using the remaining embedding vectors that hash to the same bucket.
# * Sentence embeddings: Word embeddings are combined into sentence embedding using the sqrtn combiner (see `tf.nn.embedding_lookup_sparse`).
# 
# References
# [1] Yoshua Bengio, Réjean Ducharme, Pascal Vincent, Christian Jauvin. A Neural Probabilistic Language Model. Journal of Machine Learning Research, 3:1137-1155, 2003.
# 
# 

# <h2>Task 1: Create an embedding using the NNLM model</h2>
# 
# To complete this task:
# <ol>
#     <li>Find the module URL for the NNLM 50 dimensional English model</li>
#     <li>Use it to instantiate a module as 'embed'</li>
#     <li>Print the embedded representation of "cat"</li>
# </ol>
# 
# NOTE: downloading hub modules requires downloading a lot of data. Instantiating the module will take a few minutes.

# In[19]:


# Task 1
embed = hub.Module("https://tfhub.dev/google/nnlm-en-dim50/1")
embeddings = embed(["cat"])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    print(sess.run(embeddings))


# When I completed this exercise, I got a vector that looked like:
# [[ 0.11233182 -0.3176392  -0.01661182...]]

# <h2>Task 2: Assess the Embeddings Informally</h2>
# 
# <ol>
#     <li>Identify some words to test</li>
#     <li>Retrieve the embeddings for each word</li>
#     <li>Determine what method to use to compare each pair of embeddings</li>
# </ol>    
# 
# So, now we have some vectors but the question is, are they any good? One way of testing whether they are any good is to try them for your task. But, first, let's just take a peak. 
# 
# For our test, we'll need three common words such that two of the words are much closer in meaning than the third.

# In[24]:


word_1 = "coffee"
word_2 = "tea"
word_3 = "computer"


# Now, we'll use the same process of using our Hub module to generate embeddings but instead of printing the embeddings, capture them in a variable called 'my_embeddings'.

# In[25]:


# Task 2b
embeddings = embed([word_1, word_2, word_3])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    my_embeddings = sess.run(embeddings)

print(my_embeddings)


# Now, we'll use Seaborn's heatmap function to see how the vectors compare to each other. I've written the shell of a function that you'll need to complete that will generate a heatmap. The one piece that's missing is how we'll compare each pair of vectors. Note that because we are computing a score for every pair of vectors, we should have len(my_embeddings)^2 scores. There are many valid ways of comparing vectors. Generality, similarity scores are symmetric. The simplest is to take their dot product. For extra credit, implement a more complicated vector comparison function.

# In[26]:


from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(my_embeddings)


# In[27]:


def plot_similarity(labels, embeddings):
  # TODO (done): fill out a len(embeddings) x len(embeddings) array
  corr = cosine_similarity(embeddings)
  sns.set(font_scale=1.2)
  g = sns.heatmap(
      corr,
      xticklabels=labels,
      yticklabels=labels,
      vmin=0,
      vmax=1,
      cmap="YlOrRd")
  g.set_xticklabels(labels, rotation=90)
  g.set_title("Semantic Textual Similarity")

plot_similarity([word_1, word_2, word_3], my_embeddings)


# What you should observe is that, trivially, all words are identical to themselves, and, more interestingly, that the two more similar words have more similar embeddings than the third word.

# <h2>Task 3: From Words to Sentences</h2>
# 
# Up until now, we've used our module to produce representations of words. But, in fact, if we want to, we can also use it to construct representations of sentences. The methods used by the module to compose a representation of a sentence won't be as nuanced as what an RNN might do, but they are still worth examining because they are so convenient.
# 
# <ol>
#     <li> Examine the documentation for our hub module and determine how to ask it to construct a representation of a sentence</li>
#     <li> Figure out how the module takes word embeddings and uses them to construct sentence embeddings </li>
#     <li> Construct a embeddings of a "cat", "The cat sat on the mat", "dog" and "The cat sat on the dog"  and plot their similarity
# </ol>

# In[33]:


# Task 3
word_list = [
  "cat", 
  "the cat sat on the mat", 
  "dog", 
  "the cat sat on the dog"
]
my_embeddings_tensor = embed(word_list)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    my_embeddings = sess.run(my_embeddings_tensor)

print(my_embeddings[:, :12])
plot_similarity(word_list, my_embeddings)


# Which is cat more similar to, "The cat sat on the mat" or "dog"? Is this desireable?
# 
# Think back to how an RNN scans a sequence and maintains its state. Naive methods of embedding composition (mapping many to one) can't possibly compete with a network trained for this very purpose!

# <h2>Task 4: Assessing the Embeddings Formally</h2>
# Of course, it's great to know that our embeddings match our intuitions to an extent, but it'd be better to have a formal, data-driven measure of the quality of the representation.
# 
# Researchers have
# The [STS Benchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark) provides an intristic evaluation of the degree to which similarity scores computed using sentence embeddings align with human judgements. The benchmark requires systems to return similarity scores for a diverse selection of sentence pairs. Pearson correlation is then used to evaluate the quality of the machine similarity scores against human judgements.

# In[39]:


def load_sts_dataset(filename):
  # Loads a subset of the STS dataset into a DataFrame. In particular both
  # sentences and their human rated similarity score.
  sent_pairs = []
  with tf.gfile.GFile(filename, "r") as f:
    for line in f:
      ts = line.strip().split("\t")
      # (sent_1, sent_2, similarity_score)
      sent_pairs.append((ts[5], ts[6], float(ts[4])))
  return pd.DataFrame(sent_pairs, columns=["sent_1", "sent_2", "sim"])


def download_and_load_sts_data():
  sts_dataset = tf.keras.utils.get_file(
      fname="Stsbenchmark.tar.gz",
      origin="http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz",
      extract=True)

  sts_dev = load_sts_dataset(
      os.path.join(os.path.dirname(sts_dataset), "stsbenchmark", "sts-dev.csv"))
  sts_test = load_sts_dataset(
      os.path.join(
          os.path.dirname(sts_dataset), "stsbenchmark", "sts-test.csv"))

  return sts_dev, sts_test


sts_dev, sts_test = download_and_load_sts_data()


# In[40]:


sts_dev.head()


# <h3>Build the Evaluation Graph</h3>
# 
# Next, we need to build the evaluation graph.

# In[41]:


sts_input1 = tf.placeholder(tf.string, shape=(None))
sts_input2 = tf.placeholder(tf.string, shape=(None))

# For evaluation we use exactly normalized rather than
# approximately normalized.
sts_encode1 = tf.nn.l2_normalize(embed(sts_input1), axis=1)
sts_encode2 = tf.nn.l2_normalize(embed(sts_input2), axis=1)
cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
clip_cosine_similarities = tf.clip_by_value(cosine_similarities, -1.0, 1.0)
sim_scores = 1.0 - tf.acos(clip_cosine_similarities)


# <h3>Evaluate Sentence Embeddings</h3>
# 
# Finally, we need to create a session and run our evaluation.

# In[42]:


sts_data = sts_dev #@param ["sts_dev", "sts_test"] {type:"raw"}


# In[48]:


text_a = sts_data['sent_1'].tolist()
text_b = sts_data['sent_2'].tolist()
dev_scores = sts_data['sim'].tolist()

print(text_a[:5])
print(text_b[:5])
print(dev_scores[:5])

def run_sts_benchmark(session):
  """Returns the similarity scores"""
  emba, embb, scores = session.run(
      [sts_encode1, sts_encode2, sim_scores],
      feed_dict={
          sts_input1: text_a,
          sts_input2: text_b
      })
  return scores


with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  session.run(tf.tables_initializer())
  scores = run_sts_benchmark(session)

pearson_correlation = scipy.stats.pearsonr(scores, dev_scores)
print('\nPearson correlation coefficient = {0}\nExplained variance = {1}\np-value = {2}'.format(
    pearson_correlation[0], pearson_correlation[0]**2, pearson_correlation[1]))


# <h3>Extra Credit</h3>
# 
# For extra credit, re-run this analysis with a different Hub module. Are the results different? If so, how?

# In[54]:


## =============================================================== ##
## define input words
## =============================================================== ##

word_list = [
  "cat", 
  "the cat sat on the mat", 
  "dog", 
  "the cat sat on the dog"
]

## tensorflow hub:
## https://tfhub.dev/


# In[55]:



## =============================================================== ##
## nnlm-en-dim50/1
## =============================================================== ##

embed = hub.Module("https://tfhub.dev/google/nnlm-en-dim50/1")
my_embeddings_tensor = embed(word_list)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    my_embeddings = sess.run(my_embeddings_tensor)

print("shape = ", my_embeddings.shape)
print(my_embeddings[:, :12])
plot_similarity(word_list, my_embeddings)


# In[56]:



## =============================================================== ##
## elmo/2
## =============================================================== ##

embed = hub.Module("https://tfhub.dev/google/elmo/2", trainable = True)
my_embeddings_tensor = embed(word_list, signature = "default", as_dict = True)["default"]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    my_embeddings = sess.run(my_embeddings_tensor)

print("shape = ", my_embeddings.shape)
print(my_embeddings[:, :12])
plot_similarity(word_list, my_embeddings)


# In[ ]:



## =============================================================== ##
## build evaluation graph (again, just to make sure)
## =============================================================== ##

sts_input1 = tf.placeholder(tf.string, shape=(None))
sts_input2 = tf.placeholder(tf.string, shape=(None))

# For evaluation we use exactly normalized rather than
# approximately normalized.

## nnlm-en-dim50/1:
#sts_encode1 = tf.nn.l2_normalize(embed(sts_input1), axis=1)
#sts_encode2 = tf.nn.l2_normalize(embed(sts_input2), axis=1)

## elmo/2:
sts_encode1 = tf.nn.l2_normalize(embed(sts_input1, signature = "default", as_dict = True)["default"], axis = 1)
sts_encode2 = tf.nn.l2_normalize(embed(sts_input2, signature = "default", as_dict = True)["default"], axis = 1)

## compute similarities:
cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
clip_cosine_similarities = tf.clip_by_value(cosine_similarities, -1.0, 1.0)
sim_scores = 1.0 - tf.acos(clip_cosine_similarities)

sts_data = sts_dev #@param ["sts_dev", "sts_test"] {type:"raw"}

text_a = sts_data['sent_1'].tolist()
text_b = sts_data['sent_2'].tolist()
dev_scores = sts_data['sim'].tolist()

## =============================================================== ##
## evaluate the model with sts benchmark
## =============================================================== ##

with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  session.run(tf.tables_initializer())
  scores = run_sts_benchmark(session)

pearson_correlation = scipy.stats.pearsonr(scores, dev_scores)
print('\nPearson correlation coefficient = {0}\nExplained variance = {1}\np-value = {2}'.format(
    pearson_correlation[0], pearson_correlation[0]**2, pearson_correlation[1]))


# <h2>Further Reading</h2>
# 
# We published a [blog post](https://developers.googleblog.com/2018/04/text-embedding-models-contain-bias.html) on how bias can affect text embeddings. It's worth a read!