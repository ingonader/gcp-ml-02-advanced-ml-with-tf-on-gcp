
# coding: utf-8

# # MNIST Image Classification with TensorFlow
# 
# This notebook demonstrates how to implement a simple linear image models on MNIST using Estimator.
# <hr/>
# This <a href="mnist_models.ipynb">companion notebook</a> extends the basic harness of this notebook to a variety of models including DNN, CNN, dropout, pooling etc.

# In[1]:


import numpy as np
import shutil
import os
import tensorflow as tf
print(tf.__version__)


# ## Exploring the data
# 
# Let's download MNIST data and examine the shape. We will need these numbers ...

# In[2]:


from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('mnist/data', one_hot=True, reshape=False)
print(mnist.train.images.shape)
print(mnist.train.labels.shape)


# In[10]:


print(type(mnist))
print(dir(mnist))


# In[3]:


HEIGHT=28
WIDTH=28
NCLASSES=10


# In[4]:


import matplotlib.pyplot as plt
IMGNO=12
plt.imshow(mnist.test.images[IMGNO].reshape(HEIGHT, WIDTH));


# ## Define the model.
# Let's start with a very simple linear classifier. All our models will have this basic interface -- they will take an image and return logits.

# In[15]:


def linear_model(img):
  #TODO (done)
  ## flatten image into tensor of [batch_size, height * width]:
  X = tf.reshape(img, [-1, HEIGHT * WIDTH]) 
  ## get logits for <NCLASSES> units from a dense layer, without activation:
  ylogits = tf.layers.dense(inputs = X, units = NCLASSES, activation = None)
  return ylogits, NCLASSES


# ## Write Input Functions
# 
# As usual, we need to specify input functions for training, evaluation, and predicition.

# In[16]:


train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'image':mnist.train.images},
    y=mnist.train.labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True,
    queue_capacity=5000
  )

#TODO (done):
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
  x = {
    'image': mnist.test.images
  },
  y = mnist.test.labels,
  batch_size = 100,
  num_epochs = 1,
  shuffle = False,
  queue_capacity = 5000  # Integer, size of queue to accumulate.
  )

def serving_input_fn():
    inputs = {'image': tf.placeholder(tf.float32, [None, HEIGHT, WIDTH])}
    features = inputs # as-is
    return tf.estimator.export.ServingInputReceiver(features, inputs)


# ## Write Custom Estimator
# I could have simply used a canned LinearClassifier, but later on, I will want to use different models, and so let's write a custom estimator

# In[30]:


def image_classifier(features, labels, mode, params):
  ylogits, nclasses = linear_model(features['image'])
  probabilities = tf.nn.softmax(ylogits)
  classes = tf.cast(tf.argmax(probabilities, 1), tf.uint8)
  
  if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
    loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits_v2(
        logits = ylogits, 
        labels = labels)
    )
    evalmetrics =  {'accuracy': tf.metrics.accuracy(classes, tf.argmax(labels, 1))}
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(),
                                                 learning_rate=params['learning_rate'], 
                                                 optimizer="Adam")
    else:
      train_op = None
  else:
    loss = None
    train_op = None
    evalmetrics = None
 
  return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"probabilities": probabilities, 
                     "classes": classes},
        loss=loss,
        train_op=train_op,
        eval_metric_ops=evalmetrics,
        export_outputs={'classes': tf.estimator.export.PredictOutput(
          {"probabilities": probabilities, "classes": classes}
        )}
    )


#  tf.estimator.train_and_evaluate does distributed training.

# In[34]:


def train_and_evaluate(output_dir, hparams):
  estimator = tf.estimator.Estimator(model_fn = image_classifier,
                                     params = hparams,
                                     model_dir = output_dir)
  train_spec = tf.estimator.TrainSpec(input_fn = train_input_fn,
                                    max_steps = hparams['train_steps'])
  exporter = tf.estimator.LatestExporter('Servo', serving_input_fn)
  eval_spec = tf.estimator.EvalSpec(input_fn = eval_input_fn,
                                  steps = None,
                                  exporters = exporter)
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
  
  ## added: saving the model manually (shouldn't really be necessary, 
  ## but see https://guillaumegenthial.github.io/serving-tensorflow-estimator.html):
  estimator.export_savedmodel('saved_model', serving_input_fn)
  


# This is the main() function

# In[35]:


OUTDIR='mnist/learned'
shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time

hparams = {'train_steps': 1000, 'learning_rate': 0.01}
train_and_evaluate(OUTDIR, hparams)


# What accuracy did you achieve?
# 
# > INFO:tensorflow:Saving dict for global step 1000: accuracy = 0.9191, global_step = 1000, loss = 0.29975206
# 

# In[37]:


## load model from checkpoint 
## (Estimator API automatically saves models, no need to save explicitly via tf.train.Saver):

## https://guillaumegenthial.github.io/serving-tensorflow-estimator.html
from pathlib import Path

export_dir = 'saved_model'
subdirs = [x for x in Path(export_dir).iterdir()
           if x.is_dir() and 'temp' not in str(x)]
latest = str(sorted(subdirs)[-1])

print(latest)


# In[44]:


from tensorflow.contrib import predictor

predict_fn = predictor.from_saved_model(latest)

pred = predict_fn({'image': mnist.test.images[0:2, :, :, 0]})
print(pred)
print(pred['classes'])


# In[84]:


import pandas as pd
pred = predict_fn({'image': mnist.test.images[:, :, :, 0]})['classes']
print(pred)
print(pred.shape)
print(type(pred))

tmp = pd.DataFrame(pred)
print(tmp[0].values)

## [[here]] -- make pandas data frame from numpy array, somehow.
dat_pred = pd.DataFrame({'labels': mnist.test.labels,
                         'pred' : pd.DataFrame(pred)[0].values})


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