#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os

tf.logging.set_verbosity(tf.logging.INFO)

TIMESERIES_COL = 'height'
N_OUTPUTS = 1  # in each sequence, 1-49 are features, and 50 is label
SEQ_LEN = None
DEFAULTS = None
N_INPUTS = None


def init(hparams):
    global SEQ_LEN, DEFAULTS, N_INPUTS
    SEQ_LEN = hparams['sequence_length']
    DEFAULTS = [[0.0] for x in range(0, SEQ_LEN)]
    N_INPUTS = SEQ_LEN - N_OUTPUTS


def linear_model(features, mode, params):
    #TODO (done): finish linear model
    X = features[TIMESERIES_COL]
    # X = tf.reshape(X, shape=[-1, SEQ_LEN])) ## no need to flatten here, since data is already flat
    X = tf.layers.dense(X, units = 1, activation = None)  ## just one output unit for prediction
    #tf.summary.scalar("loss", loss)
    return X

def dnn_model(features, mode, params):
  #TODO (done): finish DNN model
  X = features[TIMESERIES_COL]

  # ## creating layer using name_scope and adding tensorboard summaries: 
  # ## (https://bitfusion.io/2017/03/30/intro-to-tensorboard/)
  # with tf.name_scope('dense01'):
  #   X = tf.layers.dense(X, units = 20, activation = tf.nn.relu, name = 'dense01')
  #   X_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'dense01')
  #   tf.summary.histogram('kernel', X_vars[0])
  #   tf.summary.histogram('bias', X_vars[1])
  #   tf.summary.histogram('act', X)
  # ## above doesn't work, so do it without that kind of tensorboard summary logging code.
  
  X = tf.layers.dense(X, units = 20, activation = tf.nn.relu, name = 'dense01')
  ## try adding tensorboard information:
  #tf.summary.histogram('dense01', X)
    
  X = tf.layers.dense(X, units = 3, activation = tf.nn.relu)
  # ## add weights to tensorboard: (doesn't work)
  # weights = tf.get_default_graph().get_tensor_by_name(
  #     os.path.split(X.name)[0] + '/kernel:0'
  # )
  #tf.summary.histogram("Layer_2_weights", weights)
  X = tf.layers.dense(X, units = 1, activation = None)

  return X

def cnn_model(features, mode, params):
  #TODO (done): finish CNN model
  ## flatten input:
  net = tf.reshape(features[TIMESERIES_COL], [-1, N_INPUTS, 1]) # as a 1D "sequence" with only one time-series observation (height)

  ## add convolutional and max pooling layers:
  net = tf.layers.conv1d(inputs = net,
                         filters = N_INPUTS // 2,
                         kernel_size = 3,
                         strides = 1,
                         padding = "same",
                         activation = tf.nn.relu)
  net = tf.layers.max_pooling1d(inputs = net,
                                pool_size = 2,
                                strides = 2)
  net = tf.layers.conv1d(inputs = net,
                         filters = N_INPUTS // 2,
                         kernel_size = 3,
                         strides = 1,
                         padding = "same",
                         activation = tf.nn.relu)
  net = tf.layers.max_pooling1d(inputs = net,
                                pool_size = 2,
                                strides = 2)
  ## flatten output:
  outlen = net.shape[1] * net.shape[2]  ## first dimension [0] is batch_size
  net = tf.reshape(net, [-1, outlen])
  ## one hidden layer:
  net = tf.layers.dense(inputs = net, 
                        units = 3, 
                        activation = tf.nn.relu)
  ## outputs for linear regression:
  net = tf.layers.dense(inputs = net,
                        units = 1,
                        activation = None)
                         
  return net

def rnn_model(features, mode, params):
  ## size of the internal state in each of the cells:
  CELL_SIZE = N_INPUTS // 3     
    
  # 1. dynamic_rnn needs 3D shape: [BATCH_SIZE, N_INPUTS, 1]
  x = tf.reshape(features[TIMESERIES_COL], [-1, N_INPUTS, 1])
  
  #TODO (done): finish rnn model
  ## define RNN cells:
  cell = tf.nn.rnn_cell.GRUCell(CELL_SIZE)
  
  ## unroll cells: 
  outputs, state = tf.nn.dynamic_rnn(cell = cell,     ## deprecated; use keras.layers.RNN(cell) instead.
                                     inputs = x,
                                     dtype = tf.float32)
  ## output contains the activations for every single time step.
  ## state contains the activation for the last time step.

  ## pass last activation through a dense layer:
  h1 = tf.layers.dense(state, units = N_INPUTS // 2, activation = tf.nn.relu)
  predictions = tf.layers.dense(h1, units = 1, activation = None)  ## [batch_size, 1]
  return predictions


# 2-layer RNN
def rnn2_model(features, mode, params):
  x = tf.reshape(features[TIMESERIES_COL], [-1, N_INPUTS, 1])
  #TODO (done): finish 2-layer rnn model
  ## define layers:
  cell1 = tf.nn.rnn_cell.GRUCell(N_INPUTS * 2)
  cell2 = tf.nn.rnn_cell.GRUCell(N_INPUTS // 2)
  ## define multi-cell that contains all those layers:
  multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
  
  ## unroll cells: 
  outputs, state = tf.nn.dynamic_rnn(cell = multi_cell, 
                                     inputs = x, 
                                     dtype = tf.float32)
  ## output contains the activations of the final layer for every single time step.
  ## state now contains a list of vectors corresponding to the final state 
  ## of each of the cells (i.e., of each layer)
  print("rnn2_model: outputs: ", outputs)
  print("rnn2_model: state: ", state)
  print("rnn2_model: state[1]: ", state[1])

  ## pass rnn output (state of last layer) to dense layer:
  h1 = tf.layers.dense(state[1], 
                       units = multi_cell.output_size // 2,
                       activation = tf.nn.relu)
  predictions = tf.layers.dense(h1, units = 1, activation = None)  ## one output (regression)
  return predictions

# create N-1 predictions
def rnnN_model(features, mode, params):
    # dynamic_rnn needs 3D shape: [BATCH_SIZE, N_INPUTS, 1]
    x = tf.reshape(features[TIMESERIES_COL], [-1, N_INPUTS, 1])

    # 2. configure the RNN
    cell1 = tf.nn.rnn_cell.GRUCell(N_INPUTS * 2)
    cell2 = tf.nn.rnn_cell.GRUCell(N_INPUTS // 2)
    cells = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
    outputs, state = tf.nn.dynamic_rnn(cells, x, dtype=tf.float32)
    # 'outputs' contains the state of the final layer for every time step
    # not just the last time step (?,N_INPUTS, final cell size)
    
    # 3. pass state for each time step through a DNN, to get a prediction
    # for each time step 
    h1 = tf.layers.dense(outputs, cells.output_size, activation=tf.nn.relu)
    h2 = tf.layers.dense(h1, cells.output_size // 2, activation=tf.nn.relu)
    predictions = tf.layers.dense(h2, 1, activation=None)  # (?, N_INPUTS, 1)
    predictions = tf.reshape(predictions, [-1, N_INPUTS])
    return predictions # return prediction for each time step


# read data and convert to needed format
def read_dataset(filename, mode, batch_size=512):
    def _input_fn():
        def decode_csv(row):
            # row is a string tensor containing the contents of one row
            features = tf.decode_csv(row, record_defaults=DEFAULTS)  # string tensor -> list of 50 rank 0 float tensors
            label = features.pop()  # remove last feature and use as label
            features = tf.stack(features)  # list of rank 0 tensors -> single rank 1 tensor
            return {TIMESERIES_COL: features}, label

        # Create list of file names that match "glob" pattern (i.e. data_file_*.csv)
        dataset = tf.data.Dataset.list_files(filename)
        # Read in data from files
        dataset = dataset.flat_map(tf.data.TextLineDataset)
        # Parse text lines as comma-separated values (CSV)
        dataset = dataset.map(decode_csv)

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None  # loop indefinitely
            dataset = dataset.shuffle(buffer_size=10 * batch_size)
        else:
            num_epochs = 1  # end-of-input after this

        dataset = dataset.repeat(num_epochs).batch(batch_size)
        return dataset.make_one_shot_iterator().get_next()

    return _input_fn


def serving_input_fn():
    feature_placeholders = {
        TIMESERIES_COL: tf.placeholder(tf.float32, [None, N_INPUTS])
    }

    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }
    features[TIMESERIES_COL] = tf.squeeze(features[TIMESERIES_COL], axis=[2])

    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)


def compute_errors(features, labels, predictions):
    labels = tf.expand_dims(labels, -1)  # rank 1 -> rank 2 to match rank of predictions

    if predictions.shape[1] == 1:
        loss = tf.losses.mean_squared_error(labels, predictions)
        rmse = tf.metrics.root_mean_squared_error(labels, predictions)
        ## note: 
        ## tf.metric.* return two values: the metric, and an update_op
        ## rmse, update_op = tf.metrics.*
        ## rmse is the current value, the update_op is a running tally
        ## (https://stackoverflow.com/questions/50120073/tensorflow-metrics-with-custom-estimator)
        return loss, rmse
    else:
        # one prediction for every input in sequence
        # get 1-N of (x + label)
        labelsN = tf.concat([features[TIMESERIES_COL], labels], axis=1)
        labelsN = labelsN[:, 1:]
        # loss is computed from the last 1/3 of the series
        N = (2 * N_INPUTS) // 3
        loss = tf.losses.mean_squared_error(labelsN[:, N:], predictions[:, N:])
        # rmse is computed from last prediction and last label
        lastPred = predictions[:, -1]
        rmse = tf.metrics.root_mean_squared_error(labels, lastPred)
        return loss, rmse

# RMSE when predicting same as last value
def same_as_last_benchmark(features, labels):
    predictions = features[TIMESERIES_COL][:,-1] # last value in input sequence
    return tf.metrics.root_mean_squared_error(labels, predictions)


# create the inference model
def sequence_regressor(features, labels, mode, params):
    # 1. run the appropriate model
    model_functions = {
        'linear': linear_model,
        'dnn': dnn_model,
        'cnn': cnn_model,
        'rnn': rnn_model,
        'rnn2': rnn2_model,
        'rnnN': rnnN_model}
    model_function = model_functions[params['model']]
    predictions = model_function(features, mode, params)

    #global_step = tf.train.get_global_step()
    
    # 2. loss function, training/eval ops
    loss = None
    rmse = [None, None]
    train_op = None
    eval_metric_ops = None
    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        loss, rmse = compute_errors(features, labels, predictions)

        if mode == tf.estimator.ModeKeys.TRAIN:
            # this is needed for batch normalization, but has no effect otherwise
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                # 2b. set up training operation
                train_op = tf.contrib.layers.optimize_loss(
                    loss,
                    tf.train.get_global_step(),
                    learning_rate=params['learning_rate'],
                    optimizer="Adam")
            ## record metrics for training:
            #tf.summary.scalar("RMSE_train_summary", rmse[1])
            #tf.summary.scalar("loss_train_summary", loss)


        # 2c. eval metric
        ## record metrics for evaluation:
        eval_metric_ops = {
            #"Loss": loss,
            "RMSE": rmse,
            "RMSE_same_as_last": same_as_last_benchmark(features, labels),
        }
   
    # 3. Create predictions
    if predictions.shape[1] != 1:
        predictions = predictions[:, -1]  # last predicted value
    predictions_dict = {"predicted": predictions}

    ## note down loss for tensorboard: [[?]]
    #tf.summary.scalar("loss_nth", loss)
    #tf.summary.scalar("rmse_nth", rmse)
    #merged_summary_op = tf.summary.merge_all()

    ## create training hooks:
    #train_hook_list= []
    #train_tensors_log = {'RMSE_train_hook': rmse[1],
    #                     'loss_train_hook': loss,
    #                     'global_step': global_step}
    #train_hook_list.append(
    #    tf.train.LoggingTensorHook(
    #        tensors = train_tensors_log, 
    #        every_n_iter = 100)
    #)

    # 4. return EstimatorSpec
    return tf.estimator.EstimatorSpec(
        mode = mode,
        predictions = predictions_dict,
        loss = loss,
        train_op = train_op,
        #training_hooks = train_hook_list,
        eval_metric_ops = eval_metric_ops,
        #evaluation_hooks = None,
        export_outputs = {
            'predictions': tf.estimator.export.PredictOutput(predictions_dict)
        }
    )


def train_and_evaluate(output_dir, hparams):
    get_train = read_dataset(hparams['train_data_path'],
                             tf.estimator.ModeKeys.TRAIN,
                             hparams['train_batch_size'])
    get_valid = read_dataset(hparams['eval_data_path'],
                             tf.estimator.ModeKeys.EVAL,
                             1000)
    estimator = tf.estimator.Estimator(model_fn=sequence_regressor,
                                       params=hparams,
                                       config=tf.estimator.RunConfig(
                                           save_checkpoints_secs=hparams['min_eval_frequency']#, ## [[here]]--was hard-coded (as 30)
                                           #save_summary_steps=1000  
                                       ),
                                       model_dir=output_dir)
    train_spec = tf.estimator.TrainSpec(input_fn=get_train,
                                        max_steps=hparams['train_steps'])
    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=get_valid,
                                      steps=None,
                                      exporters=exporter,
                                      start_delay_secs=hparams['eval_delay_secs'],   ## [[here]]--was hard-coded (as 10)
                                      throttle_secs=hparams['min_eval_frequency']    ## [[here]]--was hard-coded (as 30)
                                     )
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
