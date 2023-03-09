


"""Trains a model, saving checkpoints and tensorboard summaries along
   the way.
   https://github.com/MadryLab/mnist_challenge/blob/master/train.py
   """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import math
from datetime import datetime
import json
import shutil
from timeit import default_timer as timer
from hyperopt import tpe, hp, fmin, space_eval

from termcolor import colored
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import pickle

import sys
version = sys.version_info

classes = list(range(0,10))


data = np.load('/home/anilkag/code/rnn_results/aditya/CIFAR-10/std_ce_64_dim_ft.npz', allow_pickle=True,)
test_X, test_Y = data['test_embd'], data['test_Y']
val_X, val_Y = data['val_embd'], data['val_Y']
train_X, train_Y = data['train_embd'], data['train_Y']

print('shapes x_train, y_train', train_X.shape, train_Y.shape)
print('shapes x_val, y_val', val_X.shape, val_Y.shape)
print('shapes x_test, y_test', test_X.shape, test_Y.shape)
    
print('\n\nnp.unique(train_Y) = ', np.unique(train_Y))




# based on https://github.com/tensorflow/models/tree/master/resnet
class ResnetModel(object):
  """ResNet model."""

  def __init__(self, threshold=0.5, alpha=0.5, mu=0.1 ):
    self.data_type = tf.float32
    self._build_model(threshold=threshold, mu=mu, alpha=alpha)

  def _feed_forward(self, x_input):
    x = x_input

    self.trn_vars = []
    embedding = x
    with tf.variable_scope('logits', reuse=tf.AUTO_REUSE):
        self.pre_softmaxs, w2, b2 = self._fully_connected(x, 10)
        self.trn_vars.extend([w2, b2])
        
    with tf.variable_scope('logit_aux', reuse=tf.AUTO_REUSE):
      pre_softmax_aux, w, b = self._fully_connected(embedding, 10)
      self.trn_vars.extend([w, b])
        
    return pre_softmax_aux, embedding

  def _build_model(self, threshold=0.5, mu=0.1, alpha=0.5):
    with tf.variable_scope('input'):
      self.is_training = tf.placeholder(tf.bool, name='training')
      self.x_input = tf.placeholder(self.data_type,shape=[None, 64])
      self.y_input_aux = tf.placeholder(tf.int64, shape=None)
      self.pre_softmax_aux, self.embedding = self._feed_forward(self.x_input)
    
      self._epsilons = tf.get_variable("epsilons", shape=(10,), 
                                 #initializer=tf.random_normal_initializer(stddev=0.01),
                                #initializer=tf.truncated_normal(stddev=0.01, shape=(10,)),
                                       initializer=tf.constant_initializer(0.5),
                                 constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
      self.all_minimization_vars = tf.trainable_variables()
    
      self._lambdas = tf.get_variable("lambdas", shape=(10,), 
                                 #initializer=tf.random_normal_initializer(stddev=0.01),
                                 #initializer=tf.truncated_normal(stddev=0.01, shape=(10,)),
                                      initializer=tf.constant_initializer(mu - 0.5),
                                 constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
      
    
    #############################
    # AUXILLIARY CROSS ENTROPY LOSS
    self.y_xent_aux = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pre_softmax_aux, labels=self.y_input_aux)
    self.xent_aux = tf.reduce_sum(self.y_xent_aux)
    
    self.predictions_aux = tf.argmax(self.pre_softmax_aux, 1)
    self.correct_prediction_aux = tf.equal(self.predictions_aux, self.y_input_aux)
    self.num_correct_aux = tf.reduce_sum(tf.cast(self.correct_prediction_aux, tf.int64))
    self.accuracy_aux = tf.reduce_mean(tf.cast(self.correct_prediction_aux, tf.float32))
    #############################
    
    self.pre_softmax = self.pre_softmaxs #tf.concat(self.pre_softmaxs, 1)
    print('self.pre_softmax = ', self.pre_softmax)
    
    self.softmax_out = tf.nn.softmax( self.pre_softmax )
    print('softmax_out = ', self.softmax_out)
    
    tol=1e-8
    
    self.binary_prob_xent = 0.0
    self.mean_binary_acc = 0.0
    self.mean_binary_cov = 0.0
    for cls in classes:
        lamda = self._lambdas[cls]
        epsilon = self._epsilons[cls]
                                  
        y_out  = self.softmax_out[:, cls] #tf.nn.sigmoid( self.pre_softmaxs[i] )
        y_pred = tf.greater_equal(y_out, threshold)  #tf.argmax(self.pre_softmax, 1)

        y_out  = tf.reshape( y_out, [-1] )
        y_pred = tf.reshape( y_pred, [-1] )
        print('y_out ', y_out)

        y_input = tf.equal( self.y_input_aux, cls )
        y_input = tf.cast(y_input, tf.float32)
        y_input = tf.reshape( y_input, [-1] )
        print('y_input ', y_input)
    
        n_plus  = tf.reduce_sum( y_input )
        n_minus = tf.reduce_sum( 1-y_input )
        n_total = n_plus + n_minus
    
        #print('y_input * tf.math.log(self.y_out + tol) = ', y_input * tf.math.log(self.y_out + tol))
        #print('(1 - y_input) * tf.math.log(1-self.y_out+tol)', (1 - y_input) * tf.math.log(1-self.y_out+tol) )
    
        eps = 1e-7
        y_out = tf.clip_by_value(y_out, eps, 1-eps)
    
        # Our class is 1 label    
        loss_1 = (1./n_plus) * tf.reduce_sum( -y_input * tf.math.log(y_out + tol) )
        #loss_1 = (1./n_total) * tf.reduce_sum( -tf.math.log(y_out + tol) )
        loss_2 = (1./n_minus) * lamda * tf.reduce_sum(-(1 - y_input) * tf.math.log(1-y_out+tol) )

        #x = tf.reshape( self.pre_softmaxs[cls], [-1] ) 
        #z = y_input
        #y_xent = tf.nn.relu(x) - x * z + tf.math.log(1 + tf.math.exp(-tf.math.abs(x)))
    
        #y_xent = lamda * (1.- z) * x + (z + lamda * (1.-z)) * tf.nn.softplus(-x)
        y_xent = (loss_1 + loss_2) - lamda * epsilon
        self.binary_prob_xent = self.binary_prob_xent + tf.reduce_sum( y_xent )

        #self.xent = alpha * tf.reduce_sum(y_xent) + (1.0-alpha)* self.xent_aux

        y_pred = tf.cast(y_pred, tf.int64)
        correct_prediction = tf.equal(y_pred, tf.cast(y_input, tf.int64))

        coverage = tf.reduce_mean( tf.cast(y_pred, tf.float32) )
        num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        self.mean_binary_acc += accuracy
        self.mean_binary_cov += coverage
        
    self.mean_binary_acc = self.mean_binary_acc / 10.0
    self.mean_binary_cov = self.mean_binary_cov / 10.0
    
    print('\n\n\n Building model with alpha = ', alpha)
    self.binary_prob_xent = self.binary_prob_xent + mu * (tf.reduce_sum(self._epsilons) - 0.1)
    self.xent = alpha * self.binary_prob_xent + (1.0-alpha) * 0.125 * self.xent_aux
    self.lambda_opt_xent = - self.binary_prob_xent
    print('self.xent = ', self.xent)

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    num_non_batch_dimensions = len(x.shape)
    prod_non_batch_dimensions = 1
    for ii in range(num_non_batch_dimensions - 1):
      prod_non_batch_dimensions *= int(x.shape[ii + 1])
    x = tf.reshape(x, [tf.shape(x)[0], -1])
    w = tf.get_variable('ffDW', [prod_non_batch_dimensions, out_dim],
        self.data_type, initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable('biases', [out_dim], self.data_type, initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b), w, b

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])



#with open('config.json') as config_file:
#    config = json.load(config_file)

config = {
  "_comment": "===== MODEL CONFIGURATION =====",
  "model_dir": "models/cifar_multi_class_lwa_64dim_lambda_opt_minus_one_mu",

  "_comment": "===== TRAINING CONFIGURATION =====",
  "tf_random_seed": 451760341,
  "np_random_seed": 216105420,
  "random_seed": 4557077,
  "max_num_training_steps": 100000, #100000, #30000,
  "num_output_steps": 1000,
  "num_summary_steps": 1000,
  "num_checkpoint_steps": 1000,
  "training_batch_size": 200, #50,

  "_comment": "===== EVAL CONFIGURATION =====",
  "num_eval_examples": 10000,
  "eval_batch_size": 200,
  "eval_checkpoint_steps": 3000,
  "eval_on_cpu": True,

  "_comment": "=====ADVERSARIAL EXAMPLES CONFIGURATION=====",
  "epsilon": 0.3,
  "k": 40,
  "a": 0.01,
  "random_start": True,
  "loss_func": "xent",
  "store_adv_path": "attack.npy"
}


import random 


# N = total number of data points
# S (+ or -)
# Coverage : #( f>th ) / N
# Accuracy : #( f>th, y==1 ) / N
# Error    : #( f>th, y==-1 ) / N

def eval_test_adversarial(cls, best_loss, Xtst, ytst, model, sess, saver, model_dir, global_step):
    print('\nEvaluate adversarial test performance at ({})'.format(datetime.now()))
    eval_checkpoint_steps = config['eval_checkpoint_steps']
    num_eval_examples = config['num_eval_examples']
    eval_batch_size = config['eval_batch_size']
    
    #Xtst, ytst = mnist.test.images, mnist.test.labels
    num_eval_examples = len(ytst)
    assert( Xtst.shape[0] == num_eval_examples )

    # Iterate over the samples batch-by-batch
    #assert( num_eval_examples % eval_batch_size == 0 )
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))                 
    aux_acc = 0
    acc = 0
    cov = 0
    loss = 0
    loss_l1 = 0
    loss_l2 = 0
    for ibatch in range(num_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)

        x_batch = Xtst[bstart:bend, :]
        y_batch_aux = ytst[bstart:bend]
        
        dict_nat = {model.x_input: x_batch, model.y_input_aux: y_batch_aux, 
                    model.is_training:False}
        cur_cov, cur_aux_acc, cur_acc, cur_xent, cur_l1, cur_l2 = sess.run([
            model.mean_binary_cov, 
            model.accuracy_aux,
            model.mean_binary_acc, 
            model.xent, 
            model.binary_prob_xent, 
            model.xent_aux], feed_dict = dict_nat)

        acc += cur_acc
        cov += cur_cov
        aux_acc += cur_aux_acc
        
        loss += cur_xent
        loss_l1 += cur_l1
        loss_l2 += cur_l2

    aux_acc /= num_batches
    acc /= num_batches
    cov /= num_batches
    loss /= num_batches
    loss_l1 /= num_batches
    loss_l2 /= num_batches

    if best_loss > loss : 
        print('\n\nSaving the new trained checkpoint..')
        best_loss = loss
        saver.save(sess, os.path.join(model_dir, 'checkpoint'), global_step=global_step)
    
    #saver.save(sess, os.path.join(model_dir, 'checkpoint'), global_step=global_step)

    print('   test==> aux-accuracy={:.2f}%, accuracy={:.2f}%, coverage={:.4}, loss={:.4}, best-loss={:.4}, binary_prob_xent={:.4}, xent_aux={:.4},'.
          format(100 * aux_acc, 100 * acc, cov, loss, best_loss, loss_l1, loss_l2))
    print('  Finished Evaluating adversarial test performance at ({})'.format(datetime.now()))
    return best_loss
    
def evaluate_one_data_batch(cls, b, B, train_X, train_Y, batch_size, sess, model, best_loss, ii):
    # Output to stdout
    idx = random.randint(0,B-1)
    x_batch = train_X[idx*batch_size: (idx+1)*batch_size]
    y_batch_aux = train_Y[idx*batch_size: (idx+1)*batch_size]

    nat_dict = {model.x_input: x_batch, model.y_input_aux: y_batch_aux,
                model.is_training:False}
    
    cov, aux_acc, acc, xent, l1, l2 = sess.run([
            model.mean_binary_cov, 
            model.accuracy_aux,
            model.mean_binary_acc, 
            model.xent, 
            model.binary_prob_xent, 
            model.xent_aux], feed_dict = nat_dict)
    
    print('  Batch {}({}/{}):    ({})'.format(ii, b, B, datetime.now()))
    print('    training==> aux-accuracy={:.2f}%, accuracy={:.4}%, xent={:.4}, binary_prob_xent={:.4},xent_aux={:.4}, coverage={:.4}'.
          format(aux_acc*100, acc*100,  xent, l1, l2, cov))
    print('    best test loss: {:.2f}'.format(best_loss))
    
def change_labels_to_odd_even(y):
    print('y ', y.shape)
    odd_idx = y % 2 != 0
    even_idx =  y % 2 == 0
    
    new_y  = np.empty_like (y)
    new_y[:] = y
    
    new_y[ even_idx ] = 0
    new_y[ odd_idx ] = 1
    return new_y



from copy import deepcopy
def get_model_dir_name(cls, mu, alpha, backbone=False):
    if backbone:
        model_dir = config['model_dir'] + '_backbone_one_sided_formulation(a='+ str(alpha)+',cls=' + str(cls) + ',mu=' + str(mu) + ')'
    else:
        model_dir = config['model_dir'] + '_one_sided_formulation(a='+ str(alpha)+',cls=' + str(cls) + ',mu=' + str(mu) + ')'
    return model_dir

def get_coverage_error_accuracy_for_model_pairs_conditional(y, y0, y1):
    # Given two model predictions, find out coverage, error, accuracy
    
    n_examples   = len(y)
    n_rejections = np.sum( y0 == y1 )
    n_class_zero = np.sum( (y0==1) * (y1==0) ) 
    n_class_one  = np.sum( (y0==0) * (y1==1) ) 
    
    print('n_rejections = ', n_rejections)
    print('n_class_zero = ', n_class_zero)
    print('n_class_one = ', n_class_one)
    assert((n_rejections + n_class_zero + n_class_one) == n_examples)
    
    coverage = 1.0 - (n_rejections / n_examples)
    
    n_correct = np.sum( y[(y0==1) * (y1==0)] == 0 ) + np.sum( y[(y0==0) * (y1==1)] == 1 )
    accuracy  = n_correct / (n_class_zero + n_class_one)
    
    error = 1.0 - accuracy
    
    print('coverage = ', coverage)
    print('accuracy = ', accuracy)
    print('error = ', error)
    
    return coverage, accuracy, error

def get_coverage_error_accuracy_for_model_pairs_aditya(y, y0, y1):
    # Given two model predictions, find out coverage, error, accuracy
    
    n_examples   = len(y)
    n_rejections = np.sum( y0 == y1 )
    n_class_zero = np.sum( (y0==1) * (y1==0) ) 
    n_class_one  = np.sum( (y0==0) * (y1==1) ) 
    
    #print('n_rejections = ', n_rejections)
    #print('n_class_zero = ', n_class_zero)
    #print('n_class_one = ', n_class_one)
    
    coverage = 1.0 - (n_rejections / n_examples)
    
    n_correct = np.sum( y[(y0==1) * (y1==0)] == 0 ) + np.sum( y[(y0==0) * (y1==1)] == 1 )
    accuracy  = n_correct / (n_examples)
    
    error = coverage - accuracy
    
    #print('coverage = ', coverage)
    #print('accuracy = ', accuracy)
    #print('error = ', error)
    
    return coverage, accuracy, error, n_rejections, n_class_zero, n_class_one

def run_model_on_data(Xtst, ytst, cls, _th, mu, alpha):
    eval_batch_size = config['eval_batch_size']
    
    num_eval_examples = len(ytst)
    assert( Xtst.shape[0] == num_eval_examples )

    # Iterate over the samples batch-by-batch
    assert( num_eval_examples % eval_batch_size == 0 )
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size)) 
    
    y_scores = np.zeros( (num_eval_examples, 10), dtype=np.float32 )
    #print('\n\ncls={}, _lambda={}, _threshold={}'.format(cls, _lambda, _th))
    #print('\nEvaluate adversarial test performance at ({})'.format(datetime.now()))
    
    # Load the graph 
    model_dir = get_model_dir_name(cls, mu, alpha)
    tf.reset_default_graph()
    tf.set_random_seed(config['tf_random_seed'])
    np.random.seed(config['np_random_seed'])

    model = ResnetModel(threshold=_th, mu=mu, alpha=alpha)

    print('model dir = ', model_dir)
    ckpt = tf.train.latest_checkpoint(model_dir)
    saver = tf.train.Saver(max_to_keep=3)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt)

        acc = 0
        cov = 0
        for ibatch in range(num_batches):
            bstart = ibatch * eval_batch_size
            bend = min(bstart + eval_batch_size, num_eval_examples)

            x_batch = Xtst[bstart:bend, :]
            y_batch = ytst[bstart:bend]
            dict_nat = {model.x_input: x_batch, model.y_input_aux: y_batch, model.is_training:False}

            raw_sigmoid_scores, cur_cov, cur_acc = sess.run([model.softmax_out, 
                model.mean_binary_cov, model.mean_binary_acc], feed_dict = dict_nat)
            
            y_scores[ bstart:bend ] = raw_sigmoid_scores

            acc += cur_acc
            cov += cur_cov

        acc /= num_batches
        cov /= num_batches

        #print('    test accuracy={:.2f}%, coverage={:.4}'.format(100 * acc, cov))
        #print('  Finished Evaluating adversarial test performance at ({})'.format(datetime.now()))
    return y_scores


def get_coverage_error_for_given_parameters_obj( _predictions, params, y_true ):
    n_examples = len(y_true)    
    belongs_to = np.zeros( (n_examples, ), dtype=np.int32 )
    
    #_lambda_idx, _threshold_idx = cur_params
    y_pred = np.zeros( (n_examples, ), dtype=np.int32 )
    
    for cls in classes:
        _lambda = params['lambda'+str(cls)]  # lambdas[ _lambda_idx[cls] ]
        _threshold = params['th'+str(cls)] # thresholds[ _threshold_idx[cls] ]
        cur_class_pos = (_predictions[cls][_lambda] >= _threshold) * 1
        
        belongs_to += cur_class_pos
        
        # set the class to be current cls wherever cur_class_pos was true
        y_pred[ cur_class_pos==1 ] = cls
        
    #TODO
    #figure out which ones are rejected (if no one says it belongs to them, or if more than one says it belongs to them)
    
    n_rejections = np.sum( belongs_to != 1 )
    coverage = 1.0 - (n_rejections / n_examples)
    
    accuracy = np.sum( ( belongs_to == 1 ) * (y_true == y_pred) ) / n_examples
    error = coverage - accuracy
    
    return error, coverage

def get_coverage_error_for_given_parameters_pred_max( _predictions, lambdas, thresholds, cur_params, y_true ):
    n_examples = len(y_true)    
    belongs_to = np.zeros( (n_examples, ), dtype=np.int32 )
    max_score = np.zeros( (n_examples, ), dtype=np.float32 )
    max_class = np.zeros( (n_examples, ), dtype=np.int32 )
    
    _lambda_idx, _threshold_idx = cur_params
    y_pred = np.zeros( (n_examples, ), dtype=np.int32 )
    
    for i in range(n_examples):
        max_score = -1000.0
        max_class = -1
        for cls in classes:
            _lambda = lambdas[ _lambda_idx[cls] ]
            _threshold = thresholds[ _threshold_idx[cls] ]
            if (_predictions[cls][_lambda][i] >= _threshold):
                if max_score < _predictions[cls][_lambda][i]:
                    max_score = _predictions[cls][_lambda][i]
                    max_class =  cls
        
        if max_class == -1:
            belongs_to[i] = 0 #every OSC rejected this example
        else:
            belongs_to[i] = 1
            y_pred[i] = max_class
    '''
    for cls in classes:
        _lambda = lambdas[ _lambda_idx[cls] ]
        _threshold = thresholds[ _threshold_idx[cls] ]
        cur_class_pos = (_predictions[cls][_lambda] >= _threshold) * 1
        
        belongs_to += cur_class_pos
        
        # set the class to be current cls wherever cur_class_pos was true
        y_pred[ cur_class_pos==1 ] = cls
    '''
        
    #TODO
    #figure out which ones are rejected (if no one says it belongs to them, or if more than one says it belongs to them)
    
    n_rejections = np.sum( belongs_to != 1 )
    coverage = 1.0 - (n_rejections / n_examples)
    
    accuracy = np.sum( ( belongs_to == 1 ) * (y_true == y_pred) ) / n_examples
    error = coverage - accuracy
    
    return error, coverage
    
def get_coverage_error_for_given_parameters( _predictions, lambdas, thresholds, cur_params, y_true ):
    n_examples = len(y_true)    
    belongs_to = np.zeros( (n_examples, ), dtype=np.int32 )
    
    _lambda_idx, _threshold_idx = cur_params
    y_pred = np.zeros( (n_examples, ), dtype=np.int32 )
    
    for cls in classes:
        _lambda = lambdas[ _lambda_idx[cls] ]
        _threshold = thresholds[ _threshold_idx[cls] ]
        cur_class_pos = (_predictions[cls][_lambda] >= _threshold) * 1
        
        belongs_to += cur_class_pos
        
        # set the class to be current cls wherever cur_class_pos was true
        y_pred[ cur_class_pos==1 ] = cls
        
    #TODO
    #figure out which ones are rejected (if no one says it belongs to them, or if more than one says it belongs to them)
    
    n_rejections = np.sum( belongs_to != 1 )
    coverage = 1.0 - (n_rejections / n_examples)
    
    accuracy = np.sum( ( belongs_to == 1 ) * (y_true == y_pred) ) / n_examples
    error = coverage - accuracy
    
    return error, coverage

def get_coverage_error_for_given_parameters_per_class( _predictions, cls, _lambda, _threshold, y_true ):
    n_examples = len(y_true)    
    belongs_to = np.zeros( (n_examples, ), dtype=np.int32 )
    
    y_pred = -1 * np.ones( (n_examples, ), dtype=np.int32 )
    
    cur_class_pos = (_predictions[cls][_lambda] >= _threshold) * 1
    belongs_to += cur_class_pos
        
    # set the class to be current cls wherever cur_class_pos was true
    y_pred[ cur_class_pos==1 ] = cls

    n_rejections = np.sum( belongs_to != 1 )
    coverage = 1.0 - (n_rejections / n_examples)
    
    accuracy = np.sum( ( belongs_to == 1 ) * (y_true == y_pred) ) / n_examples
    error = coverage - accuracy
    
    return error, coverage

def get_predictions_normalized(lambdas, _predictions):
    for _lambda in lambdas:
        sum_scores = 0.0 * _predictions[0][_lambda]
        for cls in classes:
            sum_scores += _predictions[cls][_lambda]
        
        for cls in classes:
            _predictions[cls][_lambda] /= sum_scores

def gather_all_predictions(val_X, test_X, alpha, lambdas):
    # Gather all predictions
    _predictions = {}
    for cls in classes:
        _predictions[cls] = {}
       
    for _lambda in lambdas:
        scores = run_model_on_data(val_X, val_Y, 1, 0.5, _lambda, alpha)
        #scores = run_model_on_data(test_X, test_Y, 1, 0.5, _lambda, alpha)
        for cls in classes:
            _predictions[cls][_lambda] = scores[:,cls]
                            
    _test_predictions = {}
    for cls in classes:
        _test_predictions[cls] = {}
        
    for _lambda in lambdas:
        scores = run_model_on_data(test_X, test_Y, 1, 0.5, _lambda, alpha)
        for cls in classes:
            _test_predictions[cls][_lambda] = scores[:,cls]
            
    #get_predictions_normalized(lambdas, _predictions)
    #get_predictions_normalized(lambdas, _test_predictions)
    return _predictions, _test_predictions

def post_processing_mix_match_one_sided_models_hyperopt(lambdas = [1.0], thresholds = [0.5],
        desired_errors = [0.01, 0.02], alpha=0.5):
    print('\n\n Mixing multiple one sided models...')
    
    lambdas = sorted(lambdas)
    thresholds = sorted(thresholds)
    print('lambdas = ', lambdas)
    print('thresholds = ', thresholds)
    
    _predictions, _test_predictions = gather_all_predictions(val_X, test_X, alpha, lambdas)
                
    # Will mix-match now on the validation set
    print('\n\nResults = ')
    
    '''
    - [DONE] Sort lambdas, thresholds
    - [DONE] Pick initial set of parameters (lambda_1, ..., lambda_10, threshold_1, ..., threshold_10)
    - [DONE] Find out the performance for this set of params (coverage, error)
    - [DONE] Navigate to its one neighbours and find out their performance, pick the one with the highest coverage for given error
    - Do this randomized start couple of times
    '''
    target_error = 0.01
    y = val_Y
    #y = test_Y
    
    space = {}
    for cls in classes:
        space['th'+str(cls)] = hp.choice('th'+str(cls), thresholds)
        space['lambda'+str(cls)] = hp.choice('lambda'+str(cls), lambdas)
    
    for error in desired_errors:
        best_coverage = 0.0
        best_params   = None
        
        def objective(params):
            cur_error, cur_coverage = get_coverage_error_for_given_parameters_obj( _predictions, params, y )
            if cur_error <= error:
                return -cur_coverage
            else:
                return 1000.0
        
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=20000
        )
        print(best)
        best_params = space_eval(space, best)
        
        if best_params is not None:
            # Lets evaluate the performance on test data with these parameters
            best_error, best_coverage = get_coverage_error_for_given_parameters_obj( _predictions, best_params, y )
            test_error, test_coverage = get_coverage_error_for_given_parameters_obj( _test_predictions, best_params, test_Y )
            

            print('\n\nFor desired_error=', error, " -> best coverage=", best_coverage, ' at error=', best_error,
                  ' with params=', best_params)
            print('For desired_error={:.4}, ==> test cov={:.4}, err=={:.4}'.format(error, test_coverage, test_error) )
        else:
            print('For desired_error={:.4}, could not find any parameters'.format(error))
    
    

def post_processing_mix_match_one_sided_models_same_lambda_th(lambdas = [1.0], thresholds = [0.5],
        desired_errors = [0.01, 0.02], alpha=0.5):
    print('\n\n Mixing multiple one sided models...')
    
    lambdas = sorted(lambdas)
    thresholds = sorted(thresholds)
    print('lambdas = ', lambdas)
    print('thresholds = ', thresholds)
    
    _predictions, _test_predictions = gather_all_predictions(val_X, test_X, alpha, lambdas)
                
    # Will mix-match now on the validation set
    print('\n\nResults = ')
    
    
    #- [DONE] Sort lambdas, thresholds
    #- [DONE] Pick initial set of parameters (lambda_1, ..., lambda_10, threshold_1, ..., threshold_10)
    #- [DONE] Find out the performance for this set of params (coverage, error)
    #- [DONE] Navigate to its one neighbours and find out their performance, pick the one with the highest coverage for given error
    #- Do this randomized start couple of times
    
    #thresholds = np.unique(_predictions[classes[0]][lambdas[-1]])[::10]
    print('thresholds = ', thresholds)
    
    n_lambdas    = len(lambdas)
    n_thresholds = len(thresholds)
    _lambda_idx  = np.random.randint( n_lambdas, size=10 )
    _threshold_idx = np.random.randint( n_thresholds, size=10 )
    
    max_search_depth = 20000
    y = val_Y
    #y = test_Y
    
    for error in desired_errors:
        best_coverage = 0.0
        best_params   = None
        
        for _lidx in range(n_lambdas):
            for _tidx in range(n_thresholds):
                _lambda_idx[:] = _lidx
                _threshold_idx[:] = _tidx
        
                cur_params = (_lambda_idx, _threshold_idx)
            
                cur_error, cur_coverage = get_coverage_error_for_given_parameters_pred_max( _predictions, lambdas, thresholds, cur_params, y )
                #cur_error, cur_coverage = get_coverage_error_for_given_parameters( _predictions, lambdas, thresholds, cur_params, y )
                #print('cur_error=', cur_error, ' --> cur_coverage=', cur_coverage)
                if (cur_error <= error) and (cur_coverage > best_coverage):
                    #print('cur_error=', cur_error, ' --> better  cur_coverage=', cur_coverage, ' parmas=', cur_params)
                    best_coverage, best_params = cur_coverage, deepcopy(cur_params)
        
        if best_params is not None:
            # Lets evaluate the performance on test data with these parameters
            _lambda_idx, _threshold_idx = best_params
            #test_error, test_coverage = get_coverage_error_for_given_parameters( _test_predictions, lambdas, thresholds, best_params, test_Y )
            test_error, test_coverage = get_coverage_error_for_given_parameters_pred_max( _test_predictions, lambdas, thresholds, best_params, test_Y )

            print('\n\nFor desired_error=', error, " -> best coverage=", best_coverage, ' with params=', best_params)
            print('For desired_error={:.4}, ==> test cov={:.4}, err=={:.4}'.format(error, test_coverage, test_error) )
        else:
            print('For desired_error={:.4}, could not find any parameters'.format(error))
        
        
def post_processing_mix_match_one_sided_models(lambdas = [1.0], thresholds = [0.5],
        desired_errors = [0.01, 0.02], alpha=0.5):
    print('\n\n Mixing multiple one sided models...')
    
    lambdas = sorted(lambdas)
    thresholds = sorted(thresholds)
    print('lambdas = ', lambdas)
    print('thresholds = ', thresholds)
    
    _predictions, _test_predictions = gather_all_predictions(val_X, test_X, alpha, lambdas)
                
    # Will mix-match now on the validation set
    print('\n\nResults = ')
    
    
    #- [DONE] Sort lambdas, thresholds
    #- [DONE] Pick initial set of parameters (lambda_1, ..., lambda_10, threshold_1, ..., threshold_10)
    #- [DONE] Find out the performance for this set of params (coverage, error)
    #- [DONE] Navigate to its one neighbours and find out their performance, pick the one with the highest coverage for given error
    #- Do this randomized start couple of times
    
    
    n_lambdas    = len(lambdas)
    n_thresholds = len(thresholds)
    _lambda_idx  = np.random.randint( n_lambdas, size=10 )
    _threshold_idx = np.random.randint( n_thresholds, size=10 )
    
    max_search_depth = 20000
    y = val_Y
    #y = test_Y
    
    for error in desired_errors:
        best_coverage = 0.0
        best_params   = None
        
        cur_params = (_lambda_idx, _threshold_idx)
        cur_error, cur_coverage = get_coverage_error_for_given_parameters( _predictions, lambdas, thresholds, cur_params, y )
        #print('cur_error=', cur_error, ' --> cur_coverage=', cur_coverage)
        if (cur_error <= error) and (cur_coverage > best_coverage):
            best_coverage, best_params = cur_coverage, deepcopy(cur_params)
            
        for d in range(max_search_depth):
            # find out the performance of all 1-neighbours of this set of variables
            updated_best_params = False
            
            for idx_l in range( 10 ):
                new_lambda_idx = np.array( _lambda_idx )
                new_threshold_idx = np.array( _threshold_idx )
                    
                cur_idx = new_lambda_idx[idx_l]
                for dx in [-1, +1]:
                    if (cur_idx + dx <0) or (cur_idx + dx >=n_lambdas): continue
                    
                    new_lambda_idx[idx_l] = cur_idx + dx
                    
                    cur_params = (new_lambda_idx, new_threshold_idx)
                    cur_error, cur_coverage = get_coverage_error_for_given_parameters( _predictions, lambdas, thresholds, cur_params, y )
                    #print('cur_error=', cur_error, ' --> cur_coverage=', cur_coverage)
                    if (cur_error <= error) and (cur_coverage > best_coverage):
                        best_coverage, best_params = cur_coverage, deepcopy(cur_params)
                        updated_best_params = True
                    
            for idx_l in range( 10 ):
                new_lambda_idx = np.array( _lambda_idx )
                new_threshold_idx = np.array( _threshold_idx )
                    
                cur_idx = new_threshold_idx[idx_l]
                for dx in [-1, +1]:
                    if (cur_idx + dx <0) or (cur_idx + dx >=n_thresholds): continue
                    
                    new_threshold_idx[idx_l] = cur_idx + dx
                    
                    cur_params = (new_lambda_idx, new_threshold_idx)
                    cur_error, cur_coverage = get_coverage_error_for_given_parameters( _predictions, lambdas, thresholds, cur_params, y )
                    #print('cur_error=', cur_error, ' --> cur_coverage=', cur_coverage)
                    if (cur_error <= error) and (cur_coverage > best_coverage):
                        best_coverage, best_params = cur_coverage, deepcopy(cur_params)           
                        updated_best_params = True
                        
            if updated_best_params == False:
                _lambda_idx  = np.random.randint( n_lambdas, size=10 )
                _threshold_idx = np.random.randint( n_thresholds, size=10 )
                #print('Could not find anything here.. will restart from random place..')
            else:
                _lambda_idx, _threshold_idx = best_params
        
        if best_params is not None:
            # Lets evaluate the performance on test data with these parameters
            _lambda_idx, _threshold_idx = best_params
            test_error, test_coverage = get_coverage_error_for_given_parameters( _test_predictions, lambdas, thresholds, best_params, test_Y )

            print('\n\nFor desired_error=', error, " -> best coverage=", best_coverage, ' with params=', best_params)
            print('For desired_error={:.4}, ==> test cov={:.4}, err=={:.4}'.format(error, test_coverage, test_error) )
        else:
            print('For desired_error={:.4}, could not find any parameters'.format(error))
        
        

def train_model(train_X, train_Y, val_X, val_Y, test_X, test_Y, cls, 
    model_dir, threshold, _lambda, alpha=0.5, max_num_training_steps=21, lr=1e-3, max_lr=1e-5,
    warm_start=True, backbone=False):
    
    print(train_Y.shape)
    print(val_Y.shape)
    print(test_Y.shape)
    
    print('\n\nmodel directory = ', model_dir)
    if not os.path.exists(model_dir): os.makedirs(model_dir)

    tf.reset_default_graph()
    tf.set_random_seed(config['tf_random_seed'])
    np.random.seed(config['np_random_seed'])
    batch_size = config['training_batch_size']

    # Setting up the data and the model
    global_step = tf.contrib.framework.get_or_create_global_step()

    model = ResnetModel(threshold=threshold, mu=_lambda, alpha=alpha)
    max_step = tf.train.AdamOptimizer(max_lr).minimize(model.lambda_opt_xent, var_list=model._lambdas)
    #max_step = tf.train.AdamOptimizer(1e-3).minimize(model.lambda_opt_xent, var_list=model._lambdas)
    
    train_step = tf.train.AdamOptimizer(lr).minimize(model.xent, global_step=global_step, var_list=model.all_minimization_vars)
    #train_step = tf.train.AdamOptimizer(lr).minimize(model.xent, global_step=global_step, var_list=model.trn_vars)
    
    best_saver = tf.train.Saver(max_to_keep=3, var_list=tf.trainable_variables())
    saver = tf.train.Saver(max_to_keep=3)
    with open(model_dir + '/config.json', 'w' ) as f: json.dump( config, f)   

    ckpt = tf.train.latest_checkpoint(model_dir)
    print('\n\nrestore model directory = ', model_dir)

    import time
    start_time = time.time()

    N = len(train_X)
    assert(N % batch_size == 0)
    B = int(N/batch_size)

    backbone_update_freq = 20
    cur_epoch_counter = 0
    early_stop_criterion = 20
    best_loss = +10000.0
    prev_loss = +10000.0
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if warm_start:
            saver.restore(sess, ckpt)
            #best_saver.restore(sess, restore_ckpt)
            
            best_loss = eval_test_adversarial(cls, best_loss, test_X, test_Y, model, sess, saver, model_dir, global_step)
            
            #print('\n\nSaving the new trained checkpoint..')
            #saver.save(sess, os.path.join(model_dir, 'checkpoint'), global_step=global_step)

        for ii in range(max_num_training_steps):            
            for b in range(B): 
                x_batch = train_X[ b*batch_size : (b+1)*batch_size ]
                y_batch_aux = train_Y[ b*batch_size : (b+1)*batch_size ]

                nat_dict = {model.x_input: x_batch, model.y_input_aux: y_batch_aux, 
                            model.is_training:True}
                sess.run(train_step, feed_dict=nat_dict)                
                sess.run(max_step, feed_dict=nat_dict)                
                
                #if (backbone == False) and (b%backbone_update_freq == 0):
                #    sess.run(backbone_train_step, feed_dict=nat_dict)                
                
                if (b % (B-1) == 0):
                    evaluate_one_data_batch(cls, b, B, train_X, train_Y, batch_size, sess, model, best_loss, ii)

            print('\n\nEvaluate adversarial accuracy on test data..', ii)
            prev_loss = best_loss
            best_loss = eval_test_adversarial(cls, best_loss, test_X, test_Y, model, sess, saver, model_dir, global_step)

            print('\nlambdas = ', sess.run(model._lambdas))
            print('\nepsilons = ', sess.run(model._epsilons))
            
            if prev_loss == best_loss:
                cur_epoch_counter += 1 
                if cur_epoch_counter >= early_stop_criterion:
                    print('\nExiting early..')
                    break
            else:
                cur_epoch_counter = 0

        #assert(1 == 2)

    print('took ', int((time.time() - start_time)), 's')
    
    
def train_learning_with_abstention(lambdas = [1.0], threshold = 0.5, max_num_training_steps=21,
            lr=1e-4, max_lr=1e-5, backbone=False, warm_start=True, alpha=0.99):
    print('\n\n Training multiple one sided models...')
    print('mus = ', lambdas)
    
    cls=1
    for _lambda in lambdas:                       
        model_dir = get_model_dir_name(cls, _lambda, alpha, backbone=False)
        
        train_model(train_X, train_Y, val_X, val_Y, test_X, test_Y, cls,
            model_dir, threshold, _lambda, alpha=alpha, max_num_training_steps=max_num_training_steps, lr=lr,
            max_lr=max_lr, backbone=backbone, warm_start=warm_start)




epochs=201 #51
#mus = [0.2] #[0.01] #[0.8] #
#mus = [0.2, 0.05, 0.01]
mus = [0.8]
#mus = [0.2, 3.0]
#mus = np.linspace(0.0001,0.5,20)
#mus = [0.01, 0.05, 0.2] #[0.5, 1.0, 2.0]
#mus = np.linspace(0.1,3,20)
#mus = np.linspace(0.1,0.5,10)
print(mus)

#thresholds = [0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9];
#thresholds =np.linspace(0, 1, num=100)
thresholds = np.hstack([ np.linspace(0, 0.4, num=300), np.linspace(0.4, 1, num=200)])
#thresholds =np.linspace(0, 1, num=1000)
print(thresholds)

alpha= 0.9999 #0.99
desired_errors = [0.005, 0.01, 0.02, 0.10];

#train_learning_with_abstention(lambdas = mus, threshold = 0.5, max_num_training_steps=epochs,
#            lr=1e-4, max_lr=1e-5, warm_start=False, alpha=alpha)
#train_learning_with_abstention(lambdas = mus, threshold = 0.5, max_num_training_steps=epochs,
#            lr=1e-3, max_lr=1e-5, warm_start=False, alpha=alpha)

#train_learning_with_abstention(lambdas = mus, threshold = 0.5, max_num_training_steps=epochs,
#            lr=1e-4, max_lr=1e-4, warm_start=False, alpha=alpha)

#train_learning_with_abstention(lambdas = mus, threshold = 0.5, max_num_training_steps=epochs,
#            lr=1e-5, warm_start=True, alpha=alpha)



train_learning_with_abstention(lambdas = mus, threshold = 0.5, max_num_training_steps=epochs,
            lr=1e-4, max_lr=5e-5, warm_start=False, alpha=alpha)
            #lr=1e-4, max_lr=1e-4, warm_start=False, alpha=alpha)

x = post_processing_mix_match_one_sided_models_same_lambda_th(lambdas = mus, thresholds = thresholds, 
                                                              desired_errors = desired_errors, alpha=alpha)
                                                              
'''
For desired_error= 0.005  -> best coverage= 0.5913999999999999  with params= (array([7, 7, 7, 7, 7, 7, 7, 7, 7, 7]), array([489, 489, 489, 489, 489, 489, 489, 489, 489, 489]))
For desired_error=0.005, ==> test cov=0.5891, err==0.0044


For desired_error= 0.01  -> best coverage= 0.6766  with params= (array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5]), array([476, 476, 476, 476, 476, 476, 476, 476, 476, 476]))
For desired_error=0.01, ==> test cov=0.6787, err==0.0116


For desired_error= 0.02  -> best coverage= 0.773  with params= (array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5]), array([446, 446, 446, 446, 446, 446, 446, 446, 446, 446]))
For desired_error=0.02, ==> test cov=0.7703, err==0.025


For desired_error= 0.1  -> best coverage= 0.9862  with params= (array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), array([276, 276, 276, 276, 276, 276, 276, 276, 276, 276]))
For desired_error=0.1, ==> test cov=0.9853, err==0.1068
'''















