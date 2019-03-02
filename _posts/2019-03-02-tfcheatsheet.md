---
layout: post
title: "TensorFlow cheatsheet"
date: 2019-03-02 16:42:00
image: 'https://raw.githubusercontent.com/unclebob7/tensorflow/master/graph/tf_icon.jpg'
description: TensorFlow cheatsheet
category: 'machine learning'
tags:
- machine learning
introduction: TensorFlow cheatsheet
---

# Tensorflow cheatsheet

## Setup


```python
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# To plot pretty figures
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "tensorflow"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)
```

## Idea of session


```python
import tensorflow as tf
# tf.enable_eager_execution()
import numpy as np
from sklearn.datasets import fetch_california_housing
```


```python
w = tf.constant(2)
w1 = tf.Variable(2)

x = tf.constant(5)
x1 = tf.Variable(5)

y = w*(x**2)
z = w*x+2

y1 = w1*(x1**2)
z1 = w1*x1+2
```


```python
init = tf.global_variables_initializer()    # prepare an init node
```


```python
with tf.Session() as sess:
    init.run()
    result = sess.run(y)
    print("resultvalue: {0}".format(type(result)))
```

    resultvalue: <class 'numpy.int32'>



```python
def constant_eval():
    y.eval()
    z.eval()
```


```python
def variable_eval():
    y1.eval()
    z1.eval()
```


```python
with tf.Session() as sess:
    init.run()
    %timeit constant_eval()    # evaluate y, z twice
    %timeit y_val, z_val = sess.run([y, z])    # evaluate y, z in 1 graph run
    %timeit variable_eval()
    %timeit y1_val, z1_val = sess.run([y1, z1])
```

    213 µs ± 10.8 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    147 µs ± 1.35 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    250 µs ± 5.86 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    173 µs ± 1.56 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)


constant-structure is algorithmically less time-perplexing than Variable-structure

## GD with manual derivatives


```python
housing = fetch_california_housing()
m, n = housing.data.shape
```

always remember to apply feature-scaling (data normalization) before GD


```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
x = np.c_[np.ones((m, 1)), scaled_housing_data]
y = housing.target
w = np.random.randn(n+1, 1)
```

setup nodes


```python
n_epochs = 2000
learning_rate = 0.01
```

source ops


```python
X = tf.constant(x, dtype=tf.float32, name="X")
Y = tf.constant(housing.target.reshape(-1 ,1),dtype=tf.float32, name="Y")
W = tf.Variable(w,dtype=tf.float32, name="W")        # internal model parameters
www = W+2    # www has been implicitly been extened as tf.Variable
```

ops


```python
Y_predict = tf.matmul(X, W, name="prediction")
error = Y_predict - Y
loss = tf.losses.mean_squared_error(Y, Y_predict)
gradients = 2/m*tf.matmul(tf.transpose(X), error)
 
# cannot directly apply "W = W - learning_rate*gradients" since costant and Variable are "source ops" that take no input
# W = W - learning_rate*gradients
training_op = tf.assign(W, W - learning_rate*gradients)   

init = tf.global_variables_initializer()
```


```python
with tf.Session() as sess:
    sess.run(init)
    WWW = sess.run(www)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("#Epoch", epoch, ": ", "MSE =", loss.eval())
        sess.run(training_op)
        
    best_W = W.eval()
    print("optimal internal model parameters:", best_W)
```

    #Epoch 0 :  MSE = 17.5893
    #Epoch 100 :  MSE = 1.06605
    #Epoch 200 :  MSE = 0.727634
    #Epoch 300 :  MSE = 0.668196
    #Epoch 400 :  MSE = 0.629465
    #Epoch 500 :  MSE = 0.601439
    #Epoch 600 :  MSE = 0.581019
    #Epoch 700 :  MSE = 0.566111
    #Epoch 800 :  MSE = 0.555209
    #Epoch 900 :  MSE = 0.547221
    #Epoch 1000 :  MSE = 0.541356
    #Epoch 1100 :  MSE = 0.537039
    #Epoch 1200 :  MSE = 0.533853
    #Epoch 1300 :  MSE = 0.531496
    #Epoch 1400 :  MSE = 0.529745
    #Epoch 1500 :  MSE = 0.528441
    #Epoch 1600 :  MSE = 0.527466
    #Epoch 1700 :  MSE = 0.526733
    #Epoch 1800 :  MSE = 0.526181
    #Epoch 1900 :  MSE = 0.525763
    optimal internal model parameters: [[ 2.06855226]
     [ 0.78806996]
     [ 0.11988165]
     [-0.1703423 ]
     [ 0.21964966]
     [-0.0035526 ]
     [-0.03853594]
     [-0.93066472]
     [-0.89567709]]


## GD with TF reverse-mode autodiff


```python
gradients_autodiff = tf.gradients(loss, [W])    # eveything define with tf. is an operation including this...
print(gradients_autodiff)
training_op_autodiff = tf.assign(W, W - learning_rate*gradients_autodiff[0])   
```


    Traceback (most recent call last):


      File "<ipython-input-5-211ebd772d73>", line 1, in <module>
        gradients_autodiff = tf.gradients(loss, [W])    # eveything define with tf. is an operation including this...


    NameError: name 'loss' is not defined




```python
with tf.Session() as sess:
    sess.run(init)
    result = sess.run(gradients_autodiff)
    print(result)
    
     
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("#Epoch", epoch, ": ", "MSE =", loss.eval())
        sess.run(training_op_autodiff)
        
    best_W = W.eval()
    print("optimal internal model parameters:", best_W)
```

    [array([[-4.94855881],
           [-4.48504353],
           [ 0.71529096],
           [-2.399652  ],
           [-0.89287376],
           [ 0.0587704 ],
           [ 3.94650126],
           [-1.81760597],
           [ 2.41193056]], dtype=float32)]
    #Epoch 0 :  MSE = 17.5893
    #Epoch 100 :  MSE = 1.06605
    #Epoch 200 :  MSE = 0.727634
    #Epoch 300 :  MSE = 0.668196
    #Epoch 400 :  MSE = 0.629465
    #Epoch 500 :  MSE = 0.601439
    #Epoch 600 :  MSE = 0.581019
    #Epoch 700 :  MSE = 0.566111
    #Epoch 800 :  MSE = 0.555209
    #Epoch 900 :  MSE = 0.547221
    #Epoch 1000 :  MSE = 0.541356
    #Epoch 1100 :  MSE = 0.537039
    #Epoch 1200 :  MSE = 0.533853
    #Epoch 1300 :  MSE = 0.531496
    #Epoch 1400 :  MSE = 0.529745
    #Epoch 1500 :  MSE = 0.528441
    #Epoch 1600 :  MSE = 0.527466
    #Epoch 1700 :  MSE = 0.526733
    #Epoch 1800 :  MSE = 0.526181
    #Epoch 1900 :  MSE = 0.525763
    optimal internal model parameters: [[ 2.06855249]
     [ 0.78806996]
     [ 0.11988167]
     [-0.17034236]
     [ 0.21964972]
     [-0.00355259]
     [-0.03853594]
     [-0.93066454]
     [-0.89567691]]


## GD with TF optimizer

to make GD computation even more abstract and intuitive...


```python
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op_optimizer = optimizer.minimize(loss)
```


    Traceback (most recent call last):


      File "<ipython-input-13-143d14dd9062>", line 2, in <module>
        training_op_optimizer = optimizer.minimize(loss)


    NameError: name 'loss' is not defined




```python
with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("#Epoch", epoch, ": ", "MSE =", loss.eval())
        sess.run(training_op_optimizer)
        
    best_W = W.eval()
    print("optimal internal model parameters:", best_W)
```

    #Epoch 0 :  MSE = 17.5893
    #Epoch 100 :  MSE = 1.06605
    #Epoch 200 :  MSE = 0.727634
    #Epoch 300 :  MSE = 0.668196
    #Epoch 400 :  MSE = 0.629465
    #Epoch 500 :  MSE = 0.601439
    #Epoch 600 :  MSE = 0.581019
    #Epoch 700 :  MSE = 0.566111
    #Epoch 800 :  MSE = 0.555209
    #Epoch 900 :  MSE = 0.547221
    #Epoch 1000 :  MSE = 0.541356
    #Epoch 1100 :  MSE = 0.537039
    #Epoch 1200 :  MSE = 0.533853
    #Epoch 1300 :  MSE = 0.531496
    #Epoch 1400 :  MSE = 0.529745
    #Epoch 1500 :  MSE = 0.528441
    #Epoch 1600 :  MSE = 0.527466
    #Epoch 1700 :  MSE = 0.526733
    #Epoch 1800 :  MSE = 0.526181
    #Epoch 1900 :  MSE = 0.525763
    optimal internal model parameters: [[ 2.06855249]
     [ 0.78806996]
     [ 0.11988167]
     [-0.17034236]
     [ 0.21964972]
     [-0.00355259]
     [-0.03853594]
     [-0.93066454]
     [-0.89567691]]


## Mini-batch GD with placeholder


```python
n_epochs = 100
learning_rate = 0.01
```


```python
reset_graph()
```


```python
batch_size = 100
n_batches = int(np.ceil(m / batch_size))    # np.float64-->int
```

specify placeholder for batch stats input


```python
X = tf.placeholder(tf.float64, shape=(None, n+1), name="X")
Y = tf.placeholder(tf.float64, shape=(None, 1), name="Y")
```


```python
np.random.seed(42)
w = np.random.randn(n+1, 1)
W = tf.Variable(w, name="W", dtype=tf.float64)
```


```python
y_pred = tf.matmul(X, W, name="preditions")
```


```python
error = Y - y_pred
```


```python
loss = tf.losses.mean_squared_error(Y, y_pred)
```


```python
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)    # optimize tf.Variable (internal model parameters)
```


```python
init = tf.global_variables_initializer()
```

specify logdir


```python
from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_dir = "tf_log"
logdir = "{}/run-{}/".format(root_dir, now)
print(logdir)
```

    tf_log/run-20190302064137/


attach summary operation at the end of the graph


```python
loss_summary = tf.summary.scalar('MSE', loss)
# write graph info to the logfile (events file)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
```


```python
def fetch_batch(epoch_index, batch_index, batch_size):
    np.random.seed(epoch_index*batch_size+batch_index)
    selected_samples = np.random.randint(m, size=batch_size)
    X_batch = x[selected_samples]
    Y_batch = y[selected_samples].reshape(-1, 1)
    return X_batch, Y_batch
```


```python
from tensorflow_graph_in_jupyter import show_graph
show_graph(tf.get_default_graph())
```


    Traceback (most recent call last):


      File "<ipython-input-27-54250237bfef>", line 1, in <module>
        from tensorflow_graph_in_jupyter import show_graph


    ModuleNotFoundError: No module named 'tensorflow_graph_in_jupyter'




```python
print(error.name)
```

    sub:0

## Visualize the graph with print() & tensorboard

```python
with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(n_epochs):
        for batch_index in range(n_batches): 
            X_batch, Y_batch = fetch_batch(epoch, batch, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, Y: Y_batch})
            step = epoch*n_batches + batch_index
            # for every operation that depneds on training data, should pass in them through feed_dict
            # visualize data with print()
            if epoch in range(1):
                    print("#batch", step, ":", "MSE=", sess.run(loss, feed_dict={X: X_batch, Y: Y_batch}))    
                
            # visualize data with tensorboard
            summary_str = sess.run(loss_summary, feed_dict={X: X_batch, Y: Y_batch})
            file_writer.add_summary(summary_str, step)      
```


```python
file_writer.flush()
file_writer.close() 
```

![mse tb](https://raw.githubusercontent.com/unclebob7/tensorflow/master/graph/mse_tb.png)

## name scopes for categorization


```python
tf.reset_default_graph()
```


```python
X = tf.placeholder(name="X_batch", dtype=tf.float64, shape=(None, 9))
```

define reusable RELU unit


```python
def relu(x, threshold=0.0):
    w_shape = (int(X.shape[1]), 1)
    w = tf.Variable(np.random.rand(w_shape[0], 1), name="weights", dtype=tf.float64)
    b = tf.Variable(0.0, name="bias", dtype=tf.float64)
    z = tf.add(tf.matmul(X, w), b, name="z")
    output = tf.maximum(z, threshold, name="relu_output")
    return output
```


```python
threshold = tf.constant(0.0, name="threshold", dtype=tf.float64)
relus = [relu(X, threshold) for node in range(3)]
output = tf.add_n(relus, name="output")
```


```python
file_writer = tf.summary.FileWriter("relu", tf.get_default_graph())
```


```python
file_writer.flush()
file_writer.close()
```

![relu praph](https://raw.githubusercontent.com/unclebob7/tensorflow/master/graph/relu_tb.png)
