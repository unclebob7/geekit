---
layout: post
title: "matrix differentiation"
date: 2019-02-25 11:19:00
image: 'https://raw.githubusercontent.com/unclebob7/geekit/gh-pages/assets/img/matdiff_entry.png'
description: tutorial on matrix differentiation
category: 'math'
tags:
- math
introduction: tutorial on matrix differentiation
---

> *Reality dosen't exist until we measure it.*
>
> Quantum physics

## How matrix calculus comes into sight ?

Almost 50 years has passed since **backpropagation**[(Paul Werbos 1974)](https://en.wikipedia.org/wiki/Paul_Werbos) was introduced into DNN. As successors of this gloroious legacy from our old men, We have developed countless high-level frameworks like :

|Library|API|Started by|Year|
|-------|---|----------|----|
|Torch|C++, Lua|R.Collobert et al|2002|
|Caffe|Python, C++, Matlab|UC Berkley|2013|
|TensorFlow|Python, JavaScript, C++, Java, etc.|Google|2015|

We have seen tons of TF operations like :

```python
gradients_autodiff = tf.gradients(loss, [W])    # eveything define with tf. is an operation including this...
print(gradients_autodiff)
training_op_autodiff = tf.assign(W, W - learning_rate*gradients_autodiff[0])

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

or even be more abstract and intuitive like this :

```python
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op_optimizer = optimizer.minimize(loss              

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("#Epoch", epoch, ": ", "MSE =", loss.eval())
        sess.run(training_op_optimizer)
        
    best_W = W.eval()
    print("optimal internal model parameters:", best_W)
```

Just to exemplify, Framework like TensorFlow offers us ultimately friendly and enclosed entries like : [autodiff](https://www.tensorflow.org/api_docs/python/tf/gradients),  [optimizer](https://www.tensorflow.org/api_docs/python/tf/train#classes) to perform optimizations.

However, does it really mean our "old-school" calculus were out-moded and should be cast aside simply becuse Google or Facebook take care of everything for us?

> not necessarily...

Start from very fundamental Linear Regression:

![linear regression](https://raw.githubusercontent.com/unclebob7/geekit/gh-pages/assets/img/Screenshot%20from%202019-02-25%2019-03-44.png)

Ultimately easy in `scalar field`, but how about in `vector field` ?

Higher dimensions come with more principles to specify :

- formulation
- type of numerator & denominator

> *Let's roll...* 

## formulation

Since `derivative` consists of `numerator` w.r.t `denominator`, there are two possible layouts(formulation) that we could specify in for our output(**gradient**)

- numerator layout (*Jacobian formulation*)
- denominator layout (*Hessian formulation*)

Basically, It's simply whether the lay out according to `numerator` or `denominator`

And I personally prefer `denominator layout`, consider how it got its name... :grin: :grin: :grin:

details are as follows([wiki](https://en.wikipedia.org/wiki/Matrix_calculus)):

![matdiff table](https://raw.githubusercontent.com/unclebob7/geekit/gh-pages/assets/img/matdiff_table.png)

## type of numerator & denominator

Since we were "forcibly" embraced by our `world of matrix`(*vector filed*), we now have 4 possible combinations out of box.

- scalar w.r.t scalar
- scalar w.r.t vector
- vector w.r.t scalar
- vector w.r.t vector 

We go directly into the toughest... (vector w.r.t vector)

details are as follows : 
![matdiff](https://raw.githubusercontent.com/unclebob7/geekit/gh-pages/assets/img/matdiff.jpg)

## Verification

```python
Y_predict = tf.matmul(X, W, name="prediction")
error = Y_predict - Y
loss = tf.losses.mean_squared_error(Y, Y_predict)
gradients = 2/m*tf.matmul(tf.transpose(X), error)
 
# cannot directly apply "W = W - learning_rate*gradients" since costant and Variable are "source ops" that take no input
# W = W - learning_rate*gradients
training_op = tf.assign(W, W - learning_rate*gradients)   

init = tf.global_variables_initializer()

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

All above snippets output the same **MSE** : 


