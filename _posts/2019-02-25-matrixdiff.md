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
>  Quantum physics

Almost 50 years has passed since **backpropagation**[(Paul Werbos 1974)](https://en.wikipedia.org/wiki/Paul_Werbos) was introduced into DNN. As successors of this gloroious legacy from our old men, We have developed countless multiple high-level framework like :

|Library|API|Started by|Year|
|-------|---|----------|----|
|Torch|C++, Lua|R.Collobert et al|2002|
|Caffe|Python, C++, Matlab|UC Berkley|2013|
|TensorFlow|Python, JavaScript, C++, Java, etc.|Google|2015|

> Torch (2002)
> Caffe (2013)
> TensorFlow (2015)

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

or even more abstract and intuitive like this :

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