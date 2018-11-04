---
layout: post
title: "SVMs"
date: 2018-10-05 16:37:00
image: 'https://res.cloudinary.com/dn18ydekv/image/upload/v1541319202/blog_svm/entrance.png'
description: tutorial on SVMs
category: 'machine learning'
tags:
- machine learning
introduction: tutorial on SVMs
---

> This tutorial focus on kernel-trick and implementation of SVC with Python. Starting with the algorithm itself !

## Optimization & Constraint
*we are not delving deep into those beginners' graphics, but going directly into the Math*

Since we are drawing a hyper-plane that seperates two possible classifications with labeled-dataset,
our quest essentially falls into the optimizing problem .
We exemplfies with the most fundamental genre **Hard margin linear SVC**

### Hard margin linear SVC classifier objective
![purpose](https://res.cloudinary.com/dn18ydekv/image/upload/v1541319721/blog_svm/oc1.png) 

In respect to the **optimization**: we visibly subject to maximizing the 'streets',
which is the largest margin space between two respective sets of support vectors of each
classification.

Accompanied the **Constraint** :We need the decision function to be greate than 1 for all
positive training instances as well as lower than -1 for all negative counterparts.
*Therefore , we expect all instances positive for the hard-margin model*

### Soft margin linear SVC classifier objective
> We introduce soft-margin SVC in case of overfitting scenario
![purpose](https://res.cloudinary.com/dn18ydekv/image/upload/v1541319723/blog_svm/oc2.png)
Most remarkable feature for *soft-margin* SVC is the very famous **C-parameter**  introduced, which allows certaion
extent of mistakes that SVC gonna tolerate .
Moreover , **slack-variable** is also introduced .

### The dual problem

#### Lagrange multipliers
The general idea is to transform a constrained optimization objective into an unconstrained one , 
by moving the *constraints* into the *objective function* . 
**Basically the same idea with Duality**
for e.g. :
```python
optimization: f(x,y) = x^2 + 2*y
constraints:  3x + y + 1 = 0
```
We simply create a new function named *Lagrangian* or *Lagrange function*
```
g(x,y,α) = f(x,y)-α*(3x + y + 1) 
```

However , this method applies only to *equality contraints* . 
Fortunately , under some regularity conditions(which are repected by the SVM objectives) , this method can be 
generalized to *inequlity constraints* as well . (e.g. : 3x+y+1 > 0)
![purpose](https://res.cloudinary.com/dn18ydekv/image/upload/v1541323714/svc_inequal.png)
- where the `α` variables are called the KKT-multiplier , and they must be greater or equal to 0 . 
- t(.) represents the `tranfer function`(e.g. : sigmoid , Gaussian , etc.) 
- the `inequality` must be hold by `equality` : ````t(i)((w)'*x(i) + b) = 1```
  this condition is called thed `complementary slackness` , implies that either α(i) = 0
  or the i instance lies on the boundary (a *support vector*)
  
*concrete algorithmic flowchat is as follows :*
![purpose](https://res.cloudinary.com/dn18ydekv/image/upload/v1541325735/blog_svm/primal2dual.png)

finally and hopefully , we get the `dual form` of the SVM problem :
![purpose](https://res.cloudinary.com/dn18ydekv/image/upload/v1541319698/blog_svm/dual_form.png)
