---
layout: post
title: "from Taylor Expansion to Gradient Descent"
date: 2018-10-05 16:37:00
image: '..\assets\img\gd_deduction\taylor_approx.png'
description: from Taylor Expansion to Gradient Descent
category: 'machine learning'
tags:
- machine learning
introduction: from Taylor Expansion to Gradient Descent
---

> This tutorial shows how to derive Gradient Descent from Taylor Expansion

## Background
People which background in Machine Learning (ML) do understand what Gradient Descent (GD) is. Concisely, it is a consistent, iterative algorithm that is used to derive arithmetic solution for a high order optimization problem that does not come with analytical solution. This tutorial focuses on the deduction of GD algorithm from Taylor Expansion theorem (TE).

## Motivation
Given a optimization problem (illustrated as a loss function), we want to iteratively optimize it in order to reach an (global/local) optima. Formulated as follows:

$$min J(\theta)$$

## Methodology

### First-order Taylor Expansion
![Figure 1](..\assets\img\gd_deduction\taylor_approx.png)
<center>Figure 1: First-order Taylor Expansion</center>

As *Figure 1* illustrated, first-order Taylor Expansion is considered as a linear approximation (red arrow) of a smooth curve (black line), it is formulated as follows:

$$J(\theta') = J(\theta) + (\theta' - \theta)\nabla_{\theta}J(\theta)$$
<center>Formula 1: First-order Taylor Expansion</center>

### from TE to GD
to align *Formula 1* with GD that we have previously learned in ML:

$$J(\theta'; \overrightarrow{v}) = J(\theta) + \eta\overrightarrow{v}\nabla_{\theta}J(\theta)$$
$$\eta\overrightarrow{v} = (\theta' - \theta)$$
<center>Formula 2: TE to GD alignment</center>

- **\eta** (scalar): learning rate (ML); length of the directional derivative (TE)
- **\overrightarrow{v}** (unit vector): the direction of `descent`  

**\eta** is just a small scalar that can be self-defined (e.g. 10^-4). So the problem left is to define **\overrightarrow{v}** so that we can maximize the `descent` and reach `optima` as fast as possible. It is formulated as follows:

$$\overrightarrow{v} = arg min_{\overrightarrow{v}} [J(\theta') - J(\theta)] = arg min_{\overrightarrow{v}} \overrightarrow{v}\nabla_{\theta}J(\theta)$$  
<center>Formula 3: Finding v</center>

And it is apparent that in order to maximize the descent of each iteration (finding the smallest negative), **\overrightarrow{v}** got to be the negative direction of **\nabla_{\theta}J(\theta)** (the gradient). It is formulated as follows:

$$\overrightarrow{v} =  - \frac{\nabla_{\theta}J(\theta)}{||\nabla_{\theta}J(\theta)||}$$

Thus that is why `Gradient Descent` is called `Gradient` Descent. (Actually, it would be more appropriate to name is as **negative Gradient Descent**)

## Reference
[1] https://blog.csdn.net/red_stone1/article/details/80212814





