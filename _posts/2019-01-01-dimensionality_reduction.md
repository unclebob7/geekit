---
layout: post
title: 'Dimensionality Reduction'
image: 'https://raw.githubusercontent.com/unclebob7/dimensionality-reduction/master/Karl_Pearson%2C_1912.jpg'
description: the complete explication and practition of the idea of 'DImensionality Reduction' with Python3
category: 'machine learning'
tags:
- machine learning
introduction: tutorial on Dimensionality Reduction
---

> This tutorial focus on the implementation of Dimensionality Reduction. 

## Context
Dated back to the long gone 2018, there has been a Gordian knot keeping baffling and following me into 2019:satisfied::satisfied::satisfied:So this reasonably becomes the very first mission to accomplish(clean up) in my 2019. 

> **So the question is as follows:**
Before we apply **eigen-decomposition** to the covariance matrix in **PCA**, we perform **mean-normalization** to the dataset, which I previously deem it a sterotypical step taken for data-processing to avoid over-fitting.However, things are not that easy.

Before we move onward, I'd like to briefly Sigmamarize eigen-decomposition which I previously used and a more commonly used **factorization** method, the SVD(singular value decomposition).

## eigen-decompositon vs. SVD

> eigen-decomposition : `A = P * D * P^-1`

> SVD : `A = U * Sigma * V^T`

* The vectors in the eigen-decomposition is matrix `P` are not necessarily orthogonalif the original dataset, which refers to matrix `D` isn't **positive semi-definite**.So the change of basis isn't a simple rotation. On the other hand, the vectors in matrices `U` and `V`in the **SVD** are genuinely **orthonormal**, so they do represent rotations(and possibly flips).

* In the SVD, the **nondiagonal** matrices `U` and `V` are not necessarily the inverse of one another.They are usually not related to each other at all. While in eigen-decomposition, the **nondiagonal** matrices `P`and `P` are inverses of each other.

* In the SVD, the entries in the **diagonal** matrix `\Sigma` are all real and non-negative. In the eigen-decomposition,the entries of `D` can be any complex number - negative, positive, imaginary, whatever.

* The SVD always exists for any sort of **rectangular** or **square** matrix, whereas the eigen-decomposition can only exists for square matrices, and even for square matrices sometimes it doesn't exist.

## initialize dataset
```python
# making random output stable
np.random.seed(42)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

# Create an array of the given shape 
# and populate it with random samples from a uniform distribution over [0, 1)
angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
# print('angles: {}\nsize of dataset: {}'.format(angles, angles.shape))

X = np.empty((m, 3))
# print(X)
X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)
```

## PCA using Scikit-Learn
**Scikit-Learn** offers perfectly packed entrance of PCA with the implementation of `SVD` , we can simply incorporate it into our `ML-pipeline`
by using
```python
pca = PCA(n_components = 0.99)
```
when given `x` value ranges (0,1), we are indicating **x%** of the variance would be preserved, while if we pass in variable of integer larger than or equals to 1, we specifies the number of principal components to use.
E.g. *preserving 99.9% variance*

```python
# 2-D
X_decorrelated_DR = pca.fit_transform(X)
X_decorrelated_DR[:] = X_decorrelated_DR[:]*(-1) 
```

![fit_transform](https://raw.githubusercontent.com/unclebob7/dimensionality-reduction/master/999_variance.PNG)

*preserving 99.9% variance*

```python
# preserving 99.9% of the variance
pca = PCA(n_components = 0.999)
# 3-D
X_decorrelated = pca.fit_transform(X)
```

![fit_transform](https://raw.githubusercontent.com/unclebob7/dimensionality-reduction/master/9999variance.PNG)

## constructe hyperplane
With the retrival of a little bit **Analytic Geometry** , we can do this very fluently.

* calculate the `normal vertor` which is the `cross product` of the 2 `principal components`  **step 1**
* use meshgrid to construct a plane  **step 2**
* use `A`(x-x0) + `B`(y-y0) + `C`(z-z0) + `D` equation to get `Z` of the preceding plane  **step 3**

![meshgrid](https://raw.githubusercontent.com/unclebob7/dimensionality-reduction/master/meshgrid.png)
![analytic geometry](https://raw.githubusercontent.com/unclebob7/dimensionality-reduction/master/analytic_geometry.PNG)
> Now we can very explicitly why we have to implement *mean-normalization* to the raw dataset, since we have to pinpoint the *original point* as the start of the 2 **PC-vector**.Moreover , x0, y0, z0 were automatically set to 0 since we stipulate the `normal vector` goes through the *original point*.

**step 1**

```python
# principal components (orthogonal vectors that decide the hyperplane)
C = pca.components_
# normal-vector of the hyperplane
normal_vector = np.cross(C[0,:], C[1,:])
```

**step 2**

```python
axes = [-1.8, 1.8, -1.3, 1.3, -1.0, 1.0]

x1s = np.linspace(axes[0], axes[1], 10)
x2s = np.linspace(axes[2], axes[3], 10)
x1, x2 = np.meshgrid(x1s, x2s)
```

**step 3**

```python
# calculate z-value of the dataset
z = np.empty((10, 10))
for i in range(0,10):
  for j in range(0,10):
    z[i, j] = -(normal_vector[0]*x1[i, j]+normal_vector[1]*x2[i ,j])/normal_vector[2]
```

## data visualization

```python
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# setup parameters of the coordinates
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')    # first figure of 1*1 space
ax.set_xlabel("$x_1$", fontsize=18, color="tab:blue")
ax.set_ylabel("$x_2$", fontsize=18, color="tab:blue")
ax.set_zlabel("$x_3$", fontsize=18, color="tab:blue")
ax.set_xlim(axes[0:2])
ax.set_ylim(axes[2:4])
ax.set_zlim(axes[4:6])

# plot the X_decorrelated_DR dataset
k = 0
Z_projected = np.empty((60, 1))
for i in range(0,60):
  for j in range(0,2):
    while k<60:
      Z_projected[k] = -(normal_vector[0]*X_decorrelated_DR[i, j]+normal_vector[1]*X_decorrelated_DR[i ,j])/normal_vector[2]
      k = k+1

# plot the original datatset
ax.plot(X[:, 0], X[:, 1], X[:, 2], "bo", alpha=0.5)
      
# plot the X_decorrelated_DR dataset
ax.plot(X_decorrelated_DR[:, 0], X_decorrelated_DR[:, 1], Z_projected[:, 0], "mo", alpha=0.5)

# plot the track of projection
for j in range(0, 60):
  ax.add_artist(Arrow3D([X[j, 0], X_decorrelated_DR[j, 0]],[X[j, 1], X_decorrelated_DR[j, 1]],[X[j, 2], Z_projected[j, 0]], mutation_scale=15, lw=1, arrowstyle="-|>", color="tab:gray"))

# plot the hyperplane(2_D) for projection
ax.plot_surface(x1, x2, z, alpha=0.2, color="k")

# plot the principal compoents(orthogonal vectors that decide the hyperplane)
ax.add_artist(Arrow3D([0, C[0, 0]],[0, C[0, 1]],[0, C[0, 2]], mutation_scale=15, lw=1, arrowstyle="-|>", color="tab:red"))
ax.add_artist(Arrow3D([0, C[1, 0]],[0, C[1, 1]],[0, C[1, 2]], mutation_scale=15, lw=1, arrowstyle="-|>", color="tab:red"))

# plot the origin
ax.plot([0], [0], [0], "r.")

plt.show()
```

* blue points : original datatset
* magenta points : projected dataset
* gray arrow : projection trace

![projection](https://raw.githubusercontent.com/unclebob7/dimensionality-reduction/master/projection.PNG)

