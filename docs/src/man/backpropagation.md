# Backpropagation

```@contents
Pages = ["backpropagation.md"]
Depth = 2
```

## Notations

We denote the intermediate quantity to be

```math
z_j^l = \sum_k w_{jk}^l a_k^{l-1} + b_j^l,
```
where the ``w`` matrix has size ``n_l \times n_{l-1}``.

So the activation is

```math
a_j^l = \sigma(z_j^l).
```

Let ``x`` and ``y = y(x)`` denote ``1`` training input and output, ``L`` denotes the number of
layers in the network, and ``a^L = a^L(x)`` denotes the vector of activations output from the
network when ``x`` is input. Let subscripts denote each component in each ``y`` or each
activation ``a^L``.

Then the cost of the example ``(x, y)`` is

```math
C_x = \frac{ 1 }{ 2 } \sum_j \bigl(a_j^L - y_j\bigr)^2.
```

## Taking derivatives

A little change in the weight from the ``k``th node in the ``(L-1)``th layer to the ``j``th
node in the ``L``th layer (``dw_{jk}^L``) will cause a change of

```math
dC_x = \frac{ \partial C_x }{ \partial w_{jk}^L } \, dw_{jk}^L
```
in the cost function.

By applying the chain rule, we know
```math
\frac{ \partial C_x }{ \partial w_{jk}^L } =
\frac{ \partial C_x }{ \partial a_j^L } \frac{ \partial a_j^L }{ \partial z_j^L }
\frac{ \partial z_j^L }{ \partial w_{jk}^L } =
\bigl(a_j^L - y_j\bigr) \sigma'\bigl(z_j^L\bigr) a_k^{L-1},
```

which forms an ``n_L \times n_{L-1}`` matrix.

This corresponds to the following graph:

![wjk](https://i.imgur.com/2MEp7VJ.png)

Similarly, we have

```math
\frac{ \partial C_x }{ \partial b_j^L } =
\frac{ \partial C_x }{ \partial a_j^L } \frac{ \partial a_j^L }{ \partial z_j^L }
\frac{ \partial z_j^L }{ \partial b_j^L } =
\bigl(a_j^L - y_j\bigr) \sigma'\bigl(z_j^L\bigr).
```

``z_j^L`` is also related to ``a_k^{L-1}``. However, we cannot change ``a_k^{L-1}`` directly. We
can only change the weights and biases. If we take the derivative, we will get

```math
\begin{equation}\label{eq:pta}
    \frac{ \partial C_x }{ \partial a_k^{L-1} } =
    \sum_j \frac{ \partial C_x }{ \partial a_j^L } \frac{ \partial a_j^L }{ \partial z_j^L }
    \frac{ \partial z_j^L }{ \partial a_k^{L-1} } =
    \sum_j \bigl(a_j^L - y_j\bigr) \sigma'\bigl(z_j^L\bigr) w_{jk}^L.
\end{equation}
```

This corresponds to the following graph:

![ak](https://i.imgur.com/mBnnEYy.png)

But ``\eqref{eq:pta}`` is still important since, from it, we can compute partial derivatives
prior to the ``L``th orders of the weights and biases.

## Conclusions

Suppose the input layer is the first layer. Using induction, we can derive

```math
\begin{align}
    \frac{ \partial C_x }{ \partial w_{jk}^l } &=
    \frac{ \partial C_x }{ \partial a_j^l } \sigma'\bigl(z_j^l\bigr) a_k^{l-1}, \label{eq:cw}\\
    \frac{ \partial C_x }{ \partial b_j^l } &=
    \frac{ \partial C_x }{ \partial a_j^l } \sigma'\bigl(z_j^l\bigr), \label{eq:cb}
\end{align}
```

where

```math
\frac{ \partial C_x }{ \partial a_j^l } =
\begin{cases}
    {\displaystyle\sum}_i \dfrac{ \partial C_x }{ \partial a_i^{l+1} }
    \dfrac{ \partial a_i^{l+1} }{ \partial z_i^{l+1} }
    \dfrac{ \partial z_i^{l+1} }{ \partial a_j^l } =
    {\displaystyle\sum}_i \dfrac{ \partial C_x }{ \partial a_i^{l+1} }
    \sigma'\bigl(z_i^{l+1}\bigr) w_{ij}^{l+1}, &\text{if }2 \leq l \leq L - 1;\\
    a_j^L - y_j, &\text{if }l = L.
\end{cases}
```

Define _error_ ``\boldsymbol{\delta}`` as[^1]

```math
\delta_j^l \equiv \frac{ \partial C_x }{ \partial z_j^l }.
```

Therefore,

```math
\boldsymbol{\delta}^l =
\begin{cases}
    \bigl(\mathrm{w}^{l+1}\bigr)^\intercal \boldsymbol{\delta}^{l+1} \odot
    \boldsymbol{\sigma}'\bigl(z^l\bigr), &\text{if }2 \leq l \leq L - 1;\\
    \nabla_a C_x \odot \boldsymbol{\sigma}'\bigl(z^L\bigr), &\text{if }l = L.
\end{cases}
```

where ``\odot`` denotes the _Hadamard product_, ``\bigl(\mathrm{w}^{l+1}\bigr)^\intercal`` is
the transpose of the weight matrix, and
``\bigl(\mathrm{w}^{l+1}\bigr)^\intercal \boldsymbol{\delta}^{l+1}`` is the
matrix-vector multiplication.

Check size:

- ``\nabla_a C_x \odot \boldsymbol{\sigma}'\bigl(z^L\bigr)`` will produce an size-``n_L`` vector.
- ``\bigl(\mathrm{w}^{l+1}\bigr)^\intercal`` is an ``n_l \times n_{l+1}`` matrix,
  while ``\bigl(\mathrm{w}^{l+1}\bigr)^\intercal \boldsymbol{\delta}^{l+1} \odot \boldsymbol{\sigma}'\bigl(z^l\bigr)``
  is a size-``n_l`` vector.

Suppose we know the error ``\boldsymbol{\delta}^{l+1}`` at the ``(l + 1)``th layer, when
applying the transpose weight matrix, we are moving the error __back__ through the network,
giving us some sort of measure of the error at the output of the ``l``th layer.

Equations ``\eqref{eq:cw}`` and ``\eqref{eq:cb}`` can be rewritten as

```math
\begin{align}
    \frac{ \partial C_x }{ \partial w_{jk}^l } &= \delta_j^l a_k^{l-1} =
    \boldsymbol{\delta}^l \otimes \bigl(\mathbf{a}^{l-1}\bigr),&\text{}\\
    \frac{ \partial C_x }{ \partial b_j^l } &= \delta_j^l.
\end{align}
```

## References

1. [Backpropagation calculus | Chapter 4, Deep learning](https://youtu.be/tIeHLnjs5U8)
1. [How the backpropagation algorithm works](http://neuralnetworksanddeeplearning.com/chap2.html)

------

[^1]: Note when ``C = C(w, b)``, ``\nabla_b C \neq \nabla C``. While ``\nabla_a C(a)`` and ``\nabla_b C(w, b)`` have a one-to-one correspondence.
