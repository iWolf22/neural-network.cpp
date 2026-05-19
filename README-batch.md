### Matrix batching

Matrix batching leverages matrix multiplication, a heavily optimized operation for modern CPUs and GPUs. Suppose the batch contains $m$ training samples,

$$
B = \{x_1, x_2, \dots, x_m\}
$$

Instead of a single activation vector, we now stack activations horizontally, one for each training samples,

$$
\textbf{A}^{(L)} =
\begin{bmatrix}
| & | & & | \\
\textbf{a}^{(L)}_1 &
\textbf{a}^{(L)}_2 &
\cdots &
\textbf{a}^{(L)}_m \\
| & | & & |
\end{bmatrix}
\in [0, 1]^{n_{L-1} \times m}
$$

Same applies to our biases,

$$
\textbf{B}^{(L)} =
\begin{bmatrix}
| & | & & | \\
\textbf{b}^{(L)} &
\textbf{b}^{(L)} &
\cdots &
\textbf{b}^{(L)} \\
| & | & & |
\end{bmatrix}
\in \mathbb{R}^{n_L \times m}
$$

#### Forward pass

Now we can represent our pre-activation/weighted sum as,

$$
\begin{align*}
\textbf{Z}^{(L)}
& =
\textbf{W}^{(L)}\textbf{A}^{(L-1)} + \textbf{B}^{(L)} \\
& =
\begin{bmatrix}
| & | & & | \\
\textbf{z}^{(L)}_1 &
\textbf{z}^{(L)}_2 &
\cdots &
\textbf{z}^{(L)}_m \\
| & | & & |
\end{bmatrix}
\in \mathbb{R}^{n_{L-1} \times m}
\end{align*}
$$

And our next activations as,

$$
\textbf{A}_i^{(L)} = \sigma(\textbf{Z}_i^{(L)})
$$

#### Error term

Let $\delta_i = \frac{\partial C}{\partial z_i}$ be the error term.

$$
\begin{align*}
\delta_i^{(L)}
& =
\frac{\partial C_x}{\partial a_i^{(L)}}
\frac{\partial a_i^{(L)}}{\partial z_i^{(L)}} \\
& =
\frac{\partial C_x}{\partial a_i^{(L)}}
\sigma^\prime(z_i^{(L)}) \\
\mathbf{\Delta}^{(L)}
& =
\frac{\partial C_x}{\partial \textbf{A}^{(L)}}
\odot
\sigma^\prime(\textbf{Z}^{(L)}) \\
& =
\begin{bmatrix}
| & | & & | \\
\boldsymbol{\delta}^{(L)}_1 &
\boldsymbol{\delta}^{(L)}_2 &
\cdots &
\boldsymbol{\delta}^{(L)}_m \\
| & | & & |
\end{bmatrix}
\in \mathbb{R}^{n_L \times m}
\end{align*}
$$

#### Backward pass

$$
\begin{align*}
\frac{\partial C_x}{\partial a_i^{(L-1)}}
& =
\sum_{j=0}^{n_L - 1}
\frac{\partial C_x}{\partial a_j^{(L)}}
\frac{\partial a_j^{(L)}}{\partial z_j^{(L)}}
\frac{\partial z_j^{(L)}}{\partial a_i^{(L-1)}} \\
& =
\sum_{j=0}^{n_L - 1}
\delta_j^{(L)}
w_{j,i}^{(L)} \\
\frac{\partial C_x}{\partial \textbf A^{(L-1)}}
& =
(\textbf W^{(L)})^T
\mathbf\Delta^{(L)}
\end{align*}
$$

#### Weight gradient

Forgive me for handwaving some of these steps!

$$
\begin{align*}
\frac{\partial C_x}{\partial w_{i,j}^{(L)}}
& =
\frac{\partial C_x}{\partial a_i^{(L)}} \frac{\partial a_i^{(L)}}{\partial z_i^{(L)}} \frac{\partial z_i^{(L)}}{\partial w_{i,j}^{(L)}} \\
& =
\delta_i^{(L)} a_j^{(L-1)} \\
\frac{\partial C}{\partial w_{i,j}^{(L)}}
& =
\frac{1}{|B|} \sum_{x \in B} \frac{\partial C_x}{\partial w_{i,j}^{(L)}} \\
& =
\frac{1}{|B|} \sum_{x \in B} \delta_i^{(L)} a_j^{(L-1)} \\
& =
\frac{1}{|B|}(\mathbf{\Delta}^{(L)} (\textbf{A}^{(L-1)})^T)_{ij} \\
\frac{\partial C}{\partial \textbf{W}^{(L)}}
& =
\frac{1}{|B|}
\mathbf{\Delta}^{(L)}
(\textbf{A}^{(L-1)})^T
\end{align*}
$$

#### Bias gradient

$$
\begin{align*}
\frac{\partial C_x}{\partial b_i^{(L)}}
& =
\frac{\partial C_x}{\partial a_i^{(L)}} \frac{\partial a_i^{(L)}}{\partial z_i^{(L)}} \frac{\partial z_i^{(L)}}{\partial b_i^{(L)}} \\
& =
\delta_i^{(L)} \\
\frac{\partial C}{\partial b_i^{(L)}}
& =
\frac{1}{|B|} \sum_{x \in B} \frac{\partial C_x}{\partial b_i^{(L)}} \\
& =
\frac{1}{|B|} \sum_{x \in B} \delta_i^{(L)}
\end{align*}
$$

#### Mean Squared Error

$$
C = \frac{1}{m}||(\textbf{A}^{(L)} - \textbf{Y})^2||_F
$$

where $\textbf Y$ is a matrix of the labels.

Differentiating gives,

$$
\frac{\partial C}{\partial \mathbf A^{(L)}} = \frac{2}{m} (\mathbf A^{(L)} - \mathbf Y)
$$
