# GloVe

可以得知，`Word2Vec`对词向量的构建是基于局部信息的，即其对局部上下文窗口的概率进行计算，而诸如LSA等方法基于全局矩阵分解；在这一条件下，可以对两者进行综合，使用一个某一种**基于全局词汇共现的统计信息**的方法完成对词汇的计算。

`GloVe`与`word2vec`两者最直观的区别在于，`word2vec`是“predictive”的模型，而`GloVe`是“count-based”的模型。

Count-based模型，如`GloVe`，本质上是对共现矩阵进行降维。首先，构建一个词汇的共现矩阵，每一行是一个word，每一列是context。共现矩阵就是计算每个word在每个context出现的频率。由于context是多种词汇的组合，其维度非常大，我们希望像network embedding一样，在context的维度上降维，学习word的低维表示。

### 共现矩阵

Glove需要首先构建出一个co-occurrence矩阵$X$，这一矩阵的$X_{i,j}$项说明了单词$j$在context项 $i$附近出现的次数，在这一情况下，$X_i=\sum_k X_{ik}$即为出现在单词$i$附近（i作为context）单词的总数，在这一情况下：
$$
P_{i,j}=P(j|i)=\frac{X_{i，j}}{X_i}
$$
即为单词j出现在context $i$附近的概率。实际上，将所有的共现矩阵的值设为相同也是有问题的，这一点可以在之后进行优化。

共现矩阵的意义在于：当两词足够接近时，其相关性也会越大，文中举出的例子是solid与ice以及gas与steam的，即可以认为，solid与ice的权重越大意味着其关联度越大，这显然是正确的。

### 公式推导

不妨令$P_{ik}=P(i|k)，P_{jk}=P(j|k)$，则可以写出函数$F$为：
$$
F(w_i,w_j,\tilde w_k)=\frac {P_{ik}}{P_{jk}}
$$
这一$F$，即是期望通过学习得到的两个单词的关系的一种量化表示，是学习的目标；

进一步地，其中,的$w_{i}, w_{j}, \tilde{w}_{k}$ 为词汇 $i, j, k$ 对应的词向量, 其维度都为 $d$, 而 $\frac{P_{i k}}{P_{i k}}$ 则可以直接通过语料计算得到, 这 里 $F$ 为一个未知的函数。由于词向量都是在一个线性向量空间，因此，可以对 $w_{i}, w_{j}$ 进行差分, 将上 式转变为如下:
$$
F\left(w_{i}-w_{j}, \tilde{w}_{k}\right)=\frac{P_{i k}}{P_{j k}}
$$
由于上式中左侧括号中是两个维度为 $d$ 的词向量，而右侧是一个标量，因此，可以将$F$猜想为向量的内积的形式, 因此, 上式可以进一步改变为:
$$
F\left(\left(w_{i}-w_{j}\right)^{T} \tilde{w}_{k}\right)=\frac{P_{i k}}{P_{j k}}
$$
这种方法也可以防止$F$将会使用某种破坏向量维度的方式进行计算；

在此基础上，若$w_i$与$w_j$完成某一次互换，希望仍能保留这一结构（relabel），因此写出：
$$
F\left(\left(w_{i}-w_{j}\right)^{T} \tilde{w}_{k}\right)=\frac{F(w_i^T \tilde w_k)}{F(w_j^T \tilde w_k)}
$$
与上式连立，写出
$$
F(w_i^T \tilde w_k)=P_{ik}=\frac{X_{ik}}{X_i}
$$
为满足这一形式，不妨令$F=\exp$，即可知：
$$
w_i^T \tilde w_k=\log(P_{ik})= \log(X_{ik})-\log(X_i)
$$
在这一基础上，加上某一偏置$b_i$即可以写出最终的简化：
$$
w_i^T \tilde w_k+b_i+\tilde {b_k}=\log(X_{ik})
$$
此时, $\log X_{i}$ 已经包含在 $b_{i}$ 当中。因此, 此时模型的目标就转化为通过学习词向量的表示, 使得上式两边尽量接近，因此，可以通过计算两者之间的平方差来作为目标函数，即:
$$
J=\sum_{i, k=1}^{V}\left(w_{i}^{T} \tilde{w}_{k}+b_{i}+b_{k}-\log X_{i k}\right)^{2}
$$
但是这样的目标函数有一个缺点, 就是对所有的共现词汇都是采用同样的权重, **因此, 作者对目标函 数进行了进一步的修正，通过语料中的词汇共现统计信息来改变他们在目标函数中的权重, 具体如 下:**
$$
J=\sum_{i, k=1}^{V} f\left(X_{i k}\right)\left(w_{i}^{T} \tilde{w}_{k}+b_{i}+b_{k}-\log X_{i k}\right)^{2}
$$
这里 $V$ 表示词汇的数量, 并且权重函数 $f$ 必须具备以下的特性:
- $f(0)=0$, 当词汇共现的次数为0时, 此时对应的权重应该为 0 。
- $f(x)$ 必须是一个非减函数, 这样才能保证当词汇共现的次数越大时，其权重不会出现下降的情 况。
- 对于那些太频繁的词, $f(x)$ 应该能给予他们一个相对小的数值，这样才不会出现过度加权。
综合以上三点特性，作者提出了下面的权重函数：
$$
f(x)=\left\{\begin{array}{cl}
\left(x / x_{\max }\right)^{\alpha} & \text { if } x<x_{\max } \\
1 & \text { otherwise }
\end{array}\right.
$$
作者在实验中设定 $x_{\max }=100$, 并且发现 $\alpha=3 / 4$ 时效果比较好。即完成了整个模型的搭建。

### 总结

作者的思路大致是基于最开始的猜想一步一步简化模型的计算目标，最终找到具体的`loss function`，其

- `Word2vec`是无监督学习，同样由于不需要人工标注，`glove`通常被认为是无监督学习，但实际上glove还是有label的，即共现次数$log(X_{i,j})$
- `Word2vec`损失函数实质上是带权重的交叉熵，权重固定；`glove`的损失函数是最小平方损失函数，权重可以做映射变换。
- Glove利用了全局信息，使其在训练时收敛更快，训练周期较`Word2vec`较短且效果更好。

