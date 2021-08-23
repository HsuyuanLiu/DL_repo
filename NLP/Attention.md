# Attention&&Transformer

## Attention

### 背景

在诸如$RNN$，$CNN$等较为传统的模型中，有着各式各样的的问题，例如，$RNN$的结构使得其并行化操作难以执行，而$CNN$模型使其只能接收到到某一段文本的信息；而应该注意的是：在句法中，某些词的顺序从整体上并不会影响整句话的意义。因此，在$Seq2Seq$等模型中，如果使用某一种新型机制，使得模型对文本的理解可以基于整体而非某一些单词的顺序是很有意义的，事实上，Attention机制是一种对于序列的加权平均的关系。它将会接受更多来自需要被注意的文本的信息。

**事实上，存在着很多Attention的变体机制，例如Self Attention.**

`Self Attention`与传统的`Attention`机制非常的不同：传统的`Attention`是基于source端和target端的隐变量计算Attention的，得到的结果是源端的每个词与目标端每个词之间的依赖关系。但Self Attention不同，它分别在source端和target端进行，仅与source input或者target input自身相关的`Self Attention`，捕捉source端或target端自身的词与词之间的依赖关系；然后再把source端的得到的self Attention加入到target端得到的Attention中，捕捉source端和target端词与词之间的依赖关系。因此，`self Attention` 中的Attention比传统的Attention mechanism效果要好，

### Attention多重应用

Attention机制的核心是对不同的词向量使用其内容而非位置作为其训练模型的依据。它引入了$K$,$Q$,$V$等向量，$Q$即为Query，是To match others的向量；$K$代表着Key，是To be matched的向量；$V$代表着Value，是To be weighted averaged 的向量。

基于此，可以认为$Q$与$K$向量实际上是等价的，其区别仅仅在于一个在Source中，一个在Target中，在实际模型中，其初始化过程也是相同的。

#### Attention in Seq2Seq

在`Seq2Seq`模型中引入`Attention`的基本机制如下：

<img src="https://gitee.com/hsuyuanliu/picrepo/raw/master/img/86be2df0-2928-4f90-ba41-c5fd9f3459f1.png" style="zoom: 50%;" />

`Encoder RNN`层，首先得到其隐藏层，然后对其乘以Key矩阵得到$K_:i$
$$
Key :\quad k_:i=W_Kh_i
$$
使用同样的操作，从`Decoder RNN`中得到了Query矩阵：
$$
Query :\quad q_:j=W_Q s_j
$$
这样就可以开始计算`Encoder`与`Decoder`层中元素两两对应的Weights数值：
$$
Weights: \quad  \alpha_:j=Softmax(K^Tq_:j)
$$
最后，通过对`value`与$\alpha$进行加权得到了所对应的`context vector`并计算得到新的`target output`。

**因此，整个过程可以总结为4步**：

- 得到注意力层的所有`encoder`$s_1,s_2...s_m$层与单个`decoder`层输入$h_t$
- 计算每个输入的相关性
- 计算注意力权重，并使用$Softmax$函数处理这一注意力分数
- 求出注意力机制的加权和

<img src="https://gitee.com/hsuyuanliu/picrepo/raw/master/img/b0c576a1-66c0-48ed-9ea8-58af70242f5d.png" style="zoom: 33%;" />

#### Attention Without RNN

在`Seq2Seq`模型中的`Attention`，实际上还是在`RNN`的大框架下实现的文本处理模型，因此也不可避免地保留了一些`RNN`中的缺陷，诸如位置影响输出结果，在这一机制下，可以选择使用完全抛弃`RNN`的模型，直接通过`Attention`层的输出得到生成的序列：

在这一模型下，实际上是去除了基于`RNN`得到的隐藏层而直接在词嵌入或者`one-hot`等方法得到的输入的基础上得到的向量上进行`Attention`机制的计算：单独的这一机制没有太大的意义，而是应该被用于在`Transformer`等具体模型中。

#### Self-Attention Layer

`Self-attention`与单纯的`Attention`不同，它的$Q,V,K$矩阵均来源于自身而非一个待匹配的`decoder`输入，

<img src="https://gitee.com/hsuyuanliu/picrepo/raw/master/img/7ec5c473-a79c-4ede-9e31-a26eb09cb91a.png" style="zoom:33%;" />

这也引出了一个问题：针对自身的注意力计算究竟有什么好处？

可以引入一个实例进行说明：

<img src="https://gitee.com/hsuyuanliu/picrepo/raw/master/img/81f73bf2-eddc-4514-8749-fe6c998541f6.png" style="zoom:67%;" />

**可以看出，`Self Attention`可以捕获同一个句子中单词之间的一些句法特征（比如有一定距离的短语结构）或者语义特征（比如图中展示的`its`的指代对象`Law`）。**

**引入Self Attention后会更容易捕获句子中长距离的相互依赖的特征，因为如果是RNN或者LSTM，需要依次序序列计算，对于远距离的相互依赖的特征，要经过若干时间步步骤的信息累积才能将两者联系起来，而距离越远，有效捕获的可能性越小。**

**但是Self Attention在计算过程中会直接将句子中任意两个单词的联系通过一个计算步骤直接联系起来，所以远距离依赖特征之间的距离被极大缩短，有利于有效地利用这些特征。**



#### Multi-head

`Multi-head Attention`机制可以使得模型分别聚焦于不同的事物：

在这一机制下，Attention层将会拥有多个$Q$,$K$,$V$权重矩阵，每一组它们都将会被随机地初始化，在训练后，可以将初始的嵌入层输入映射到不同的子空间中，这些不同的表示可以有助于实现不同方向上信息的接收与操作。

<img src="https://gitee.com/hsuyuanliu/picrepo/raw/master/img/52648cec-531d-40b5-af6c-17ae1ed6738e.png" style="zoom:50%;" />

对这些得到的Multi-head矩阵进行拼接后需要乘以一个额外的权重矩阵$W_O$,这一矩阵也需要经过训练。最终得到了Attention层的输出向量。

<img src="https://gitee.com/hsuyuanliu/picrepo/raw/master/img/fa0e856f-9d00-41a4-8d02-a53662633e4c.png" style="zoom:60%;" />

### Attention的一些总结

在以往的模型中，如果希望通过Source输入得到Target输出，往往是在对Source进行处理后得到某一个状态后统一生成输出，即满足以下范式：
$$
y_1=f(C)  \qquad
y_2=f(C,y_1) \qquad
y_n=f(C,y_1,y_2...y_{n-1})
$$

此时`f`是`Decoder`的非线性变换函数。从这里可以看出，在生成目标句子的单词时，不论生成哪个单词，它们使用的输入句子Source的语义编码C都是一样的，没有任何区别，因此可以认为：source中的任一单词对Target中的任意单词的作用是相同的。

而Attention机制的引入是基于：**目标句子中的每个单词都应该学会其对应的源语句子中单词的注意力分配概率信息。**这意味着在生成每个单词的时候，原先都是相同的中间语义表示$C$会被替换成根据当前生成单词而不断变化的$C_i$。理解Attention模型的关键就是这里，即由固定的中间语义表示$C$换成了根据当前输出单词来调整成加入注意力模型的变化的$C_i$。增加了注意力模型的Encoder-Decoder框架理解起来如图所示：

<img src="https://gitee.com/hsuyuanliu/picrepo/raw/master/img/d21aa0c5-3ba0-46b8-b5d1-30c0b8ce84ba.png" style="zoom:80%;" />

即，生成每一个Target单词的输出公式变为：
$$
y_1=f(C_1)  \\y_2=f(C_2,y_1) \\...\\y_n=f(C_n,y_1,y_2...y_{n-1})
$$
而Attention的计算机制则如下：**将Source中的构成元素想象成是由一系列的<Key,Value>数据对构成，此时给定Target中的某个元素Query，通过计算Query和各个Key的相似性或者相关性，得到每个Key对应Value的权重系数，然后对Value进行加权求和，即得到了最终的Attention数值。所以本质上Attention机制是对Source中元素的Value值进行加权求和，而Query和Key用来计算对应Value的权重系数。**

即可以写为以下公式：
$$
Attention(Query,Source)=\sum\nolimits_{i=1}^{Length(source)}Similarity(Query,key_i)*value_i
$$
用图像表示如下：

<img src="https://gitee.com/hsuyuanliu/picrepo/raw/master/img/50eb6930-04be-47cf-9076-f69284bb94b1.png" style="zoom:87%;" />

> 除了将其作为一种信息关联机制外，也可以将Attention机制看作一种软寻址（Soft Addressing）:Source可以看作存储器内存储的内容，元素由地址Key和值Value组成，当前有个Key=Query的查询，目的是取出存储器中对应的Value值，即Attention数值。通过Query和存储器内元素Key的地址进行相似性比较来寻址，之所以说是软寻址，指的不像一般寻址只从存储内容里面找出一条内容，而是可能从每个Key地址都会取出内容，取出内容的重要性根据Query和Key的相似性来决定，之后对Value进行加权求和，这样就可以取出最终的Value值，也即Attention值。



即总共可以分为三个阶段：

**在第一个阶段，可以引入不同的函数和计算机制，根据Query和某个$Key_i$，计算两者的相似性或者相关性**，最常见的方法包括：求两者的向量点积、求两者的向量$Cosine$相似性或者通过再引入额外的神经网络来求值，即如下方式：
$$
Dot \;  production: \quad Similarity(Query,key_i)= Query \cdot key_i \\
Cosine :\quad Similarity(Query,key_i)= \frac{Query \cdot key_i}{||Query|| \cdot ||key_i||} \\
MLP:\quad Similarity(Query,key_i) =MLP((Query,key_i))
$$
第一阶段产生的分值根据具体产生的方法不同其数值取值范围也不一样，**第二阶段引入类似`Softmax`的计算方式对第一阶段的得分进行数值转换**，一方面可以进行归一化，将原始计算分值整理成所有元素权重之和为1的概率分布；另一方面也可以通过`Softmax`的内在机制更加突出重要元素的权重。
$$
\alpha_i=Softmax(Similarity)=\frac{e^{Sim_i}}{\sum\nolimits_{j=1}^{L_x}e^{Sim_j} }
$$
第二阶段的计算结果$\alpha_i$即为$value_i$对应的权重系数，然后进行加权求和即可得到Attention数值。

## Transformer模型基本结构

Transformer的基本结构如下所示：

<img src="https://gitee.com/hsuyuanliu/picrepo/raw/master/img/4d953f3f-0751-40f8-a388-26377e8987d2.png" style="zoom:100%;" />

`Transformer`的本质上是一个`Encoder-Decoder`的结构，可以将其分为两部分，即编码器与解码器；编码器由6个编码`block`组成，而同样解码器是6个解码`block`组成，其中每一个编/解码器的`block`如上图所示。



编码器的输入首先会经历一个`self-attention`层，，这一层将会使其从输入的其它单词处学习到信息，然后在经过正则化后，输入至一个前馈神经网络，相同的神经网络将会被用于到`Encoder`的各个层中。

在`Decoder`中也有着同样的结构，但在两层之中加入了一个新的`Encoder-Decoder`层解码器，这可以使得其得到解码器相关部分的输入信息。

<img src="https://gitee.com/hsuyuanliu/picrepo/raw/master/img/7087eb56-723e-46b3-a3a5-eb43c5dc4348.png" style="zoom:80%;" />

在每一层`self-attention`背后，都会跟着一个`Add&Normalize`层，在这里，会完成一次拼接操作后映射产生一个新的$Z$向量，进行下一步处理。

<img src="https://gitee.com/hsuyuanliu/picrepo/raw/master/img/c12553ce-eb65-4d7a-911c-9d5bffde4e1a.png" style="zoom:80%;" />

而对于解码器，它在`encoder-decoder`层接收一来自顶层`encoder`的$K$,$V$矩阵，来帮助`decoder`正确处理输入序列。

> The encoder start by processing the input sequence. The output of the top encoder is then transformed into a set of attention vectors $K$ and $V$. These are to be used by each decoder in its “encoder-decoder attention” layer which helps the decoder focus on appropriate places in the input sequence:

在每层`decoder`输出一个结束符后，输出序列进入下一层`decoder`，此外，应当注意，`decoder`层的其实输入与`encoder`层的起始输入都需要加上一个基于位置的偏移。

与`Encoder`层不同，在`Decoder`层中，`Self-Attention`层的计算基于先前位置的词向量，其位置后的词向量会被遮蔽。

> In the decoder, the self-attention layer is only allowed to attend to earlier positions in the output sequence. This is done by masking future positions (setting them to `-inf`) before the `softmax` step in the self-attention calculation.

最终，`Decoder`层的输出结果将会在经过一个`Softmax`层后完成输出操作。。 :slightly_smiling_face:





### 参考

[Seq2Seq model detailed explain](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html)



[Visualizing-translation model based on  Seq2Seq](http://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)



[An Introduction of Transformer Model](http://jalammar.github.io/illustrated-transformer/) 



[Some detailed Principles about why attention works](https://zhuanlan.zhihu.com/p/37601161)





