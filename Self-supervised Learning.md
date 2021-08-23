# Self-supervised Learning

事实上，**Self-supervised Learning是无监督学习的一种，它的主要目标在于学习到模型的一种通用的特征表示**，并将其用于下游任务；其主要的方式就是通过自己监督自己，比如把一段话里面的几个单词去掉，用他的上下文去预测缺失的单词，或者将图片的一些部分去掉，依赖其周围的信息去预测缺失的 patch。

> This is where we train a model using labels that are naturally part of the input data, rather than requiring separate external labels.

基于此，Self-supervised Learning 实际上是一种良好的预训练模型的方式，在这一机制下，Self-supervised Learning 的评价主要依据下游任务(Downstream Task)的各项性能进行评价。

## BERT

> BERT（Bidirectional Encoder Representations from Transformers） 是一个语言表示模型。它的主要模型结构是由`trasnformer`的`encoder`堆叠而成，它其实是一个二阶段的框架，分别是`pretraining`，以及在各个具体任务上进行`finetuning`。

BERT的主要结构是由多层`Transformer`搭建而成，事实上，其Bidirectional 的性质就体现在其`Encoder`层在会接收整个句子的`attention`而非只有向左的`attention`。其BASE版本中单个模型中将会含有12个`Encoder`，而LARGE模型中将会含有24层`Encoder`。

BERT的基本模式如下：

<img src="https://gitee.com/hsuyuanliu/picrepo/raw/master/img/68747470733a2f2f692e6c6f6c692e6e65742f323032312f30352f30382f52664d383134714c4f7647736a77322e706e67.png" style="zoom:67%;" />

也就是说，**BERT实际上完成了一个“填空”操作**，在这一机制下，每次训练中，BERT将会随机遮蔽输入序列的某一个词，之后通过`Encoder`操作以及一些其它的诸如回归的操作，希望训练网络得到原始的序列中的值，这一过程的结果将会是类似于词向量的向量，它们将会比较比较精确地对输入进行表示。

此外，应当注意，在不同位置的同一单词将会得到不同的表示方式；同样的，一词多义也将由这一形式进行体现。

> Beyond masking 15% of the input, BERT also mixes things a bit in order to improve how the model later fine-tunes. Sometimes it randomly replaces a word with another word and asks the model to predict the correct word in that position.

对于种种`pretrain`的方式，一种被经常使用的是Two Sentence Model；

在这一情况下，第一个序列由`cls`起始，`cls`作为句子的起始并无太大意义；句子之间会使用`seq`进行隔开，其输入序列的基本生成操作如下：

- 分词操作
- 插入分隔符与起始符
- 完成`Token`操作，实际上该表征是由三部分组成的，分别是对应的**token**，**分割**和**位置** `embeddings`，它们共同作用，求和后得到最终的输入向量

<img src="https://gitee.com/hsuyuanliu/picrepo/raw/master/img/68747470733a2f2f692e6c6f6c692e6e65742f323032312f30352f30382f6c315a7879336447325452694353562e706e67.png" style="zoom:50%;" />



在这里：

> - `Token Embeddings`， 词的向量表示
> - `Segment Embeddings`，辅助BERT区别句子对中的两个句子的向量表示
> - `Position Embeddings` ，让BERT学习到输入的顺序属性

基于此得到的预训练模型，进行微调后（主要针对输出操作）后即可用于具体的各个任务，**在这一过程中，对下游任务的训练也会导致`BERT`中参数经历的一些变化**。

<img src="https://gitee.com/hsuyuanliu/picrepo/raw/master/img/68747470733a2f2f692e6c6f6c692e6e65742f323032312f30352f30382f686d696678655958516b41674f42522e706e67.png" style="zoom:67%;" />