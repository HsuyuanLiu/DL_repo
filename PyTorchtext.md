# PyTorchtext中数据的使用

torchtext包含以下组件：

- Field :主要包含以下数据预处理的配置信息，比如指定分词方法，是否转成小写，起始字符，结束字符，补全字符以及词典等等
- Dataset :继承自pytorch的Dataset，用于加载数据，提供了TabularDataset可以指点路径，格式，Field信息就可以方便的完成数据加载。同时torchtext还提供预先构建的常用数据集的Dataset对象，可以直接加载使用，splits方法可以同时加载训练集，验证集和测试集。
- Iterator : 主要是数据输出的模型的迭代器，可以支持batch定制

### Field

Torchtext采用声明式方法加载数据，需要先声明一个Field对象，**这个Field对象指定要怎么处理某个数据**,再针对这一个Field进行诸如tokenize等操作。其基本的参数如下：

```
sequential: Whether the datatype represents sequential data. If False,
            no tokenization is applied. Default: True.
use_vocab: Whether to use a Vocab object. If False, the data in this
            field should already be numerical. Default: True.
init_token: A token that will be prepended to every example using this
            field, or None for no initial token. Default: None.
lower: Whether to lowercase the text in this field. Default: False.
  tokenize: The function used to tokenize strings using this field into sequential       
  examples. If "spacy", the SpaCy English tokenizer is used. Default: str.split.
batch_first: Whether to produce tensors with the batch dimension first.
            Default: False.

```

基于此，需要在处理field对象之前使对tokenize进行定义。

