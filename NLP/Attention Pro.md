# Attention Pro

本文希望以更详细的方式对Attention的具体实现进行说明，Transformer的结构如下图所示。我们知道end2end的机器翻译模型一般都是Encoder+Decoder的组合，Encoder对源句子进行编码，将编码信息传给Decoder，Decoder翻译出目标句子。Transformer也不例外，下图左边即为Encoder，右边即为Decoder；

<img src="https://gitee.com/hsuyuanliu/picrepo/raw/master/img/p13.png" style="zoom:50%;" />

### 计算步骤

在Encoder中，可以通过对self-attention的计算了解整个过程：

<img src="https://gitee.com/hsuyuanliu/picrepo/raw/master/img/p14.png" style="zoom:47%;" />

以针对第二个词$e_2$计算的过程为例：

**首先对其词向量$e_2$进行线性变换，即乘以矩阵$Q$，得到Query，这就是标准Attention中的Query。其次，对周围所有的其他词，乘以另一矩阵$K$，因此可以得到很多的Key。然后，Query和所有Key做点积，并用softmax归一化，得到了Query在周围词上的Attention score distribution。接着，周围词乘以另一个线性变换矩阵$V$，即可以得到Value，最后，Value和Attention score distribution进行加权求和，并加上$e2$自己，送给FFNN，作为下一阶段的$e_2'$输入。而图中右下角的公式中的分母只是个缩放因子。**

将其作为一个整体，可以由下图表示：

![atten](https://gitee.com/hsuyuanliu/picrepo/raw/master/img/atten.PNG)

而在Decoder部分每一层有三个子层组成，包括Self-Attention、Encoder-Decoder Attention和FFNN，但值得注意的是，其计算步骤中，应当蒙住其后位置的词，这是因为，Decoder只能看到当前要翻译的词之前的所有词，而看不到之后的所有词，因此其也被称为Masked Self-Attention。这也说明Transformer只是在Encoder阶段可以并行化，Decoder阶段依然要一个个词顺序翻译，依然是串行的。

但在这一机制下（即单一的self-attention下），decoder只可以采取已经完成翻译的信息而不能接收到来自encoder的信息，因此需要引入一个Encoder-Decoder Attention层，**在这一层中，接收源句子的每个词所一一对应的输出向量，作为Encoder-Decoder Attention的Keys和Values，而从目标句子当前要翻译的词的Decoder Self-Attention出来的向量就是Encoder-Decoder Attention的Query。**

而所谓Multi-head，其作用类似于CNN中的多个kernel，分别提取了语句中的不同信息，提取多个Attention输出向量，并拼接起来。

<img src="https://gitee.com/hsuyuanliu/picrepo/raw/master/img/p25.png"  style="zoom:50%;" />

但是，值得注意的是，在进行`Multi-head`的计算时，所使用的$K,Q,V$矩阵都应做相应的变化，即其将会从单一`head`的$d \times d$维变化为$d \times \frac{d}{H}$维度，其中H为head的数目，这样才可以保证输入词向量矩阵与输出矩阵的大小保持一致。

# 代码分析

```python
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
```

即其基本结构在于在decode中套一层encode，

而其attention的实现在于：

```python
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
```

在单一网络中，multi-head attention将会接受一套$K,Q,V$矩阵并对其完成拆分后重新计算并按序进行排列，最终得到与原始数据相同大小的输出。













[Code Example](http://nlp.seas.harvard.edu/2018/04/03/attention.html)





