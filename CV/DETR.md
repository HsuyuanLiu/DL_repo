# DETR

### 概述

整体来看，该模型首先通过CNN提取特征，然后得到的特征进入transformer, 最后将transformer输出的结果转化为class和box。

<img src="https://files.mdnice.com/user/18116/f8b39efd-ea33-48bf-ae98-7eb333460bef.png" style="zoom:110%;" />

基于此，可以发现其与Faster-RCNN等模型有较大不同，即其不需要通过**大量的锚框**进行搜索，也不依赖于较复杂的pipeline进行数据的处理，从这一点上说，这种做目标检测的方法更合理。先前的模型，需要使用密集的先验覆盖整张图中可能出现物体的部分，然后进行预测，而DETR更倾向于直接指出目标的位置而不依赖于先验的预测。

<img src="https://files.mdnice.com/user/18116/f2c9b3dd-04ac-4dd0-b64a-11c47c9795f4.png" style="zoom:187%;" />

但实际上，DETR的特征提取部分还是基于CNN网络，即其Transformer接收的输入实际上上CNN进行特征提取后得到的特征图而非原始图像。

### Transformer

##### Encoder

在DETR中，首先用CNN backbone处理$3 *H_0*W_0$ 维的图像，得到 $C*H*W$维的feature map,然后将backbone输出的feature map和position encoding相加，输入Transformer Encoder中处理，得到用于输入到Transformer Decoder的image embedding。

<img src="https://files.mdnice.com/user/18116/b7d4644c-ea15-4f20-92e4-fa5c3308e984.png" style="zoom:80%;" />

应当注意，在输入的过程中，主要会经历以下步骤：

- **维度压缩**: 将CNN backbone输出的 $C \times H \times W$ 维的feature map先用 $1 \times 1$ convolution 处理, 将channels数量从 $C$ 压缩到 $d$ ， 即得到 $d \times H \times W$ 维的新feature map;
- 序列化数据转化: 将空间的维度 (高和宽) 压缩为一个维度，即把上一步得到的 $d \times H \times W$ 维的feature map通过reshape成 $d \times H W$ 维的feature map;
- 加上positoin encoding: 由于transformer模型是顺序无关的，而 $d \times H W$ 维feature map中 $H W$​ 维度显然与原图的位置有关，所以需要加上position encoding反映位置信息。

针对位置信息，需要为每个特征的位置与通道进行编码，其公式如下所示：
$$
PE_{(pos,2i)}=sin(pos/10000^{2i/d_{model}})
$$

$$
PE_{(pos,2i+1)}=cos(pos/10000^{2i/d_{model}})
$$

其中$i$是通道的下标，$pos$是位置的下标，$d_{model}$为特征的总通道数；通过这一操作，即可以分别表示奇偶数通道的基本特征。

##### Decoder

其结构基本上和原始的Transformer相同，

主要有两个输入：

1. image embedding (由Transformer Encoder输出) 与 position encoding 之和;
2. object queries

Object queries有 $N$​​ 个（$N$  是一个预先设定的、远远大于待检测的目标数目的整数），输入Transformer Decoder后分别得到 $N$  个decoder output embedding，经过FFN处理后就得到了 $N$​​ 个预测的boxes和这些boxes的类别。

具体实现上，object queries是$N$​​  个learnable embedding，训练刚开始时可以随机初始化。在训练过程中，因为需要生成不同的boxes，object queries会被迫使变得不同来反映位置信息。

##### FFN

在Decoder部分后，通过两种FFN分别预测bounding box的中心位置、高和宽，与class标签。

### Loss Function

可以得知，Transformer生成 $N$个prediction boxes，但ground truth boxes 数量将会远远小于$N$，在这一情况下，无法做到预测与实际一一对应，因此为了完成匹配，引入了一个新类别$\phi$用于表示**没有物体**，在这一情况下，多出来的 $N-m$ 个prediction embedding就会和$\phi$​类别配对。这样就可以将prediction boxes和image objects的配对看作两个等量的集合二分图匹配了。

在这一情况下，定义好cost后即可使用`匈牙利算法`进行快速的二分图匹配。

每对prediction box和image object匹配的cost $L_{\text {maich }}$​ 定义如下 
$$
-\mathbb{1}\left\{c_{i} \neq \varnothing\right\} \hat{p}_{\sigma}(i)\left(c_{i}\right)+\mathbb{1}_{\left\{c_{i} \neq \varnothing\right\}} \mathcal{L}_{\text {box }}\left(b_{i}, \hat{b}_{\sigma(i)}\right)
$$
其中,

- $c_{i}$ 为第 $i$ 个image object的class标签, $\sigma(i)$ 为与第 $i$ 个object配对的prediction box的 index;

- $1_{\left\{c_{i} \neq \phi\right\}}$ 是一个函数，当 $c_{i} \neq \phi$ 时为 1 , 否则为 0 ;

- $\hat{p}_{\sigma(i)}\left(c_{i}\right)$ 表示Transformer预测的第 $\sigma(i)$ 个prediction box类别为 $c_{i}$ 的概率;

- $b_{i}, \hat{b}_{\sigma(i)}$ 分别为第 $i$ 个image object的box和第 $\sigma(i)$ 个prediction box的位置;

- $L_{\text {bow }}\left(b_{i}, \hat{b}_{\sigma(i)}\right)$ 计算的是ground truth box和prediction box之间的差距

上式总结起来就是, 当配对的image object为 $\phi$ 时, 我们人为规定配对的cost $L_{\text {match }}=0$; 当 配对的image object为真实的物体 (即不为 $\phi$ ）时, 如果预测的prediction box类别和image object类别相同的概率越大 (越小)，或者两者的box差距越小 (越大)，配对的cost $L_{\text {match }}$ 越 小 (越大)。

$L_{b o x}$ 的计算公式如下:
$$
\lambda_{\mathrm{iOu}} \mathcal{L}_{\mathrm{iOu}}\left(b_{i}, \hat{\vec{b}}_{\sigma(i)}\right)+\lambda \mathrm{L}_{1}|| b_{i}-\hat{\vec{b}}_{\sigma(i)}||_{1}
$$
其中 $\left\|b_{i}-\hat{b}_{\sigma(i)}\right\|$ 为两个box中心坐标的L1距离。
**这样，我们就完全定义好了每对prediction box和 image object 配对时的cost。再利用匈牙利算 法即可得到二分图最优匹配。**

最后根据最优二分图匹配计算set prediction loss：
上述过程得到了prediction boxes和image objects之间的最优匹配。基于这个最优匹 配可以计算set prediction loss, 即评价Transformer生成这些prediction boxes的效果好坏。
Set prediction loss $L_{\text {Hungarian }}$​​​ 计算公式如下:
$$
\mathcal{L}_{\text {Hungarian }}(y, \hat{y})=\sum_{i=1}^{N}\left[-\log \hat{p}_{\hat{\sigma}(i)}\left(c_{i}\right)+\mathbb{1}_{\left\{c_{i} \neq \varnothing\right\}} \mathcal{L}_{\mathrm{box}}\left(b_{i}, \hat{b}_{\hat{\sigma}}(i)\right)\right]
$$
其中 $\hat{\sigma}$​​ 为最佳匹配, 将第 $i$​​ 个image object 匹配到第 $\hat{\sigma}(i)$​​ 个prediction box。 这个式子各个部 分的含义与 $L_{\text {match }}$​​ 计算公式完全一致。唯一需要注意的是，这里用的是 classification probability $\hat{p}_{\hat{\sigma}(i)}\left(c_{i}\right)$​​ 的对数形式, 而 $L_{\text {match }}$​​ 中直接用的线性形式；且这里考虑了 被匹配到 $\phi$​​ object的 $\hat{p}_{\hat{\sigma}(i)}\left(c_{i}\right)$​​, 而 $L_{\text {match }}$​​ 则直接定义为 0 .
用 $L_{\text {Hungarian }}$​​ 做反向传播即可优化Transformer。



#### object queries

