# Faster-RCNN

### RPN网络

其基本流程如下：

1. 在前期操作过程中，将输入的图像提取为某些特征图
2. 使用某一卷积核对此特征图进行卷积
3. 在滑动窗口的特征图上的每一个点上设置对应的锚点
4. 将结果分别输入到两个小的$1*1$的网络中reg（回归，目标框的位置）和cls（分类，确定该框中是否为目标）模块中
5. 训练集标记好了每个框的位置，和reg输出的框的位置比较，用梯度下降来训练网络

基于此，RPN的基本架构为：

<img src="https://i.loli.net/2021/07/15/woFcpEzMbALfJPQ.jpg" style="zoom:80%;" />

在RPN部分，其基本做法为：

不妨令特征图的大小为：$C*H*W$，即其中有单个特征图有$C$个通道，在此基础上，每一个点生成K个锚点，则可以生成$K*H*W$的输出锚点，然后则：

- 然后对 feature 进行卷积，输出 `cls_logits` 大小是 $K*N*W$ ，对应每个 anchor 是否有目标；
- 同时feature 进行卷积，输出 `bbox_pred` 大小是  $4K*N*W$  ，对应每个点的4个框位置回归信息 $(dx,dy,dw,dh)$;

<img src="https://i.loli.net/2021/07/15/Orq6nPy7hRZmuGd.png" style="zoom:100%;" />
在这一基础上，即可执行后续的处理：

对于二分类部分，在使用9个锚点的情况下，通过卷积的输出图像为$W*H*9$大小(也有资料说是18),，在这一基础上，通过一个softmax函数判定其为positive或者negative，从而筛选出positive的anchors。

对于回归部分，可以使用proposals进行bounding box regression，得到检测框。



在分别进行了回归与分类操作后，Proposal Layer负责综合**所有回归中学习的变换量**以及**positive anchors**，计算出精准的proposal，送入后续RoI Pooling Layer。



综上，RPN网络的作用即为**生成anchors -> softmax分类器提取positvie anchors -> bbox reg回归positive anchors -> Proposal Layer生成proposals**

