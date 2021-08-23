# PyTorch

### 基本计算

在`PyTorch`框架下，使用着一个名为张量的基本数据结构，其可以完成从其它基本结构的一次转换过程，即：

```python
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
```

对于`numpy`，其可以使用`.numpy()`方法以及`from_numpy`方法完成基本的数据转换操作。

### 自动求导

神经网络（`NN`）是在某些输入数据上执行的嵌套函数的集合。 这些函数由*参数*（由权重和偏差组成）定义，这些参数在 `PyTorch` 中存储在张量中。

**正向传播**：在正向传播中，`NN` 对正确的输出进行最佳猜测。 它通过其每个函数运行输入数据以进行猜测。

**反向传播**：在反向传播中，`NN` 根据其猜测中的误差调整其参数。 它通过从输出向后遍历，收集有关函数参数（*梯度*）的误差导数并使用梯度下降来优化参数来实现。

在`autograd`方法中，其要点在于对该属性为true的属性执行自动求导计算的步骤：

在这一方法下，如果希望对某一张量进行方向计算，即可以直接对定义的函数执行一次backward()即可以完成整个链条上的求导操作：

```python
import torch
import math

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0")  # Uncomment this to run on GPU

# Create Tensors to hold input and outputs.
# By default, requires_grad=False, which indicates that we do not need to
# compute gradients with respect to these Tensors during the backward pass.
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# Create random Tensors for weights. For a third order polynomial, we need
# 4 weights: y = a + b x + c x^2 + d x^3
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Tensors during the backward pass.
a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y using operations on Tensors.
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute and print loss using operations on Tensors.
    # Now loss is a Tensor of shape (1,)
    # loss.item() gets the scalar value held in the loss.
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call a.grad, b.grad. c.grad and d.grad will be Tensors holding
    # the gradient of the loss with respect to a, b, c, d respectively.
    loss.backward()

    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        # Manually zero the gradients after updating weights
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
```

这即是自动求导过程的一个良好实例，在对损失函数进行求导后反向计算即可以更新输入。

更进一步，如果希望在网络中使用这一方法：

```python
import torch
import math

# Create Tensors to hold input and outputs.
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# For this example, the output y is a linear function of (x, x^2, x^3), so
# we can consider it as a linear layer neural network. Let's prepare the
# tensor (x, x^2, x^3).
p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)

# In the above code, x.unsqueeze(-1) has shape (2000, 1), and p has shape
# (3,), for this case, broadcasting semantics will apply to obtain a tensor
# of shape (2000, 3)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. The Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
# The Flatten layer flatens the output of the linear layer to a 1D tensor,
# to match the shape of `y`.
model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-6
for t in range(2000):

    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    y_pred = model(xx)

    # Compute and print loss. We pass Tensors containing the predicted and true
    # values of y, and the loss function returns a Tensor containing the
    # loss.
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

# You can access the first layer of `model` like accessing the first item of a list
linear_layer = model[0]

print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')
```

##### 整个计算步骤如下：

不妨令一个张量为$X$,其通过某一运算得到另一个张量$Y$，即：
$$
Y=f(X)=[y_1,y_2, ...y_m]
$$
这一情况下，即可以使用$Jacobbi$ 矩阵计算出$ \frac{\partial Y}{\partial X}$；

然后可以通过$Y$计算出某一损失函数$loss\_func$，且向量**v**恰好是标量损失**l**关于向量**Y**的梯度，如下：
$$
v=\left(\begin{array}{lll}
\frac{\partial l}{\partial y_{1}} & \cdots & \frac{\partial l}{\partial y}
\end{array}\right)^{T}
$$
向量v称为`grad_tensor`，并作为参数传递给`backward()` 函数。
为了得到损失的梯度**l**关于权重**X**的梯度，雅可比矩阵**J**是向量乘以向量**v**：


$$
J \cdot v=\left(\begin{array}{ccc}
\frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{1}} \\
\vdots & \ddots & \vdots \\
\frac{\partial y_{1}}{\partial x_{n}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
\end{array}\right)\left(\begin{array}{c}
\frac{\partial l}{\partial y_{1}} \\
\vdots \\
\frac{\partial l}{\partial y_{m}}
\end{array}\right)=\left(\begin{array}{c}
\frac{\partial l}{\partial x_{1}} \\
\vdots \\
\frac{\partial l}{\partial_{n}}
\end{array}\right)
$$




**需要注意的一件重要事情是，当调用`z.backward()`时，一个张量会自动传递为`z.backward(torch.tensor(1.0))`。`torch.tensor(1.0)`是用来终止链式法则梯度乘法的外部梯度。**这个外部梯度作为输入传递给`MulBackward`函数，以进一步计算**x**的梯度。传递到`.backward()`中的张量的维数必须与正在计算梯度的张量的维数相同。例如，如果梯度支持张量x和y如下：

```python
 x = torch.tensor([0.0, 2.0, 8.0], requires_grad = True)
 y = torch.tensor([5.0 , 1.0 , 7.0], requires_grad = True)
 z = x * y
```

然后，要计算`z`关于`x`或者`y`的梯度，需要将一个外部梯度传递给`z.backward()`函数，如下所示：

```python
 z.backward(torch.FloatTensor([1.0, 1.0, 1.0])
```

即，在计算过程中，需要将这一结果带入到具体的值计算过程中而非直接得出某一个公式。

### 数据集准备

数据集有两种形式进行存放：`Dataset`与`Dataloader`，在传入网络的时候，应当通过指定batch_size的方式，生成迭代器，将`Dataset`转化为`Dataloader`进行批梯度下降计算：

```python
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
```

而针对构造数据集的操作，需要构建出一个数据集的类：`Dataset`，其中应当包含以下方法：`init`， `len`，`getitem`；

init方法在数据集初始化时运行，完成对字典的初始化操作，对文件夹以及标注的路径进行初始化以及读取操作。

len方法得到数据集中数据的个数

getitem通过索引从数据集返回一个样本，并从中读取到数据的内容以及标注，并返回两值。

```python
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```

在完成`Dataset`转化为`Dataloader`的操作后，即可以通过迭代器对数据集进行遍历，且每一次都会得到一个批次的图像与标注，且每一次都会随机生成批次。

### 网络

网络将会决定数据的处理方式，而所有的网络的，都将会继承于`torch.nn`类中，它们都将会继承`torch.module`对象，并由多个层的复杂结构组成。

```python
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
```

其定义方式即为继承`nn.Module`类，并通过`init`方法进行定义，而使用`forward`函数进行数据的流动过程。

在这一过程中，可以使用`nn.Sequential`对一个模块中的网络进行打包，即在包内，数据按照既定的顺序进行流动。

### 优化方法

在优化中，需要定义损失函数以及所使用的优化器以及其参数，在这里，可以将train以及validation写为两部分，其做法如下：

```python
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

在这一基础上，对其进行调用，完成训练过程：

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
```

