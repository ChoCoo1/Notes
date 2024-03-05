本章主要通过**概率近似正确**(Probably Approximately Correct ,PAC)学习框架对上述问题进行形式化并给出答案。PAC框架借助**样本复杂度**(sample complexity)，指的是欲达到近似解所需要的样本点数目，和学习算法的**时间空间复杂度**(time and space complexity)，依赖于概念类进行计算表示的代价，来定义可学习的概念类
## PAC学习模型
对于PAC模型相关概念以及符号的介绍：
- **样本**(sample)或**实例**(instance)集合为$\mathcal{X}$，有时也用来表示**输入空间**(input space)
- 所有可能的**标签**(label)或者**目标值**(target value)的集合记作$\mathcal{Y}$ ，在本章中将其限制在**二分类**(binary classification)，即$\mathcal{Y}=\{0,1\}$，之后会将其推广
- **概念**(concept)c： $\mathcal{X}\rightarrow \mathcal{Y}$ 指的是一个从 $\mathcal{X}$ 到 $\mathcal{Y}$ 的映射。我们可以从$\mathcal{X}$中对应标签为1的子集中**鉴别**（学习）概念c，例如一个概念可以是三角形哪点构成的集合或者标识空间中的点是否为三角形内点的函数，我们可以将待学习的概念为三角形
- **概念类**(concept class)指的是我们可能想要学习的概念构成的集合，记作$\mathcal{C}$
假定所有样本都是**独立同分布**的，并且服从的是某个固定但是未知的分布$\mathcal{D}$。学习器需要考虑的是一个固定的、由所有可能的概念组成的集合$\mathcal{H}$，称为**假设集**(hypothesis set)，并且$\mathcal{H}$和$\mathcal{C}$不是**必须一致**的。
学习器在学习的过程中，会得到分布$\mathcal{D}$的独立同分布样本集$S=(x_1,\dots,x_n)$ 以及对应标签集$(c(x_1),\dots,c(x_n))$ ，该标签集根据特定的、待学习的目标概念$c\in\mathcal{C}$ 得到，其任务就是利用带标签的样本集S，选择一个假设$h_S\in\mathcal{H}$ ，使其关于概念c有尽可能小的**泛化误差**(generalization error)，假设$h\in\mathcal{H}$ 的泛化误差（有时也称为**风险**（risk）或是**真实误差**(true error)，或者直接简称为**误差**(error)记为$R(h)$ 定义如下
### 定义2.1 泛化误差
给定一个假设 $h\in\mathcal{H}$ ，一个目标概念 $c\in\mathcal{C}$ ，以及一个潜在的分布$\mathcal{D}$，则 $h$ 的泛化误差或风险定义为:$$R(h) = \underset{x\sim\mathcal{D}}{\mathbb{P}}\left[h(x)\neq c(x) \right]=\underset{x\sim\mathcal{D}}{\mathbb{E}}\left[1_{h(x)\neq c(x)}\right]$$
其中，$1_\omega$指的是事件$\omega$的**示性函数**（依事件出现与否应取1和0的函数） ，需要注意的是，一个假设的泛化误差并不是直接可得的，由于分布和概念军事未知的，但学习器可以在有标签的样本集S上度量一个假设的**经验误差**(expirical error)
### 定义2.2 经验误差