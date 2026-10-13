# 训练集 EDA 解读

仅分析训练集 8,125 条；验证和测试未用于探索或决定特征。

训练集最多类 DERMASON=2127，最少类 BOMBAY=313；数量比为 6.80。因此主要比较采用等权 Macro-F1，并报告逐类 Recall。

尺寸与形状特征存在强相关；相关性图数值按两位小数显示，1.00 不代表两个特征完全相同。最高绝对相关的五组如下：

Area / ConvexArea: Pearson r=0.999940

Compactness / ShapeFactor3: Pearson r=0.998683

Perimeter / EquivDiameter: Pearson r=0.991365

AspectRation / Compactness: Pearson r=-0.987649

ConvexArea / EquivDiameter: Pearson r=0.985154

原始特征尺度差异明显，LR / MLP 使用仅训练集拟合的 StandardScaler；RF 使用原数值。

按照已固定设计保留全部原始特征，不依据测试表现删列或添加 PCA。离群值不默认删除；箱线图仅隐藏散点以便观察主体分布。