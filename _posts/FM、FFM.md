这个不错，还包含了工具包介绍：http://mufool.com/2017/11/20/fm/





FM有一个衍生算法FFM（Field-aware FM），大概思路是将特征进行分组，学习出更多的隐式向量V，FM可以看做FFM只有一个分组的特例。FFM复杂度比较高，比较适合高度稀疏数据；而FM可以应用于非稀疏数据，更加通用。





#### FM模型原理

二阶FM模型表达式：
$$
y(x) := w_0 + \sum_{i=1}^nw_ix_i + \sum_{i=1}^n\sum_{j=i+1}^nx_ix_j
$$
