Hyperopt 设计伊始,是包括基于高斯过程与回归树的贝叶斯优化算法的,但是现在这些都还没有被实现.



同时,Hyperopt所有的算法都可以通过[MongoDB](https://link.zhihu.com/?target=http%3A//www.mongodb.org/)进行串行或者并行计算.



目前主要由以下搜索方法：

- 随机搜索
- 模拟退火
- Tree of Parzen Estimators (TPE)

