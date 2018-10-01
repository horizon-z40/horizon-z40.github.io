KKT条件是在满足一些有规则的条件下，一个非线性规划问题能有最优化解法的一个必要和充分条件。这是一个广义化拉格朗日乘数的成果。

一般情况下，最优化问题会碰到以下三种情况：

##### （1）无约束条件

解决方法通常是对变量求导，导数为0的点可能是极值点。

##### （2）等式约束条件

$$
\text{min} \quad f(x)\\
s.t.  \quad h_k(x)=0 \qquad k=1,2,\cdots,l
$$

则解决方法是消元法或者拉格朗日法.

##### （3）不等式约束条件

$$
\text{min} \quad f(X) \\
s.t. \quad h_j(X)=0 \qquad j=1,2,\cdots,p \\
\qquad g_k(X) \le 0 \qquad k=1,2,\cdots,q
$$

则可以定义拉格朗日函数：
$$
L(X,\lambda,\mu) = f(X) ＋\sum_{j=1}^{p} \lambda_j h_j(X) + \sum_{k=1}^q \mu_k g_k(X)
$$

##### （4）KKT条件

此时若要求解上述优化问题，必须满足下述条件（也是我们的求解条件）：
$$
\begin{align}
&\frac{\partial{L}}{\partial{X}} \Big{|}_{X=X^*} = 0 \\
&\lambda_j \neq 0 \\
&\mu_k \ge 0 \\
&\mu_k g_k(X^*)=0 \\
& h_j(X^*)=0 \quad j=1,2,\cdots,p \\
&g_k(X^*) \le 0 \quad k=1,2,\cdots,q
\end{align}
$$
(4)是对拉格朗日函数取极值时候带来的一个必要条件

(5)是拉格朗日系数约束（等式情况）(6)是不等式约束情况

(7)是互补松弛条件

(8)、(9)是原约束条件。



【参考资料】：https://blog.csdn.net/johnnyconstantine/article/details/46335763