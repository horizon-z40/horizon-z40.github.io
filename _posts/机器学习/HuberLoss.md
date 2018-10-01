​    **Huber Loss** 相当于平方误差的推广，通过设置delta的值，使损失函数鲁棒性更强，从而减弱离群点（outliers）对模型的影响。当delta为无穷大时，Huber Loss 退化为Squared Loss.



Huber loss是为了增强平方误差损失函数（squared loss function）对噪声（或叫离群点，outliers）的鲁棒性提出的。