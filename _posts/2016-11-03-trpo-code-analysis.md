---
layout: post
title: TRPO算法与代码解析
---

# Introduction
[Andrej Karpathy][Andrej Karpathy Blog]指出，Policy Gradients (PG)是default的Reinforcement Learning (RL)算法。文章[Benchmarking Deep Reinforcement Learning for Continuous Control][Benchmarking Article]指出，Truncated Natural Policy Gradient (TNPG)算法，Trust Region Policy Optimization (TRPO)算法，Deep Deterministic Policy Gradient (DDPG)算法取得了最好的实验结果。除此之外，文章中未提到的
[Asynchronous Advantage Actor-Critic (A3C)][A3C Artical]算法的表现也超过了DQN。以上四种算法均属于PG。

DDPG算法与代码解析参考[Deep Deterministic Policy Gradients in TensorFlow][DDPG Blog]。

TNPG与TRPO算法的区别仅在于TRPO用了Backtracking line search来确定步长，从而使目标函数有足够的优化，而TNPG并没有使用Backtracking line search。本文对TRPO算法与代码进行解析，TNPG只需要去掉Backtracking line search这一步即可。

关于TRPO算法的文章主要有两篇。文章[Trust Region Policy Optimization][TRPO Artical]提出了TRPO算法。文章[High-Dimensional Continuous Control using Generalized Advantage Estimation][TRPO GAE Artical]使用Generalized Advantage Estimator (GAE)改进了TRPO算法。

本文使用Wojciech Zaremba的基于Tensorflow的[代码][TRPO Code]。

# Start with some Theory

## Policy Gradients
我们用函数来近似策略函数，记作 $\pi_{\theta}(a|s)$。目标函数为expected discounted reward，

$$J(\theta)=E[\sum_{t=0}^{\infty}\gamma^{t}r_{t}]$$
要最大化目标函数，最直接的想法就是使用梯度下降算法


## Advantage Function Estimation


# Show me the Code!

















[Andrej Karpathy Blog]:http://karpathy.github.io/2016/05/31/rl/
[Benchmarking Article]:https://arxiv.org/abs/1604.06778
[A3C Artical]:https://arxiv.org/abs/1602.01783
[DDPG Blog]:http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html
[TRPO Artical]:https://arxiv.org/abs/1502.05477
[TRPO GAE Artical]:https://arxiv.org/abs/1506.02438
[TRPO Code]:https://github.com/wojzaremba/trpo
