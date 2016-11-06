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
我们用函数来近似policy，记作 $\pi_{\theta}(a|s)$。目标函数为expected discounted reward，

$$J(\theta)=E[\sum_{t=0}^{\infty}\gamma^{t}r_{t}]$$

要最大化目标函数 $J(\theta)$，最直接的想法就是使用梯度下降算法，需要计算 $\nabla_{\theta}J(\theta)$。这个看起来超难计算的！幸好我们有 *Policy Gradients Theorem*，

$$\nabla_{\theta}J(\theta)=E_{\pi_{\theta}}[\nabla_{\theta}log\pi(a|s)Q^{\pi_{\theta}}(s,a)]$$

用advantage function代替state-action value function，容易证明上式仍然成立

$$\nabla_{\theta}J(\theta)=E_{\pi_{\theta}}[\nabla_{\theta}log\pi_{\theta}(a|s)A^{\pi_{\theta}}(s,a)]$$

其中 $A^{\pi_{\theta}}(s,a)=Q^{\pi_{\theta}}(s,a)-V^{\pi_{\theta}}(s)$。

现在 $\nabla_{\theta}J(\theta)$ 好算多了！我们只需要知道 $A^{\pi_{\theta}}(s,a)$ 就行了！记 $\hat A_{t}$ 为 $A^{\pi_{\theta}}(s_{t},a_{t})$ 的估计，则policy gradient estimator为

$$\widehat{\nabla_{\theta}J(\theta)}=\frac{1}{N}\sum_{n=1}^{N}\sum_{t=0}^{\infty}\hat A_{t}^{n}\nabla_{\theta}log\pi_\theta(a_{t}^{n}|s_{t}^{n})$$

一个方法是用REINFORCE算法通过a batch of trajectories直接估计$A^{\pi_{\theta}}(s,a)$。下一节我们用函数近似方法来估计 $A^{\pi_{\theta}}(s,a)$。

## Advantage Function Estimation
类似于 $TD(\lambda)$ 方法，以下都是 $A^{\pi_{\theta}}(s_{t},a_{t})$ 的估计

$$\begin{align}
\hat A_{t}^{(1)}&=r_{t}+\gamma V_{\phi}(s_{t+1})-V_{\phi}(s_{t})\\
\hat A_{t}^{(2)}&=r_{t}+\gamma r_{t+1}+\gamma^{2} V_{\phi}(s_{t+2})-V_{\phi}(s_{t})\\
\hat A_{t}^{(3)}&=r_{t}+\gamma r_{t+1}+\gamma^{2} r_{t+2}+\gamma^{3} V_{\phi}(s_{t+3})-V_{\phi}(s_{t})\\
...\\
\hat A_{t}^{(k)}&=r_{t}+\gamma r_{t+1}+...+\gamma^{k-1} r_{t+k-1}+\gamma^{k} V_{\phi}(s_{t+k})-V_{\phi}(s_{t})\\
...\\
\hat A_{t}^{(\infty)}&=\sum_{l=0}^{\infty}\gamma^{l}r_{t+l}-V_{\phi}(s_{t}) \tag{1}\label{A_inf}
\end{align}$$

其中 $V_{\phi}(s_{t})$ 是value function $V^{\pi_{\theta}}(s_{t})$ 的函数近似。随着k的增加，估计的variance增加，bias减小。

Generalized Advantage Estimator (GAE)是使用以上估计的exponentially-weighted average，记作 $\hat A_{t}^{GAE(\gamma,\lambda)}$，

$$\begin{align}
\hat A_{t}^{GAE(\gamma,\lambda)}&=(1-\lambda)(\hat A_{t}^{(1)}+\lambda \hat A_{t}^{(2)}+\lambda^3 \hat A_{t}^{(3)}+...)\\
&=\sum_{l=0}^{\infty}(\gamma\lambda)^{l}\delta_{t+l}^{V_{\phi}}
\end{align}$$

其中 $\delta_{t+l}^{V_{\phi}}=r_{t}+\gamma V_{\phi}(s_{t+1})-V_{\phi}(s_{t})$，用第二个等式可以方便的计算 $\hat A_{t}^{GAE(\gamma,\lambda)}$。容易看出 $\lambda=0$ 时，$\hat A_{t}^{GAE(\gamma,\lambda)}=\hat A_{t}^{(1)}$；$\lambda=1$ 时，$\hat A_{t}^{GAE(\gamma,\lambda)}=\hat A_{t}^{(\infty)}$。GAE通过exponentially-weighted average进行了bias-variance tradeoff，$\lambda$越大，后面的估计的权重越大，bias越小，variance越大。

以上我们分别用函数近似了policy和value function，这种方法叫做Actor-Critic算法。我们通过policy gradient estimator $\widehat{\nabla_{\theta}J(\theta)}$ 来更新 $\pi_{\theta}(a\|s)$ 的参数 $\theta$。那么如何更新 $V_{\phi}(s)$ 的参数 $\phi$？最直观的想法是最小化L2损失

$$\min_{\phi}\sum_{n=1}^{N}\sum_{t=0}^{\infty}(\hat V(s_{t})-V_{\phi}(s_{t}))^2$$

其中 $\hat V(s_{t})=\sum_{l=0}^{\infty}\gamma^{l}r_{t+l}$。可以通过梯度下降算法或者trust region算法来更新 $\phi$。

[A3C][A3C Artical]算法使用 $\hat A_{t}^{(\infty)}$ 计算 policy gradient，然后用梯度下降算法来更新policy参数；并使用梯度下降算法来更新value function参数。

[TRPO+GAE][TRPO GAE Artical]算法使用 $\hat A_{t}^{GAE(\gamma,\lambda)}$ 计算 policy gradient，然后用TRPO算法来更新policy参数；并使用trust region算法来更新value function参数。


# Show me the Code!

下一节的代码解析使用

$\eqref{A_inf}$















[Andrej Karpathy Blog]:http://karpathy.github.io/2016/05/31/rl/
[Benchmarking Article]:https://arxiv.org/abs/1604.06778
[A3C Artical]:https://arxiv.org/abs/1602.01783
[DDPG Blog]:http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html
[TRPO Artical]:https://arxiv.org/abs/1502.05477
[TRPO GAE Artical]:https://arxiv.org/abs/1506.02438
[TRPO Code]:https://github.com/wojzaremba/trpo
