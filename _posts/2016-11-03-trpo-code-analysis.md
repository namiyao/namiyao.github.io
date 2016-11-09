---
layout: post
title: TRPO算法与代码解析
description: "推导Trust Region Policy Optimization算法，结合Tensorflow代码讲解算法实现细节"
date: 2016-11-09
tags: []
comments: true
share: true
---

# Introduction
[Andrej Karpathy][Andrej Karpathy Blog]指出，Policy Gradients (PG)是default的Reinforcement Learning (RL)算法。文章[Benchmarking Deep Reinforcement Learning for Continuous Control][Benchmarking Article]指出，Truncated Natural Policy Gradient (TNPG)算法，Trust Region Policy Optimization (TRPO)算法，Deep Deterministic Policy Gradient (DDPG)算法取得了最好的实验结果。除此之外，文章中未提到的
[Asynchronous Advantage Actor-Critic (A3C)][A3C Artical]算法的表现也超过了DQN。以上四种算法均属于PG。

DDPG算法与代码解析参考[Deep Deterministic Policy Gradients in TensorFlow][DDPG Blog]。

TNPG与TRPO算法的区别仅在于TRPO用了[Backtracking line search][Backtracking line search wiki]来确定步长，确保目标函数有足够的优化，而TNPG并没有使用Backtracking line search。本文对TRPO算法与代码进行解析，TNPG只需要去掉Backtracking line search这一步即可。

关于TRPO算法的文章主要有两篇。文章[Trust Region Policy Optimization][TRPO Artical]提出了TRPO算法。文章[High-Dimensional Continuous Control using Generalized Advantage Estimation][TRPO GAE Artical]使用Generalized Advantage Estimator (GAE)改进了TRPO算法。

本文使用Wojciech Zaremba的基于Tensorflow的[代码][TRPO Code]。

# Theory and Code

## Policy Gradients
用函数来近似policy，记作 $\pi_{\theta}(a|s)$。目标函数为expected discounted reward，

$$J(\theta)=E_{\pi_{\theta}}[\sum_{t=0}^{\infty}\gamma^{t}r_{t}]$$

要最大化目标函数 $J(\theta)$，最直接的想法就是使用[mini-batch gradient descent optimization algorithms][gradient descent optimization algorithms Blog]，需要计算 $\nabla_{\theta}J(\theta)$。可是我们连 $J(\theta)$ 的解析表达式都没有啊喂，要怎么算导数啊喂！幸好我们有 **Policy Gradients Theorem**，

$$\nabla_{\theta}J(\theta)=E_{\pi_{\theta}}[\nabla_{\theta}log\pi_{\theta}(a|s)Q^{\pi_{\theta}}(s,a)]$$

用advantage function代替state-action value function，容易证明上式仍然成立，

$$\nabla_{\theta}J(\theta)=E_{\pi_{\theta}}[\nabla_{\theta}log\pi_{\theta}(a|s)A^{\pi_{\theta}}(s,a)]$$

其中   $A^{\pi_{\theta}}(s,a)=Q^{\pi_{\theta}}(s,a)-V^{\pi_{\theta}}(s)$。

现在 $\nabla_{\theta}J(\theta)$ 好算多了！只需要知道 $A^{\pi_{\theta}}(s,a)$ 就行了！记 $\hat A_{t}$ 为 $A^{\pi_{\theta}}(s_{t},a_{t})$ 的估计，则policy gradient estimator为

$$\widehat{\nabla_{\theta}J(\theta)}=\frac{1}{N}\sum_{n=1}^{N}\sum_{t=0}^{\infty}\hat A_{t}^{n}\nabla_{\theta}log\pi_\theta(a_{t}^{n}|s_{t}^{n})$$

一个方法是用REINFORCE算法估计$A^{\pi_{\theta}}(s,a)$。下一节使用另一个方法，用函数近似 $V^{\pi_{\theta}}(s)$ 来估计 $A^{\pi_{\theta}}(s,a)$。

## Advantage Function Estimation
类似于 $TD(\lambda)$ 方法，以下都是 $A^{\pi_{\theta}}(s_{t},a_{t})$ 的估计

$$\begin{align}
\hat A_{t}^{(1)}&=r_{t}+\gamma V_{\phi}(s_{t+1})-V_{\phi}(s_{t})\\
\hat A_{t}^{(2)}&=r_{t}+\gamma r_{t+1}+\gamma^{2} V_{\phi}(s_{t+2})-V_{\phi}(s_{t})\\
\hat A_{t}^{(3)}&=r_{t}+\gamma r_{t+1}+\gamma^{2} r_{t+2}+\gamma^{3} V_{\phi}(s_{t+3})-V_{\phi}(s_{t})\\
...\\
\hat A_{t}^{(k)}&=r_{t}+\gamma r_{t+1}+...+\gamma^{k-1} r_{t+k-1}+\gamma^{k} V_{\phi}(s_{t+k})-V_{\phi}(s_{t})\\
...\\
\hat A_{t}^{(\infty)}&=\sum_{l=0}^{\infty}\gamma^{l}r_{t+l}-V_{\phi}(s_{t}) \tag{1} \label{A_inf}
\end{align}$$

其中 $V_{\phi}(s_{t})$ 是value function $V^{\pi_{\theta}}(s_{t})$ 的函数近似。随着k的增加，估计 $\hat A_{t}^{(k)}$ 的variance增加，bias减小。

Generalized Advantage Estimator (GAE)是以上所有估计的exponentially-weighted average，记作 $\hat A_{t}^{GAE(\gamma,\lambda)}$，

$$\begin{align}
\hat A_{t}^{GAE(\gamma,\lambda)}&=(1-\lambda)(\hat A_{t}^{(1)}+\lambda \hat A_{t}^{(2)}+\lambda^3 \hat A_{t}^{(3)}+...)\\
&=\sum_{l=0}^{\infty}(\gamma\lambda)^{l}\delta_{t+l}^{V_{\phi}}
\end{align}$$

其中 $\delta_{t+l}^{V_{\phi}}=r_{t+l}+\gamma V_{\phi}(s_{t+l+1})-V_{\phi}(s_{t+l})$，用第二个等式可以方便的计算 $\hat A_{t}^{GAE(\gamma,\lambda)}$。容易看出 $\lambda=0$ 时，$\hat A_{t}^{GAE(\gamma,\lambda)}=\hat A_{t}^{(1)}$；$\lambda=1$ 时，$\hat A_{t}^{GAE(\gamma,\lambda)}=\hat A_{t}^{(\infty)}$。GAE通过exponentially-weighted average进行了bias-variance tradeoff，$\lambda$越大，后面的估计的权重越大，bias越小，variance越大。

以上分别用函数近似了policy和value function，这种方法叫做Actor-Critic算法。有了 $A^{\pi_{\theta}}(s_{t},a_{t})$ 的估计，就可以用policy gradient estimator $\widehat{\nabla_{\theta}J(\theta)}$ 来更新 $\pi_{\theta}(a\|s)$ 的参数 $\theta$。那么如何更新 $V_{\phi}(s)$ 的参数 $\phi$？最直观的想法是最小化L2损失

$$\min_{\phi}\sum_{n=1}^{N}\sum_{t=0}^{\infty}(\hat V(s_{t})-V_{\phi}(s_{t}))^2$$

其中 $\hat V(s_{t})=\sum_{l=0}^{\infty}\gamma^{l}r_{t+l}$。下面的代码使用$\eqref{A_inf}$ 来估计advantage function，使用[mini-batch gradient descent optimization algorithms][gradient descent optimization algorithms Blog]中的[Adam][Adam Blog]算法来更新参数 $\phi$。

```python
def discount(x, gamma):
    assert x.ndim >= 1
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def rollout(env, agent, max_pathlength, n_timesteps):
    paths = []
    timesteps_sofar = 0
    while timesteps_sofar < n_timesteps:
        obs, actions, rewards, action_dists = [], [], [], []
        ob = env.reset()
        agent.prev_action *= 0.0
        agent.prev_obs *= 0.0
        for _ in xrange(max_pathlength):
            action, action_dist, ob = agent.act(ob)
            obs.append(ob)
            actions.append(action)
            action_dists.append(action_dist)
            res = env.step(action)
            ob = res[0]
            rewards.append(res[1])
            if res[2]:
                path = {"obs": np.concatenate(np.expand_dims(obs, 0)),
                        "action_dists": np.concatenate(action_dists),
                        "rewards": np.array(rewards),
                        "actions": np.array(actions)}
                paths.append(path)
                agent.prev_action *= 0.0
                agent.prev_obs *= 0.0
                break
        timesteps_sofar += len(path["rewards"])
    return paths

class VF(object):
    coeffs = None

    def __init__(self, session):
        self.net = None
        self.session = session

    def create_net(self, shape):
        print(shape)
        self.x = tf.placeholder(tf.float32, shape=[None, shape], name="x")
        self.y = tf.placeholder(tf.float32, shape=[None], name="y")
        self.net = (pt.wrap(self.x).
                    fully_connected(64, activation_fn=tf.nn.relu).
                    fully_connected(64, activation_fn=tf.nn.relu).
                    fully_connected(1))
        self.net = tf.reshape(self.net, (-1, ))
        l2 = (self.net - self.y) * (self.net - self.y)
        self.train = tf.train.AdamOptimizer().minimize(l2)
        self.session.run(tf.initialize_all_variables())


    def _features(self, path):
        o = path["obs"].astype('float32')
        o = o.reshape(o.shape[0], -1)
        act = path["action_dists"].astype('float32')
        l = len(path["rewards"])
        al = np.arange(l).reshape(-1, 1) / 10.0
        ret = np.concatenate([o, act, al, np.ones((l, 1))], axis=1)
        return ret

    def fit(self, paths):
        featmat = np.concatenate([self._features(path) for path in paths])
        if self.net is None:
            self.create_net(featmat.shape[1])
        returns = np.concatenate([path["returns"] for path in paths])
        for _ in range(50):
            self.session.run(self.train, {self.x: featmat, self.y: returns})

    def predict(self, path):
        if self.net is None:
            return np.zeros(len(path["rewards"]))
        else:
            ret = self.session.run(self.net, {self.x: self._features(path)})
            return np.reshape(ret, (ret.shape[0], ))

···

self.vf = VF(self.session)

···

# Generating paths.
print("Rollout")
paths = rollout(
    self.env,
    self,
    config.max_pathlength,
    config.timesteps_per_batch)

# Computing returns and estimating advantage function.
for path in paths:
    path["baseline"] = self.vf.predict(path)
    path["returns"] = discount(path["rewards"],config.gamma)
    path["advant"] = path["returns"] - path["baseline"]

···

if self.train:
    self.vf.fit(paths)
```

[A3C][A3C Artical]算法使用 $\hat A_{t}^{(\infty)}$ 计算 policy gradient，使用mini-batch gradient descent optimization algorithms来更新policy参数和value function参数。

[TRPO+GAE][TRPO GAE Artical]算法使用 $\hat A_{t}^{GAE(\gamma,\lambda)}$ 计算 policy gradient，使用TRPO算法来更新policy参数，使用trust region method来更新value function参数。

下一节将结合代码详细讲解TRPO算法。

## Trust Region Policy Optimization
当使用[mini-batch gradient descent optimization algorithms][gradient descent optimization algorithms Blog]来更新参数时，需要给定learning rate，i.e. step size。PG算法中的step size的选取是极其重要的！因为step size决定了下一次抽样的策略函数，如果step size选的不好，下一次的mini-batch就会从很差的策略里产生。不同于supervise learning，因为训练样本早就确定了，即使这次step size步子扯大了，下个mini-batch还能扯回来。现在的难点在于PG算法只给出了梯度的估计，并没有目标函数可以用来进行[line search][line search wiki]以确定好的step size。借鉴[trust region method][trust region method video]的想法，如果可以给出目标函数 $J(\theta)$ 的局部近似函数，这个函数在trust region中是目标函数的一个很好的近似, 那就可以在trust region中最大化近似函数来更新参数 $\theta$。

令

$$L_{\theta_{old}}(\theta)=J(\theta_{old})+E_{\pi_{\theta_{old}}}[\frac{\pi_{\theta}(a|s)}{\pi_{\theta_{old}}(a|s)}A_{\pi_{\theta_{old}}}(s,a)]，$$

有

$$\begin{align}
L_{\theta_{old}}(\theta_{old})&=J(\theta_{old})+E_{\pi_{\theta_{old}}}[A_{\pi_{\theta_{old}}}(s,a)]  \\
&=J(\theta_{old})
\end{align}$$

$$\begin{align}
\nabla L_{\theta_{old}}(\theta)\big |_{\theta=\theta_{old}}&=E_{\pi_{\theta_{old}}}[\frac{\nabla\pi_{\theta}(a|s)\big |_{\theta=\theta_{old}}}{\pi_{\theta_{old}}(a|s)}A_{\pi_{\theta_{old}}}(s,a)] \\
&=E_{\pi_{\theta}}[\nabla log\pi_{\theta}(a|s)A_{\pi_{\theta}}(s,a)]\big |_{\theta=\theta_{old}} \\
&=\nabla J(\theta)\big |_{\theta=\theta_{old}}
\end{align}$$

所以 $L_{\theta_{old}}(\theta)$ 为 $J(\theta)$ 在 $\theta_{old}$ 附近的近似函数。

文章[Trust Region Policy Optimization][TRPO Artical]证明了如下定理,

$$J(\theta)\ge L_{\theta_{old}}(\theta)-CD_{KL}^{max}(\theta_{old},\theta)$$

其中，

$$C=\frac{2\epsilon\gamma}{(1-\gamma)^2},$$

$$D_{KL}^{max}(\theta_{old},\theta)=\max_{s}D_{KL}(\pi_{\theta_{old}}(·|s)\|\pi_{\theta}(·|s)).$$

想利用上面的定理来更新参数有如下困难。由于 $\gamma$ 通常取较大值，因而 penalty coefficient $C$ 会很大，导致step size非常小。解决方法是将penalty项转变成对[KL divergence][KL divergence wiki]的约束，即一个trust region，问题转化为，

$$\begin{align}
&\max_\theta L_{\theta_{old}}(\theta) \\
&\text{ subjec to }D_{KL}^{max}(\theta_{old},\theta)\le \delta.
\end{align}$$

观察约束条件，对状态空间中的每一个状态，都有KL divergence的约束，这么多约束条件在实际计算中是不可行的。直观上的一个解决办法是，使用average来代替max，将多个约束条件转换成一个约束条件，实验结果也表明这个代替有相似的实验表现，因此是可行的。用下面的带约束的优化问题来更新policy参数，

$$\begin{align}
&\max_\theta L_{\theta_{old}}(\theta) \\
&\text{ subjec to }D_{KL}^{ave(\theta_{old})}(\theta_{old},\theta)\le \delta.
\end{align}$$

其中，

$$D_{KL}^{ave(\theta_{old})}(\theta_{old},\theta)=E_{\pi_{\theta_{old}}}[D_{KL}\left (\pi_{\theta_{old}}(·|s)\|\pi_{\theta}(·|s)\right)].$$

在当前policy $\pi_{\theta_{old}}$ 抽样mini-batch trajectories来估计目标函数和约束函数，有

$$\hat L_{\theta_{old}}(\theta)=\frac{1}{N}\sum_{n=1}^{N}\sum_{t=0}^{\infty}\frac{\pi_\theta(a_{t}^{n}|s_{t}^{n})}{\pi_{\theta_{old}}(a_{t}^{n}|s_{t}^{n})}\hat A_{t}^{n}$$

$$\hat D_{KL}^{ave(\theta_{old})}(\theta_{old},\theta)=\frac{1}{N}\sum_{n=1}^{N}\sum_{t=0}^{\infty}D_{KL}\left (\pi_{\theta_{old}}(·|s_{t}^{n})\|\pi_{\theta}(·|s_{t}^{n})\right)$$

这里的 $\hat A_{t}^{n}$ 用 $\eqref{A_inf}$ 式计算，已在上一段面的代码中给出。

这个带约束的优化问题怎么解啊喂！文章教我们首先用目标函数的一阶近似和约束函数的二阶近似来计算 $\Delta\theta=\theta-\theta_{old}$ 的方向，然后用[Backtracking line search][Backtracking line search wiki]来确定step size，使得目标函数增大的同时满足约束条件。

KL divergence与Fisher information matrix有如下关系([证明][KL FIM])，

$$D_{KL}\left (\pi_{\theta_{old}}(·|s)\|\pi_{\theta}(·|s)\right)=\frac{1}{2}\Delta\theta^TI(\theta_{old})\Delta\theta+
\mathcal{O}(\|\Delta\theta\|^3)$$

其中，

$$I(\theta)=E_{\pi_\theta}[\nabla_\theta log\pi_\theta(·|s)\nabla_\theta log\pi_\theta(·|s)^T]$$

为Fisher information matrix。由上式知，KL divergence的一阶导为 $0$，$I(\theta)$ 等于KL divergence的Hessian矩阵。为什么用Hessian矩阵而不用Fisher information matrix来近似KL divergence呢？因为好计算啊！Tensorflow自带的求导功能使得Hessian矩阵的计算非常简单。

近似后的优化问题变为，

$$\begin{align}
&\max_{\Delta \theta} \Delta\theta^T\nabla_{\theta}L_{\theta_{old}}(\theta_{old}+\Delta \theta)\big |_{\theta=\theta_{old}}\\
&\text{ subjec to } \frac{1}{2}\Delta\theta^TH(\theta)\big |_{\theta=\theta_{old}}\Delta\theta\le \delta.
\end{align}$$

其中 $H(\theta)$ 是 $\hat D_{KL}^{ave(\theta_{old})}(\theta_{old},\theta)$ 的Hessian矩阵。

用Tensorflow的求导功能很容易计算目标函数一阶导和约束函数的Hessian矩阵，记作 $g$ 和 $A$。

```python
class TRPOAgent(object):

...

    def __init__(self, env):
        self.env = env
        if not isinstance(env.observation_space, Box) or \
           not isinstance(env.action_space, Discrete):
            print("Incompatible spaces.")
            exit(-1)
        print("Observation Space", env.observation_space)
        print("Action Space", env.action_space)
        self.session = tf.Session()
        self.end_count = 0
        self.train = True
        self.obs = obs = tf.placeholder(
            dtype, shape=[
                None, 2 * env.observation_space.shape[0] + env.action_space.n], name="obs")
        self.prev_obs = np.zeros((1, env.observation_space.shape[0]))
        self.prev_action = np.zeros((1, env.action_space.n))
        self.action = action = tf.placeholder(tf.int64, shape=[None], name="action")
        self.advant = advant = tf.placeholder(dtype, shape=[None], name="advant")
        self.oldaction_dist = oldaction_dist = tf.placeholder(dtype, shape=[None, env.action_space.n], name="oldaction_dist")

        # Create neural network.
        action_dist_n, _ = (pt.wrap(self.obs).
                            fully_connected(64, activation_fn=tf.nn.tanh).
                            softmax_classifier(env.action_space.n))
        eps = 1e-6
        self.action_dist_n = action_dist_n
        N = tf.shape(obs)[0]
        p_n = slice_2d(action_dist_n, tf.range(0, N), action)
        oldp_n = slice_2d(oldaction_dist, tf.range(0, N), action)
        ratio_n = p_n / oldp_n
        Nf = tf.cast(N, dtype)
        surr = -tf.reduce_mean(ratio_n * advant)  # Surrogate loss就是L
        var_list = tf.trainable_variables()
        kl = tf.reduce_sum(oldaction_dist * tf.log((oldaction_dist + eps) / (action_dist_n + eps))) / Nf
        ent = tf.reduce_sum(-action_dist_n * tf.log(action_dist_n + eps)) / Nf

        self.losses = [surr, kl, ent]
        self.pg = flatgrad(surr, var_list)
        # KL divergence where first arg is fixed
        # replace old->tf.stop_gradient from previous kl
        kl_firstfixed = tf.reduce_sum(tf.stop_gradient(
            action_dist_n) * tf.log(tf.stop_gradient(action_dist_n + eps) / (action_dist_n + eps))) / Nf
        grads = tf.gradients(kl_firstfixed, var_list)
        self.flat_tangent = tf.placeholder(dtype, shape=[None])
        shapes = map(var_shape, var_list)
        start = 0
        tangents = []
        for shape in shapes:
            size = np.prod(shape)
            param = tf.reshape(self.flat_tangent[start:(start + size)], shape)
            tangents.append(param)
            start += size
        gvp = [tf.reduce_sum(g * t) for (g, t) in zip(grads, tangents)]
        self.fvp = flatgrad(gvp, var_list)  
        self.gf = GetFlat(self.session, var_list)   #获取参数值
        self.sff = SetFromFlat(self.session, var_list)  #设置参数值
        self.vf = VF(self.session)  #
        self.session.run(tf.initialize_all_variables())        
```

用[拉格朗日乘子法和KKT条件][Lagrange Multiplier KKT]求解上面带约束的优化问题，容易得到解为 $\Delta \theta=\alpha_{max}\cdot s$，其中 $s=A^{-1}(-g)$ 是更新方向。$\alpha_{max}=\sqrt{\frac{2\delta}{s^TAs}}$ 是step size，这个step size是一阶近似目标函数的情况下得到的，是满足约束条件的最大的step size。可以用[Backtracking line search][Backtracking line search wiki] 缩小step size，使得原始目标函数 $\hat L_{\theta_{old}}(\theta_{old}+\Delta \theta)$ 有足够的优化，同时仍然满足约束条件。

观察 $s=A^{-1}(-g)$，要计算更新方向 $s$，必须计算Hessian矩阵 $A$  的逆，当参数特别多时（例如用neural network来表示policy时），$A$ 的维度很高，求逆计算不可行。要计算 $s$，即求解方程 $As=-g$。**Here comes the trick！** 用[conjugate gradient method][conjugate gradient method wiki] 近似求解方程，不需要求逆，只需要有个函数可以计算矩阵-向量乘积 $y\rightarrow Ay$ 即可。

下面是求解方程 $Ax=b$ 的[Conjugate Gradient Algorithm](https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_resulting_algorithm)，其中 $A$ 为实对称正定矩阵。

![Conjugate Gradient Algorithm][Conjugate Gradient Algorithm Image]

这是最常用的Conjugate Gradient Algorithm。算法的直观解释参见[Conjugate Gradient Method][Conjugate Gradient Method Video]和[Overview of Conjugate Gradient Method][Overview of Conjugate Gradient Method Video]，算法的推导过程参见[Derivation of the conjugate gradient method][Conjugate Gradient Derivation]。

```python
def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)
    for i in xrange(cg_iters):
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    return x

···

def fisher_vector_product(p):
    feed[self.flat_tangent] = p
    return self.session.run(self.fvp, feed) + config.cg_damping * p

g = self.session.run(self.pg, feed_dict=feed)
stepdir = conjugate_gradient(fisher_vector_product, -g)                  
```

有了更新方向 $s$，下面用[Backtracking line search][Backtracking line search wiki] 选取合适的step size，使得目标函数充分的增大。

1. 令 $\alpha=\alpha_{max}=\sqrt{\frac{2\delta}{s^TAs}}$，$\tau=0.5$，$c=0.1$
2. 重复 $\alpha\leftarrow \tau\alpha$ 直到 $\hat L_{\theta_{old}}(\theta_{old}+\alpha s)-\hat L_{\theta_{old}}(\theta_{old})\ge \alpha cm$，其中 $m=\nabla_\alpha \hat L_{\theta_{old}}(\theta_{old}+\alpha s)=s^T(-g)$
3. 返回 $\alpha$

```python
def linesearch(f, x, fullstep, expected_improve_rate):
    accept_ratio = .1
    max_backtracks = 10
    fval = f(x)
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        newfval = f(xnew)
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        if ratio > accept_ratio and actual_improve > 0:
            return xnew
    return x

···

shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
lm = np.sqrt(shs / config.max_kl)
fullstep = stepdir / lm
neggdotstepdir = -g.dot(stepdir)

def loss(th):
    self.sff(th)
    return self.session.run(self.losses[0], feed_dict=feed)

theta = linesearch(loss, thprev, fullstep, neggdotstepdir / lm)
self.sff(theta)
```

下面的代码是完整的训练过程。

```python
class TRPOAgent(object):

···

    def learn(self):
        config = self.config
        start_time = time.time()
        numeptotal = 0
        i = 0
        while True:
            # Generating paths.
            print("Rollout")
            paths = rollout(
                self.env,
                self,
                config.max_pathlength,
                config.timesteps_per_batch)

            # Computing returns and estimating advantage function.
            for path in paths:
                path["baseline"] = self.vf.predict(path)
                path["returns"] = discount(path["rewards"], config.gamma)
                path["advant"] = path["returns"] - path["baseline"]

            # Updating policy.
            action_dist_n = np.concatenate([path["action_dists"] for path in paths])
            obs_n = np.concatenate([path["obs"] for path in paths])
            action_n = np.concatenate([path["actions"] for path in paths])
            baseline_n = np.concatenate([path["baseline"] for path in paths])
            returns_n = np.concatenate([path["returns"] for path in paths])

            # Standardize the advantage function to have mean=0 and std=1.
            advant_n = np.concatenate([path["advant"] for path in paths])
            advant_n -= advant_n.mean()

            # Computing baseline function for next iter.

            advant_n /= (advant_n.std() + 1e-8)

            feed = {self.obs: obs_n,
                    self.action: action_n,
                self.advant: advant_n,
                    self.oldaction_dist: action_dist_n}


            episoderewards = np.array(
                [path["rewards"].sum() for path in paths])

            print "\n********** Iteration %i ************" % i
            if episoderewards.mean() > 1.1 * self.env._env.spec.reward_threshold:
                self.train = False
            if not self.train:
                print("Episode mean: %f" % episoderewards.mean())
                self.end_count += 1
                if self.end_count > 100:
                    break
            if self.train:
                self.vf.fit(paths)
                thprev = self.gf()

                def fisher_vector_product(p):
                    feed[self.flat_tangent] = p
                    return self.session.run(self.fvp, feed) + config.cg_damping * p

                g = self.session.run(self.pg, feed_dict=feed)
                stepdir = conjugate_gradient(fisher_vector_product, -g)
                shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
                lm = np.sqrt(shs / config.max_kl)
                fullstep = stepdir / lm
                neggdotstepdir = -g.dot(stepdir)

                def loss(th):
                    self.sff(th)
                    return self.session.run(self.losses[0], feed_dict=feed)
                theta = linesearch(loss, thprev, fullstep, neggdotstepdir / lm)
                self.sff(theta)

                surrafter, kloldnew, entropy = self.session.run(
                    self.losses, feed_dict=feed)
                if kloldnew > 2.0 * config.max_kl:
                    self.sff(thprev)

                stats = {}

                numeptotal += len(episoderewards)
                stats["Total number of episodes"] = numeptotal
                stats["Average sum of rewards per episode"] = episoderewards.mean()
                stats["Entropy"] = entropy
                exp = explained_variance(np.array(baseline_n), np.array(returns_n))
                stats["Baseline explained"] = exp
                stats["Time elapsed"] = "%.2f mins" % ((time.time() - start_time) / 60.0)
                stats["KL between old and new distribution"] = kloldnew
                stats["Surrogate loss"] = surrafter
                for k, v in stats.iteritems():
                    print(k + ": " + " " * (40 - len(k)) + str(v))
                if entropy != entropy:
                    exit(-1)
                if exp > 0.8:
                    self.train = False
            i += 1
```

至此TRPO算法全部推导完成啦！更多代码细节可以[查看完整代码][TRPO Code]。











[Andrej Karpathy Blog]:http://karpathy.github.io/2016/05/31/rl/
[Benchmarking Article]:https://arxiv.org/abs/1604.06778
[A3C Artical]:https://arxiv.org/abs/1602.01783
[DDPG Blog]:http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html
[TRPO Artical]:https://arxiv.org/abs/1502.05477
[TRPO GAE Artical]:https://arxiv.org/abs/1506.02438
[TRPO Code]:https://github.com/wojzaremba/trpo
[gradient descent optimization algorithms Blog]:http://sebastianruder.com/optimizing-gradient-descent/index.html
[trust region method video]:https://www.youtube.com/watch?v=P0Rhzv9GfYs
[line search wiki]:https://en.wikipedia.org/wiki/Line_search
[Backtracking line search wiki]:https://en.wikipedia.org/wiki/Backtracking_line_search
[KL FIM]:http://stats.stackexchange.com/questions/51185/connection-between-fisher-metric-and-the-relative-entropy
[Lagrange Multiplier KKT]:http://blog.csdn.net/huanongjingchao/article/details/17298569
[conjugate gradient method wiki]:https://en.wikipedia.org/wiki/Conjugate_gradient_method
[Conjugate Gradient Algorithm Image]:https://wikimedia.org/api/rest_v1/media/math/render/svg/e300dfefdbd374cdee765397528a65a5736a50d3
[Backtracking line search Blog]:http://www.cnblogs.com/kemaswill/p/3416231.html
[Conjugate Gradient Derivation]:https://en.wikipedia.org/wiki/Derivation_of_the_conjugate_gradient_method
[Conjugate Gradient Method Video]:https://www.youtube.com/watch?v=h4cG8jLGmKg&list=PLKP6-DnQsS-JfEjRhtDznV4L38znQ1PGw&index=9
[Overview of Conjugate Gradient Method Video]:https://www.youtube.com/watch?v=eAYohMUpPMA&list=PLKP6-DnQsS-JfEjRhtDznV4L38znQ1PGw&index=8
[Adam Blog]:http://sebastianruder.com/optimizing-gradient-descent/index.html#adam
[KL divergence wiki]:https://en.wikipedia.org/wiki/Kullback–Leibler_divergence
