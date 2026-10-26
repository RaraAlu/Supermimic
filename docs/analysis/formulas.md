# SuperMimic公式精讲

这份文档把训练里用到的每一个公式拆开来讲，面向想从头弄懂的读者。每个公式都按同样的顺序展开：

1. **一句话直觉**：它想解决什么问题；
2. **公式拆分**：先写小块，再拼成整体；
3. **符号说明**：每个字母代表什么、单位是什么；
4. **数值算例**：代入真实参数算一遍；
5. **代码位置**：在本仓库哪里实现。

- 依据：论文arXiv:2610.28480v1（标注节号），以及本仓库实现（标注文件路径）。数值参数取自论文表A.5与mjlab 1.6.0内置G1配置。
- 相关：[algorithm.md](algorithm.md)讲论文的整体思路，[training_pipeline.md](training_pipeline.md)讲训练流程。
  本文只讲公式本身，三份可以对照阅读。

## 0. 符号速查

| 符号 | 含义 | 典型值或维度 |
| --- | --- | --- |
| $t$ | 控制步序号 | 每步 $\Delta t = 0.02$ s（50 Hz） |
| $s_t$ | 仿真里的真实状态 | 71维：根位置、根朝向、根速度、关节角、关节速度 |
| $\hat{s}_t$ | 世界模型预测的状态（戴帽子表示“模型的估计”） | 同上 |
| $a_t$ | 策略输出的动作 | 29维，每个关节一个数 |
| $o_t$ | 机器人能“感觉到”的观测 | 99维 |
| $\bar{o}_t$ | 不含传感器噪声的观测 | 99维 |
| $\eta_t$ | 传感器噪声 | 每维均匀分布 |
| $\epsilon_t$ | 采样时加在动作上的噪声 | $\mathcal{N}(0, 0.1^2)$ |
| $g_t$ | 目标：未来10帧参考动作 | 730维 |
| $\xi$ | 仿真上下文：摩擦、关节标定偏置、躯干质心偏移 | 33维，每回合固定 |
| $f_\omega$ | 世界模型（参数为 $\omega$ ） | MLP(2048, 2048, 2048) |
| $\pi_\theta$ | 策略（参数为 $\theta$ ） | 编码器加解码器 |
| $V_\psi$ | 价值函数critic（参数为 $\psi$ ） | MLP(256, 128) |
| $r_{t+1}$ | 执行 $a_t$ 后得到的奖励 | 每步最多0.1 |
| $d_t$ | 是否终止（摔倒或偏离太远） | 0或1；世界模型里是0到1的概率 |
| $c_t$ | 存活权重：活到第 $t$ 步的程度 | 0到1 |
| $\gamma$ | 折扣因子 | 0.99 |
| $\lambda$ | GAE混合系数 | 0.95 |
| $H$ | 想象推演的步数 | 32 |
| $k$ | 世界模型训练窗口的步数 | 8 |

## 1. 时间、回报与折扣

### 1.1 一个控制步

**直觉**：策略每0.02 s做一次决定；在这0.02 s里，仿真器把物理积分成4个小步，每个5 ms。

$$
\Delta t = 0.02\ \mathrm{s}, \qquad \text{一个控制步} = 4 \times 5\ \mathrm{ms}
$$

### 1.2 回报：把一串奖励加起来

**直觉**：强化学习的目标不是让“这一步”最好，而是让“从现在开始的一整段”最好。
回报就是把未来的奖励加起来，但越远的奖励打的折扣越大。

$$
G_t = r_{t+1} + \gamma\, r_{t+2} + \gamma^2\, r_{t+3} + \cdots = \sum_{i=0}^{\infty} \gamma^i\, r_{t+1+i}
$$

- $r_{t+1}$ ：在第 $t$ 步动作之后立即拿到的奖励；
- $\gamma^i$ ：第 $i$ 步之后那份奖励的“打折系数”。

### 1.3 折扣的直觉：有效视野

$\gamma = 0.99$ 时，各步的打折系数如下：

| 往后第几步 | 对应时间 | $\gamma^i$ |
| --- | --- | --- |
| 10 | 0.2 s | 0.904 |
| 32（想象长度 $H$ ） | 0.64 s | 0.725 |
| 100 | 2 s | 0.366 |
| 500（一个回合） | 10 s | 0.007 |

经验上把 $1/(1-\gamma)$ 叫作“有效视野”：

$$
\frac{1}{1 - \gamma} = \frac{1}{0.01} = 100\ \text{步} = 2\ \mathrm{s}
$$

意思是：策略主要关心未来大约2秒内发生的事。

### 1.4 满分回报

**直觉**：如果每一步都拿满分0.1，一直拿下去，回报最多是多少？

这是等比数列求和：

$$
G_{\max} = \sum_{i=0}^{\infty} \gamma^i \times 0.1 = \frac{0.1}{1 - \gamma} = \frac{0.1}{0.01} = 10
$$

这个10会在第8节的重置采样里用到，作为“回报做到多好算满分”的尺子。

代码：`supermimic/tasks/tracking/env.py`中的`self.G_max`。

## 2. 奖励函数

奖励回答一个问题：**这一步机器人模仿参考动作模仿得有多像？** 它由6个“像不像”的打分项和2个“别乱来”的惩罚项组成。

### 2.1 基本零件：指数打分核

**直觉**：把“误差”变成“0到1之间的分数”。误差为0得满分1，误差越大分数越接近0，但永远不会是负数。

$$
\kappa(e;\, \sigma) = \exp\left( -\frac{e}{\sigma^2} \right)
$$

- $e$ ：平方误差（例如位置差的平方，单位 $\mathrm{m}^2$ ）；
- $\sigma$ ：容忍尺度，与误差同单位（例如米）。它决定“差多少开始明显扣分”。

**算例**（身体位置， $\sigma = 0.3$ m）：

| 平均位置偏差 | 平方误差 $e$ | 分数 $\kappa$ |
| --- | --- | --- |
| 0 | 0 | 1.000 |
| 0.1 m | 0.01 | 0.895 |
| 0.3 m（正好等于 $\sigma$ ） | 0.09 | 0.368 |
| 0.6 m | 0.36 | 0.018 |

规律：偏差等于 $\sigma$ 时，分数约为 $e^{-1} \approx 0.37$ 。

### 2.2 先把参考动作“挪到机器人身边”

**直觉**：参考动作是在原地录的，机器人可能已经往旁边漂了几厘米、或者朝向偏了一点。
如果直接比较世界坐标，机器人就算姿势完全正确也会因为“站的位置不对”而被扣分。
所以先把参考动作平移、旋转到机器人当前所在的位置和朝向，再比较姿势。

第一步，算出参考与机器人之间只差多少偏航（绕竖直轴的旋转）：

$$
\Delta R = \mathrm{yaw}\left( R\, R_{\mathrm{ref}}^\top \right)
$$

- $R$ 、 $R_{\mathrm{ref}}$ ：机器人和参考的根（骨盆）朝向，3×3旋转矩阵；
- $\mathrm{yaw}(\cdot)$ ：只保留绕竖直轴的那部分旋转，去掉前后、左右的倾斜。

第二步，确定锚点：水平位置用机器人的，高度用参考的：

$$
p_{\mathrm{anchor}} = \left( p_x,\ p_y,\ p_{\mathrm{ref},z} \right)
$$

第三步，把参考的每个身体部位 $b$ 搬过来：

$$
p_b^{\mathrm{des}} = p_{\mathrm{anchor}} + \Delta R \left( p_{\mathrm{ref},b} - p_{\mathrm{ref}} \right), \qquad R_b^{\mathrm{des}} = \Delta R\, R_{\mathrm{ref},b}
$$

括号里是“这个部位相对参考骨盆在哪”，乘 $\Delta R$ 把它转到机器人的朝向，再加到锚点上。

代码：`supermimic/tasks/tracking/mdp.py`中`MDP.reward`的`dR`、`anchor`、`ref_bp`、`ref_bR`。

### 2.3 六个“像不像”的打分项

跟踪的身体部位集合 $\mathcal{T}$ 有14个：骨盆、左右髋、膝、踝、躯干、左右肩、肘、腕。 $\lvert \mathcal{T} \rvert = 14$ 。

**(1) 身体位置**：各部位位置差的平方，取平均。

$$
\bar{e}_{\mathrm{pos}} = \frac{1}{14} \sum_{b \in \mathcal{T}} \left\lVert p_b^{\mathrm{des}} - p_b \right\rVert^2, \qquad r_{\mathrm{pos}} = \kappa(\bar{e}_{\mathrm{pos}};\ 0.3)
$$

**(2) 身体朝向**：用“两个朝向之间要转多少角度”衡量差别。

$$
\theta_b = \text{从}\ R_b\ \text{转到}\ R_b^{\mathrm{des}}\ \text{的角度}, \qquad \bar{e}_{\mathrm{ori}} = \frac{1}{14} \sum_{b} \theta_b^2, \qquad r_{\mathrm{ori}} = \kappa(\bar{e}_{\mathrm{ori}};\ 0.4)
$$

角度的算法：旋转矩阵 $M = R_b^{\mathrm{des}} R_b^\top$ 的迹满足 $\mathrm{tr}(M) = 1 + 2\cos\theta_b$ ，所以

$$
\theta_b = \arccos\left( \frac{\mathrm{tr}(M) - 1}{2} \right)
$$

$\sigma = 0.4$ rad约23°。

**(3) 身体线速度**、**(4) 身体角速度**：直接比较世界坐标下的速度（不做2.2的挪动）。

$$
\bar{e}_{\mathrm{lv}} = \frac{1}{14} \sum_{b} \left\lVert v_{\mathrm{ref},b} - v_b \right\rVert^2, \qquad r_{\mathrm{lv}} = \kappa(\bar{e}_{\mathrm{lv}};\ 1.0)
$$

$$
\bar{e}_{\mathrm{av}} = \frac{1}{14} \sum_{b} \left\lVert \omega_{\mathrm{ref},b} - \omega_b \right\rVert^2, \qquad r_{\mathrm{av}} = \kappa(\bar{e}_{\mathrm{av}};\ 3.14)
$$

**(5) 根位置**、**(6) 根朝向**：只看骨盆，而且**不做挪动**，直接比世界坐标。它们负责惩罚“整体漂走了”。

$$
e_{\mathrm{rp}} = \left\lVert p_{\mathrm{ref}} - p \right\rVert^2, \qquad r_{\mathrm{rp}} = \kappa(e_{\mathrm{rp}};\ 0.3)
$$

$$
e_{\mathrm{ro}} = \theta_{\mathrm{root}}^2, \qquad r_{\mathrm{ro}} = \kappa(e_{\mathrm{ro}};\ 0.4)
$$

六项加权求和：

$$
r^{\mathrm{track}} = r_{\mathrm{pos}} + r_{\mathrm{ori}} + r_{\mathrm{lv}} + r_{\mathrm{av}} + 0.5\, r_{\mathrm{rp}} + 0.5\, r_{\mathrm{ro}}
$$

| 项 | 衡量什么 | $\sigma$ | 权重 |
| --- | --- | --- | --- |
| 身体位置 | 姿势形状（各部位的相对位置） | 0.3 m | 1.0 |
| 身体朝向 | 各部位的转角 | 0.4 rad | 1.0 |
| 身体线速度 | 各部位移动快慢 | 1.0 m/s | 1.0 |
| 身体角速度 | 各部位转动快慢 | 3.14 rad/s | 1.0 |
| 根位置 | 整体有没有漂走 | 0.3 m | 0.5 |
| 根朝向 | 整体朝向对不对 | 0.4 rad | 0.5 |

全部满分时 $r^{\mathrm{track}} = 4 \times 1 + 2 \times 0.5 = 5$ 。

### 2.4 两个“别乱来”的惩罚项

**(7) 动作平滑**：相邻两步动作差得越多扣得越多，防止关节抖动。

$$
r_{\mathrm{rate}} = -0.1 \left\lVert a_t - a_{t-1} \right\rVert^2
$$

**(8) 关节软限位**：关节角超出“安全区”的部分要扣分。安全区是把机械限位向中间收缩10%：

$$
q_j^{\mathrm{mid}} = \frac{q_j^{\mathrm{lo}} + q_j^{\mathrm{hi}}}{2}, \qquad h_j = \frac{q_j^{\mathrm{hi}} - q_j^{\mathrm{lo}}}{2}
$$

$$
q_j^{\min} = q_j^{\mathrm{mid}} - 0.9\, h_j, \qquad q_j^{\max} = q_j^{\mathrm{mid}} + 0.9\, h_j
$$

$$
r_{\mathrm{lim}} = -2 \sum_{j=1}^{29} \left[ \max(0,\ q_j - q_j^{\max}) + \max(0,\ q_j^{\min} - q_j) \right]
$$

$\max(0, x)$ 的意思是：没超出时为0，不扣分；超出多少就扣多少。

两项相加：

$$
r^{\mathrm{reg}} = r_{\mathrm{rate}} + r_{\mathrm{lim}}
$$

### 2.5 合成：乘上时间步长

$$
r_{t+1} = \Delta t \left( r^{\mathrm{track}} + r^{\mathrm{reg}} \right)
$$

**为什么乘 $\Delta t$ ？** 这样“每秒能拿多少分”与控制频率无关：50 Hz下每步最多 $5 \times 0.02 = 0.1$ ，
一秒50步最多5分；如果换成100 Hz，每步最多0.05，一秒还是5分。整段回报就像在对时间做积分。

**算例**：某一步各部位平均偏差0.1 m（ $r_{\mathrm{pos}} = 0.895$ ），其余五项满分，动作变化 $\lVert a_t - a_{t-1} \rVert^2 = 0.5$ ，没有超限位：

$$
r_{t+1} = 0.02 \times \left( 0.895 + 1 + 1 + 1 + 0.5 + 0.5 - 0.1 \times 0.5 \right) = 0.02 \times 4.845 = 0.0969
$$

代码：`supermimic/tasks/tracking/mdp.py`中的`MDP.reward`；`MAX_STEP_REWARD`就是0.1。

## 3. 终止与存活权重

### 3.1 什么时候算“失败”

**直觉**：机器人骨盆高度差太多（比如摔倒或跳起来没跟上），或者身体歪得太厉害，就结束这一回合。

高度误差：

$$
e_z = \left\lvert z_{\mathrm{ref}} - z \right\rvert
$$

倾斜误差：

$$
e_{\mathrm{tilt}} = \left\lvert (R_{\mathrm{ref}})_{33} - R_{33} \right\rvert
$$

终止条件：

$$
d = \mathbb{1}\left[ e_z > 0.25\ \mathrm{m}\ \ \text{或}\ \ e_{\mathrm{tilt}} > 0.8 \right]
$$

**$R_{33}$ 是什么？** 旋转矩阵右下角那个数，等于“身体的竖直轴”和“世界的竖直轴”夹角的余弦：

$$
R_{33} = \cos\varphi, \qquad \varphi = \text{身体相对竖直方向倾斜的角度}
$$

参考动作站直时 $(R_{\mathrm{ref}})_{33} = \cos 0 = 1$ ，所以终止条件变成：

$$
1 - \cos\varphi > 0.8 \quad\Longleftrightarrow\quad \cos\varphi < 0.2 \quad\Longleftrightarrow\quad \varphi > 78.5^\circ
$$

也就是论文附录A说的“约78°”。

代码：`MDP.termination`。

### 3.2 软终止：把“是否摔倒”变成可导的概率

**直觉**：仿真里摔没摔倒是0或1，没法求导。想象推演需要梯度告诉策略“再歪一点就要摔了”，
所以用sigmoid把硬阈值抹平成一个平滑的斜坡。

sigmoid函数：

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

| $x$ | $\sigma(x)$ |
| --- | --- |
| −4 | 0.018 |
| −2 | 0.119 |
| 0 | 0.5 |
| 2 | 0.881 |
| 4 | 0.982 |

把误差减去阈值、再除以一个“斜坡宽度”，得到“这一关挂掉的概率”：

$$
p_z = \sigma\left( \frac{e_z - 0.25}{0.02} \right), \qquad p_{\mathrm{tilt}} = \sigma\left( \frac{e_{\mathrm{tilt}} - 0.8}{0.04} \right)
$$

两关都过才算活下来，所以存活概率是两个“过关概率”相乘，终止概率是1减去它：

$$
\hat{d} = 1 - (1 - p_z)(1 - p_{\mathrm{tilt}})
$$

**算例**：高度误差0.21 m（离阈值还差4 cm）， $x = (0.21 - 0.25)/0.02 = -2$ ， $p_z = 0.119$ ；
倾斜正常时 $p_{\mathrm{tilt}} \approx 0$ ，于是 $\hat{d} \approx 0.12$ 。
误差到0.25 m时 $\hat{d} = 0.5$ ；到0.29 m时约0.88。

代码：`MDP.soft_termination`（宽度参数`tz=0.02`、`tt=0.04`）。

### 3.3 存活权重：活着才能继续拿分

**直觉**：一旦摔倒，之后的奖励就拿不到了。存活权重 $c_t$ 表示“走到第 $t$ 步时还活着的程度”。

$$
c_0 = 1, \qquad c_{t+1} = c_t \left( 1 - d_{t+1} \right)
$$

展开就是逐步连乘：

$$
c_t = \prod_{i=1}^{t} \left( 1 - d_i \right)
$$

**算例**：如果每一步都有10%的终止概率：

$$
c_1 = 0.9, \quad c_2 = 0.81, \quad c_3 = 0.729
$$

在想象推演里 $\hat{d}$ 是可导的，所以 $\hat{c}$ 也可导：策略做出“更歪”的动作，会让后面所有步的 $\hat{c}$ 一起变小，
梯度就会把它往“站稳”的方向推。

代码：`supermimic/scripts/train.py`中`policy_update`的`alive = torch.cumprod(1 - d, 0)`。

## 4. 动作与观测

### 4.1 动作如何变成关节目标

**直觉**：策略不直接输出力矩，而是输出“关节想去的角度”相对默认姿势的偏移，再由PD控制器去追。

$$
q_j^{\mathrm{sp}} = q_j^0 + \Delta q_j^0 + \alpha_j\, a_j
$$

- $q_j^0$ ：默认姿势（膝盖微曲的站姿）；
- $\Delta q_j^0$ ：关节标定偏置（见4.2），属于仿真上下文 $\xi$ ；
- $\alpha_j$ ：动作缩放；
- $a_j$ ：策略输出。

PD控制器产生的力矩（忽略力矩上限）：

$$
\tau_j = K_{p,j} \left( q_j^{\mathrm{sp}} - q_j \right) - K_{d,j}\, \dot{q}_j
$$

刚度由电机的反射转动惯量（armature） $I_j$ 和10 Hz的固有频率决定，阻尼比取2：

$$
\omega_n = 2\pi \times 10 \approx 62.83\ \mathrm{rad/s}, \qquad K_{p,j} = I_j\, \omega_n^2, \qquad K_{d,j} = 2 \times 2 \times I_j\, \omega_n
$$

动作缩放让“动作为1”对应“静止时产生四分之一的最大力矩”：

$$
\alpha_j = \frac{0.25\, \tau_j^{\max}}{K_{p,j}}
$$

验证：关节静止在 $q^0 + \Delta q^0$ 时，动作 $a_j = 1$ 产生的力矩是 $K_{p,j} \alpha_j = 0.25\, \tau_j^{\max}$ 。

mjlab 1.6.0内置G1的实际数值：

| 电机（关节） | $I$ （kg·m²） | $K_p$ | $K_d$ | $\tau^{\max}$ （N·m） | $\alpha$ （rad） |
| --- | --- | --- | --- | --- | --- |
| 5020（肩、肘、腕滚转） | 0.003610 | 14.25 | 0.907 | 25 | 0.439 |
| 7520-14（髋俯仰、髋偏航、腰偏航） | 0.010178 | 40.18 | 2.558 | 88 | 0.548 |
| 7520-22（髋滚转、膝） | 0.025102 | 99.10 | 6.309 | 139 | 0.351 |
| 4010（腕俯仰、腕偏航） | 0.004250 | 16.78 | 1.068 | 5 | 0.075 |
| 双5020（踝、腰俯仰与滚转） | 0.007219 | 28.50 | 1.814 | 50 | 0.439 |

这些刚度很低（膝关节约99 N·m/rad），所以 $q^{\mathrm{sp}}$ 更像“力矩旋钮”而不是精确的位置指令，论文附录A也这样描述。

代码：`supermimic/tasks/tracking/env.py`中`TrackingSim.step`的`target`；增益来自mjlab的`g1_constants.py`。

### 4.2 关节标定偏置

**直觉**：真实机器人的编码器零点会有一点误差。仿真里每回合给每个关节随机一个小偏置
$\Delta q_j^0 \sim U[-0.01, 0.01]$ rad，并且把它同时加到目标和观测里：

$$
\text{目标：}\ q^0 + \Delta q^0 + \alpha a, \qquad \text{观测：}\ q - q^0 - \Delta q^0
$$

效果是机器人“以为”自己的关节角是 $q - \Delta q^0$ ，正如一个零点偏了的编码器。策略因此学会容忍这种误差。

### 4.3 观测与噪声

观测由6段拼成（论文表A.1的5段，加上实现额外的1段）：

| 片段 | 公式 | 维度 | 噪声半宽 |
| --- | --- | --- | --- |
| 关节角（相对默认姿势） | $q - q^0 - \Delta q^0$ | 29 | 0.01 rad |
| 关节速度 | $\dot{q}$ | 29 | 0.5 rad/s |
| 骨盆角速度（骨盆坐标系） | $R^\top \omega$ | 3 | 0.2 rad/s |
| 投影重力（骨盆坐标系） | $R^\top (0, 0, -1)^\top$ | 3 | 0.05 |
| 上一步动作 | $a_{t-1}$ | 29 | 无 |
| 参考相对朝向（实现额外加入） | $\mathrm{6D}(R^\top R_{\mathrm{ref}})$ | 6 | 0.05 |

**投影重力**是“重力方向在身体坐标系里看起来朝哪”，站直时为 $(0, 0, -1)$ ，向前倾时第一个分量变大，
它就是IMU能测到的“我歪没歪”。

噪声每一维独立地从均匀分布里抽：

$$
o = \bar{o} + \eta, \qquad \eta_i \sim U[-w_i,\ w_i]
$$

策略看到的是最近10帧观测拼成的“历史”，共 $10 \times 99 = 990$ 维。

**“6D”表示**：把旋转矩阵的前两列拼成6个数。它比四元数或欧拉角更适合神经网络，因为没有跳变和多值问题。

代码：`MDP.obs`、`MDP.add_noise`、`OBS_NOISE`。

### 4.4 目标：未来10帧参考

每一帧取73个数，全部在“当前参考帧的偏航坐标系”中表示，避免受整体朝向影响：

| 片段 | 维度 |
| --- | --- |
| 未来帧的根位置相对当前根 | 3 |
| 未来帧的根朝向（6D） | 6 |
| 未来帧的根线速度（根坐标系） | 3 |
| 未来帧的根角速度（根坐标系） | 3 |
| 未来帧的关节角 | 29 |
| 未来帧的关节速度乘0.1 | 29 |

$10 \times 73 = 730$ 维。乘0.1只是把关节速度缩到与其他量相近的数量级。到达段尾时，超出部分用最后一帧填充。

代码：`supermimic/tasks/tracking/motion.py`中的`MotionLib.goal`。

## 5. 世界模型

### 5.1 它要学什么

**直觉**：学会“给定现在的状态和动作，0.02秒后机器人会在哪、什么姿势、多快”。

$$
\hat{s}_{t+1} = f_\omega\left( \hat{s}_t,\ a_t,\ \xi \right)
$$

注意输入里**没有**目标 $g$ ：物理规律和机器人要模仿哪段舞蹈无关。这正是“一个模型服务整个动作库”的原因。

### 5.2 输入：只用“与站在哪、朝哪无关”的特征

**直觉**：机器人在房间左边还是右边、面朝东还是朝西，物理规律都一样。把这些无关信息去掉，模型学得更快。

$$
x = \left[\ z,\ \ \mathrm{6D}\left( H_{\mathrm{yaw}}^\top R \right),\ \ R^\top v,\ \ R^\top \omega,\ \ q,\ \ \dot{q}\ \right]
$$

| 片段 | 含义 | 维度 |
| --- | --- | --- |
| $z$ | 骨盆高度（水平位置被丢掉） | 1 |
| $\mathrm{6D}(H_{\mathrm{yaw}}^\top R)$ | 去掉偏航后的朝向，即只剩前后、左右倾斜 | 6 |
| $R^\top v$ 、 $R^\top \omega$ | 在骨盆坐标系里看的速度 | 3 + 3 |
| $q$ 、 $\dot{q}$ | 关节角与关节速度 | 29 + 29 |

共71维；再拼上动作29维和归一化后的 $\xi$ 33维，网络输入133维。

### 5.3 输入输出的归一化

**直觉**：关节角是零点几弧度，速度可能是几十，量纲差很多。先减均值、除标准差，让每一维都在差不多的尺度上。

$$
\tilde{x} = \frac{x - \mu_x}{\sqrt{\sigma_x^2 + \epsilon^2}}
$$

均值和方差在训练中滚动更新，累计2亿个样本后冻结。网络输出 $y$ 也是“归一化后的”，用相反的变换还原：

$$
\Delta = y \odot \sigma_\Delta + \mu_\Delta
$$

$\odot$ 表示逐元素相乘。

代码：`supermimic/tasks/tracking/rl/networks.py`中的`RunningNorm`、`WorldModel.step`。

### 5.4 输出：增量，并在局部坐标系里表示

**直觉**：与其让网络直接猜“下一时刻的绝对位置”，不如让它猜“这一小步变了多少”，而且是从机器人自己的视角来说。

网络输出70个数，切成六段：

$$
\Delta = \left( \Delta p,\ \Delta\theta,\ \Delta v,\ \Delta \omega,\ \Delta q,\ \Delta \dot{q} \right), \qquad \text{维度}\ 3 + 3 + 3 + 3 + 29 + 29 = 70
$$

逐项积分回世界坐标：

**位置**：位移在偏航坐标系里表示，所以先转回世界坐标再加。

$$
p' = p + H_{\mathrm{yaw}}\, \Delta p
$$

**朝向**： $\Delta\theta$ 是“在骨盆坐标系里转了多少”的旋转向量（方向是转轴，长度是角度），用指数映射变成旋转矩阵再右乘：

$$
R' = R\, \exp\left( [\Delta\theta]_\times \right)
$$

其中 $[\cdot]_\times$ 把三维向量变成反对称矩阵，指数映射由罗德里格斯公式给出：

$$
\exp\left( [u]_\times \right) = I + \frac{\sin\lVert u \rVert}{\lVert u \rVert} [u]_\times + \frac{1 - \cos\lVert u \rVert}{\lVert u \rVert^2} [u]_\times^2
$$

**速度**：增量在骨盆坐标系里，乘 $R$ 转回世界坐标。

$$
v' = v + R\, \Delta v, \qquad \omega' = \omega + R\, \Delta\omega
$$

**关节**：直接相加。

$$
q' = q + \Delta q, \qquad \dot{q}' = \dot{q} + \Delta \dot{q}
$$

网络最后一层初始化为0，所以训练刚开始时 $\Delta = \mu_\Delta$ ：模型先按“平均变化量”往前推，不会一上来就乱跳。

代码：`WorldModel.step`、`WorldModel.targets`。

### 5.5 训练方式：连续预测8步

**直觉**：只学“一步”的模型，在真实状态上很准，但推演时要拿自己的预测接着往下猜，误差会越滚越大。
所以训练时就让它连续自己推8步，逼它学会在“自己略有偏差的预测”上也能继续猜准。

从缓冲区取一段真实的连续8步：状态 $s_0, s_1, \ldots, s_8$ 和动作 $a_0, \ldots, a_7$ 。递推：

$$
\hat{s}_0 = s_0, \qquad \hat{s}_{j+1} = f_\omega\left( \hat{s}_j,\ a_j,\ \xi \right), \quad j = 0, 1, \ldots, 7
$$

对比“老师带着走”（teacher forcing）的单步训练 $\hat{s}_{j+1} = f_\omega(s_j, a_j, \xi)$ ：区别在于输入用的是自己的预测 $\hat{s}_j$ ，而不是真实的 $s_j$ 。

### 5.6 损失函数

**第一步，误差向量**：在“上一时刻真实状态”的局部坐标系里比较预测与真实，得到70维误差 $e_j$ （定义方式与5.4的增量一致）。

**第二步，按尺度归一化**：每一维除以它的典型变化量，再设一个下限，防止几乎不动的量被除以极小的数而主导损失：

$$
\tilde{e}_{j,d} = \frac{e_{j,d}}{\max\left( \sigma_{\Delta,d},\ \epsilon_d \right)}
$$

| 分量 | 下限 $\epsilon_d$ |
| --- | --- |
| 位置 | 3 mm |
| 朝向 | 0.005 rad |
| 线速度 | 0.02 m/s |
| 角速度 | 0.05 rad/s |
| 关节角 | 3 mrad |
| 关节速度 | 0.05 rad/s |

**第三步，掩码**：如果这8步中间有回合结束（摔倒或超时后重置），之后的步不再是同一条连续轨迹，不计入：

$$
m_j = \prod_{i=0}^{j-1} \left( 1 - \mathrm{done}_i \right)
$$

**第四步，汇总**：每步取70维的平均，8步相加，再对一批 $B$ 个窗口取平均：

$$
\mathcal{L} = \frac{1}{B} \sum_{i=1}^{B} \sum_{j=1}^{8} m_{ij} \cdot \frac{1}{70} \sum_{d=1}^{70} \tilde{e}_{ij,d}^{\,2}
$$

**与论文式(1)的关系**：

$$
\mathcal{L}_{\mathrm{WM}} = \sum_{j=1}^{k} \Big[ \underbrace{\lVert \hat{s}_j - s_j \rVert^2}_{\text{状态}} + \underbrace{(\hat{r}_j - r_j)^2}_{\text{奖励}} + \underbrace{\lVert \hat{\bar{o}}_j - \bar{o}_j \rVert^2}_{\text{观测}} + \underbrace{\mathrm{BCE}(\hat{d}_j, d_j)}_{\text{终止}} \Big]
$$

论文允许每个预测头“要么学、要么直接算”，直接算的头不计损失。本实现的奖励、观测、终止都用第2–4节的公式直接算，
所以只剩第一项“状态”。

代码：`train.py`中的`_wm_rollout`、`_wm_err`、`ERR_FLOOR`；掩码在`ReplayBuffer.sample_windows`。

## 6. 策略：在世界模型里想象，再求导

### 6.1 想象一步的流水线

**直觉**：从缓冲区挑一个真实发生过的状态作为起点，然后完全在世界模型里“做梦”32步，每一步都是下面五个式子。

**(1) 策略看着带噪的观测历史和目标，做出动作：**

$$
a_t = \pi_\theta\left( \tilde{o}_{t-9:t},\ g_t \right)
$$

**(2) 世界模型预测下一状态：**

$$
\hat{s}_{t+1} = f_\omega\left( \hat{s}_t,\ a_t,\ \xi \right)
$$

**(3) 从预测状态算出新观测，再加上新的传感器噪声：**

$$
\tilde{o}_{t+1} = \mathrm{obs}\left( \hat{s}_{t+1} \right) + \eta_{t+1}
$$

**(4) 用第2节的公式算奖励，用第3.2节的公式算软终止：**

$$
\hat{r}_{t+1} = r\left( \hat{s}_{t+1},\ a_t,\ a_{t-1} \right), \qquad \hat{d}_{t+1} = d\left( \hat{s}_{t+1} \right)
$$

**(5) 更新存活权重：**

$$
\hat{c}_{t+1} = \hat{c}_t \left( 1 - \hat{d}_{t+1} \right)
$$

### 6.2 目标函数：前32步的真分数 + 之后的“估分”

论文式(2)：

$$
\hat{J}(\theta) = \underbrace{\sum_{t=0}^{31} \gamma^t\, \hat{c}_t\, \hat{r}_{t+1}}_{\text{想象出来的32步}} + \underbrace{\gamma^{32}\, \hat{c}_{32}\, V_\psi\left( \hat{s}_{32}, g_{32}, \xi \right)}_{\text{critic估计的“32步以后还能拿多少”}}
$$

策略的训练就是让 $\hat{J}$ 变大，损失取负号：

$$
\mathrm{loss} = -\frac{1}{B} \sum_{b=1}^{B} \hat{J}_b
$$

**算例：两部分各占多少？** 假设机器人一直完美跟踪，每步0.1分、永不摔倒：

$$
\text{前32步} = 0.1 \sum_{t=0}^{31} 0.99^t = 0.1 \times \frac{1 - 0.99^{32}}{1 - 0.99} = 0.1 \times 27.5 = 2.75
$$

$$
\text{自举部分} = 0.99^{32} \times \frac{0.1}{1 - 0.99} = 0.725 \times 10 = 7.25
$$

两者加起来正好是满分10。也就是说，**目标函数里约73%的分量来自critic的估计**。
这解释了论文§4.3的结论：没有critic时，H≤16就训练失败，因为策略只能看到不到0.64秒的后果，变得“短视”。

### 6.3 梯度怎么穿过世界模型：两步的链式法则

**直觉**：动作 $a_0$ 会改变下一个状态，下一个状态又会改变后面的观测、动作和奖励。BPTT把这些影响全部沿链条反向累加。

只看两步、忽略存活权重和折扣，目标是 $J = r_1(\hat{s}_1) + r_2(\hat{s}_2)$ ，其中：

$$
a_0 = \pi_\theta(o_0), \qquad \hat{s}_1 = f(\hat{s}_0, a_0), \qquad a_1 = \pi_\theta\big( o(\hat{s}_1) \big), \qquad \hat{s}_2 = f(\hat{s}_1, a_1)
$$

$a_0$ 对 $J$ 的影响分成两条：

$$
\frac{\partial J}{\partial a_0} = \underbrace{\frac{\partial r_1}{\partial \hat{s}_1} \frac{\partial f}{\partial a_0}}_{\text{直接影响下一步得分}} + \underbrace{\frac{\partial r_2}{\partial \hat{s}_2} \left( \frac{\partial f}{\partial \hat{s}_1} + \frac{\partial f}{\partial a_1} \frac{\partial \pi}{\partial o} \frac{\partial o}{\partial \hat{s}_1} \right) \frac{\partial f}{\partial a_0}}_{\text{通过改变下一状态，间接影响再下一步}}
$$

括号里的两项正好对应两条路径：

- $\partial f / \partial \hat{s}_1$ ：状态本身沿物理规律传下去；
- $\frac{\partial f}{\partial a_1} \frac{\partial \pi}{\partial o} \frac{\partial o}{\partial \hat{s}_1}$ ：状态变了，策略看到的观测变了，下一个动作也跟着变。

32步时，这样的括号要连乘很多次。世界模型的参数 $\omega$ 在这一步是冻结的：它不被更新，但梯度要“穿过”它对输入的导数。

### 6.4 为什么想象不能太长：连乘的放大与衰减

假设每一步括号里的量大致是一个数 $\rho$ 。连乘32次：

$$
1.1^{32} \approx 21, \qquad 1.0^{32} = 1, \qquad 0.9^{32} \approx 0.034
$$

$\rho$ 稍大于1，梯度就爆炸；稍小于1，远处的影响就几乎消失。步数越多越不稳定，模型的误差也会一步步累积。
所以论文用短的 $H = 32$ ，再用critic补上后面的部分。

### 6.5 两道梯度裁剪

**第一道，按整体长度缩放**：把所有参数的梯度看成一个大向量 $G$ ，长度超过1就整体等比例缩小到1：

$$
G \leftarrow G \cdot \min\left( 1,\ \frac{1}{\lVert G \rVert} \right)
$$

**第二道，逐个元素截断**：每个分量限制在 $[-0.1,\ 0.1]$ 之内：

$$
G_i \leftarrow \min\left( \max(G_i,\ -0.1),\ 0.1 \right)
$$

第一道保住方向、只缩长度；第二道防止个别参数的梯度特别大。BPTT的梯度偶尔会有尖峰，两道一起用更稳。

### 6.6 Adam的 $\beta$ ：记忆多久以前的梯度

Adam对梯度的平方做指数滑动平均，系数为 $\beta_2$ 。它大约“记住”最近 $1/(1 - \beta_2)$ 步：

$$
\beta_2 = 0.999 \Rightarrow \text{约1000步}, \qquad \beta_2 = 0.95 \Rightarrow \text{约20步}
$$

策略用0.95：世界模型每次迭代都在变，梯度的统计特性也随之变化，短记忆能更快适应。

### 6.7 和PPO比：两种估计梯度的方法

PPO不知道环境怎么运作，只能看“这个动作的结果比平均好还是差”，来调整这个动作的概率（似然比估计）：

$$
\nabla_\theta J \approx \mathbb{E}\left[ \nabla_\theta \log \pi_\theta(a \mid o) \cdot A \right]
$$

$A$ 是优势：这个动作比平均好多少。它只告诉策略“往这个动作靠还是远离”，没有告诉“动作具体该怎么改”，
噪声很大，需要大量样本平均。

SuperMimic有可微的世界模型，可以直接算“动作每个维度变一点，回报变多少”（路径导数）：

$$
\nabla_\theta J = \sum_t \frac{\partial J}{\partial a_t} \frac{\partial a_t}{\partial \theta}
$$

它给出了具体的修改方向，短时域内噪声小得多。论文图5(b)实测，同一个世界模型里，后者达到相同回报所需的想象步数少约64倍。

## 7. Critic与GAE

### 7.1 Critic学什么

**直觉**：critic是个“估分员”：看一眼当前状态、目标和仿真上下文，估计“从这里开始，往后还能拿多少分”。

$$
V_\psi(s, g, \xi) \approx \mathbb{E}\left[ \sum_{i \ge 0} \gamma^i c_i\, r_{t+1+i} \ \middle\vert\ s_t = s \right]
$$

它能看到策略看不到的信息（比如 $\xi$ 、与参考的精确误差），这叫“非对称”设计：训练时可以作弊，部署时不需要它。

### 7.2 最简单的训练目标：一步时序差分

如果相信critic对下一步的估计，那么“这一步的真实奖励 + 下一步的估分”就是这一步的一个好目标：

$$
y_t^{(1)} = \hat{r}_{t+1} + \gamma \left( 1 - \hat{d}_{t+1} \right) V_{t+1}
$$

$(1 - \hat{d}_{t+1})$ 的意思是：如果下一步摔倒了，后面就没分了。

### 7.3 λ回报：在“多信自己”和“多信实测”之间折中

只用一步，偏差大（太依赖critic自己的估计）；一直用到第32步的真实奖励，方差大。GAE用 $\lambda$ 把两者混合，从后往前递推：

$$
\hat{R}_{32} = V_{32}
$$

$$
\hat{R}_t = \hat{r}_{t+1} + \gamma \left( 1 - \hat{d}_{t+1} \right) \left[ (1 - \lambda)\, V_{t+1} + \lambda\, \hat{R}_{t+1} \right]
$$

方括号是“下一步之后的价值”，由两部分加权：

- $(1 - \lambda) V_{t+1}$ ：critic对下一步的直接估分；
- $\lambda \hat{R}_{t+1}$ ：下一步自己用同样方法递推出来的目标。

两个极端：

$$
\lambda = 0:\ \ \hat{R}_t = \hat{r}_{t+1} + \gamma (1 - \hat{d}_{t+1}) V_{t+1} \quad \text{（一步时序差分）}
$$

$$
\lambda = 1:\ \ \hat{R}_t = \sum_{i=t}^{31} \gamma^{i-t} \left( \prod_{m=t+1}^{i} (1 - \hat{d}_m) \right) \hat{r}_{i+1} + \gamma^{32-t} \left( \prod_{m=t+1}^{32} (1 - \hat{d}_m) \right) V_{32} \quad \text{（一直用想象到底的奖励）}
$$

（空连乘 $\prod_{m=t+1}^{t}$ 按1计。）

$\lambda = 0.95$ 、 $\gamma = 0.99$ 时，每往后一步，权重乘以 $\gamma\lambda = 0.9405$ ，有效“看多远”约为：

$$
\frac{1}{1 - \gamma\lambda} = \frac{1}{0.0595} \approx 17\ \text{步}
$$

### 7.4 损失

目标 $\hat{R}_t$ 算好后固定不动，critic用均方误差去拟合：

$$
\mathcal{L}_V(\psi) = \frac{1}{32 B} \sum_{t=0}^{31} \sum_{b=1}^{B} \left( V_\psi(X_{t,b}) - \hat{R}_{t,b} \right)^2
$$

实现把这 $32 \times 4096 = 131{,}072$ 个样本打乱，分4个小批，重复5轮，共20次参数更新。

代码：`train.py`中`policy_update`的critic部分。

## 8. 按回报重置采样

### 8.1 把动作库切成1秒的小格

一段长 $L$ 帧的动作切成

$$
N_c = \left\lceil \frac{L - 1}{50} \right\rceil
$$

个分箱。用 $L - 1$ 是因为最后一帧只能当目标，不能当起点。每次重置时先挑一个分箱，再在格子里随机挑一帧作为起点。

### 8.2 每个分箱的“难度分数”

每当一个回合结束，用它拿到的折扣回报更新起点所在分箱的估计。回合在第 $T$ 步结束：

$$
G = \sum_{t=0}^{T-1} \gamma^t\, r_{t+1} + \mathbb{1}_{\mathrm{timeout}} \cdot \frac{\gamma^T\, \bar{r}}{1 - \gamma}
$$

**第二项是什么？** 如果是因为超时（到段尾或满10秒）结束，而不是摔倒，那么机器人其实还能继续拿分。
假设它以后每步都拿到本回合的平均分 $\bar{r}$ ，从第 $T$ 步起的折扣和就是等比数列：

$$
\sum_{i=T}^{\infty} \gamma^i\, \bar{r} = \frac{\gamma^T\, \bar{r}}{1 - \gamma}
$$

如果是摔倒结束，后面没分了，这一项为0。然后做滑动平均：

$$
G_b \leftarrow 0.8\, G_b + 0.2\, G
$$

### 8.3 越难越常练：softmax

把每个分箱的分数变成“被挑中的概率”，分数越低（越难）概率越高：

$$
q_b = \mathrm{softmax}_b\left( -25\, \frac{G_b}{G_{\max}} \right) = \frac{\exp\left( -25\, G_b / G_{\max} \right)}{\sum_{b'} \exp\left( -25\, G_{b'} / G_{\max} \right)}
$$

**算例**：两个分箱，一个已经做得很好（ $G = 10$ ），一个还差（ $G = 6$ ），

$$
\frac{q_{\text{难}}}{q_{\text{易}}} = \frac{e^{-25 \times 0.6}}{e^{-25 \times 1.0}} = e^{10} \approx 22{,}000
$$

系数25让分布非常“尖”：难的地方几乎会被独占式地练习。

### 8.4 三种分布混合

只按难度挑，会把简单的忘掉；所以混入均匀分布和“历史难度”：

$$
P(b) = 0.5 \cdot \frac{1}{N_{\mathrm{bin}}} + 0.4 \cdot q_b + 0.1 \cdot \frac{\bar{w}_b}{\sum_{b'} \bar{w}_{b'}}
$$

$$
\bar{w}_b \leftarrow 0.999\, \bar{w}_b + 0.001\, q_b
$$

| 成分 | 占比 | 作用 |
| --- | --- | --- |
| 均匀 | 50% | 保证每段动作都有练习，防止遗忘 |
| 当前难度 | 40% | 集中练当前做得差的地方 |
| 历史难度 | 10% | 以前难过的地方，学会后还保留一些练习 |

代码：`TrackingSim.sample_frames`、`TrackingSim.step`末尾的分箱更新。

## 9. 学习率调度

**直觉**：开始时步子大、学得快；后期步子小、精细调整。

前25万次迭代，倍数从1沿余弦曲线降到0.03：

$$
\mathrm{lr}(n) = \mathrm{lr}_0 \times \left( 0.03 + 0.97 \times \frac{1 + \cos\left( \pi n / 250{,}000 \right)}{2} \right), \qquad n < 250{,}000
$$

之后保持不变：

$$
\mathrm{lr}(n) = 0.03\, \mathrm{lr}_0, \qquad n \ge 250{,}000
$$

$n$ 为迭代次数， $\mathrm{lr}_0 = 2 \times 10^{-4}$ 。

| 迭代次数 $n$ | 学习率倍数 | 说明 |
| --- | --- | --- |
| 0 | 1.000 | 开始 |
| 96,000 | 0.688 | 单卡约3 h处（论文C1的检查点） |
| 125,000 | 0.515 | 衰减过半 |
| 250,000 | 0.030 | 衰减结束 |
| 320,000 | 0.030 | 训练结束 |

代码：`train.py`中的`lr_at`。

## 10. 评估指标

### 10.1 单段动作的归一化回报

**直觉**：把这段动作实际拿到的分，除以它理论上能拿到的满分，得到0到1之间的“完成度”。

长 $L_c$ 帧的动作，从第一帧播到最后一帧共 $L_c - 1$ 步，每步满分0.1：

$$
\mathrm{NR}_c = \frac{\sum_t r_t}{0.1 \times (L_c - 1)}
$$

中途摔倒后不再计分，所以摔得越早，分数越低。

### 10.2 两种汇总

**各段平均**：每段动作同等重要。

$$
\mathrm{NR}_{\mathrm{mean}} = \frac{1}{C} \sum_{c=1}^{C} \mathrm{NR}_c
$$

**按帧加权**：长动作分量更重。

$$
\mathrm{NR}_{\mathrm{frame}} = \frac{\sum_c \sum_t r_t}{0.1 \sum_c (L_c - 1)}
$$

**算例**：两段动作，短的100步、完成度1.0；长的1000步、完成度0.5。

$$
\mathrm{NR}_{\mathrm{mean}} = \frac{1.0 + 0.5}{2} = 0.75, \qquad \mathrm{NR}_{\mathrm{frame}} = \frac{100 \times 1.0 + 1000 \times 0.5}{1100} \approx 0.545
$$

差别可以很大。论文没说用的是哪一种，所以两种都报告。

代码：`supermimic/scripts/evaluate.py`中的`evaluate`。

## 11. 每次迭代的数据量

**真实数据**：更新数据比0.4表示“每个仿真步对应0.4次更新”，反过来就是每次更新采 $1/0.4 = 2.5$ 步：

$$
4096\ \text{环境} \times 2.5\ \text{步} = 10{,}240\ \text{条真实转移}
$$

**世界模型看到的预测次数**：

$$
10\ \text{批} \times 2048\ \text{窗口} \times 8\ \text{步} = 163{,}840
$$

**想象数据**：

$$
4096\ \text{条推演} \times 32\ \text{步} = 131{,}072\ \text{条想象转移}
$$

所以每采1条真实转移，策略就在想象里看到约 $131{,}072 / 10{,}240 \approx 13$ 条。

**每条真实数据被用了多少次？** 缓冲区每个环境存1000步，每次迭代进2.5步，一条数据大约停留

$$
\frac{1000}{2.5} = 400\ \text{次迭代}
$$

这期间世界模型一共抽样 $163{,}840 \times 400 \approx 6{,}550$ 万次，而缓冲区里共有 $4096 \times 1000 = 409.6$ 万条数据，
平均每条被用

$$
\frac{6{,}553.6\ \text{万}}{409.6\ \text{万}} = 16\ \text{次}
$$

这就是基于模型方法省数据的地方：同一条真实经验被反复拿来学习物理规律。
