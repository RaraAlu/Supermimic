# GitHub公式排版约定

仓库文档采用[GitHub支持的数学表达式语法](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions)。
数值、变量及算法含义沿用各文档的定义；排版时优先让公式在正文与表格中都能读清。

## 行内公式

使用美元符号加反引号，保护下标、反斜线与Markdown标记：

```markdown
折扣因子为 $`\gamma=0.99`$，双卡学习率为 $`\sqrt{2}\,\alpha_0`$。
```

显示效果：折扣因子为 $`\gamma=0.99`$，双卡学习率为 $`\sqrt{2}\,\alpha_0`$。
文件名、命令、参数名保留普通代码格式，例如`--lr_scaling sqrt`。

| 量 | 表达式 | 含义 |
| --- | --- | --- |
| 学习率 | $`\alpha_2=\sqrt{2}\,\alpha_0`$ | 双卡缩放 |
| 窗口长度 | $`k=8`$ | 每个世界模型训练窗口的步数 |
| 想象时域 | $`H=32`$ | 每条想象轨迹的步数 |

## 独立公式与多行对齐

使用`math`围栏；围栏内不再套美元定界符。独立公式放在顶层段落，不缩进到列表里。长公式在等号或加号处换行，使用`aligned`对齐：

````markdown
```math
\begin{aligned}
\hat{J}
&= \sum_{t=0}^{H-1}\gamma^t\hat{c}_t\hat{r}_{t+1} \\
&\quad+\gamma^H\hat{c}_H V_\psi(\hat{s}_H,g_H,\xi).
\end{aligned}
```
````

显示效果：

```math
\begin{aligned}
\hat{J}
&= \sum_{t=0}^{H-1}\gamma^t\hat{c}_t\hat{r}_{t+1} \\
&\quad+\gamma^H\hat{c}_H V_\psi(\hat{s}_H,g_H,\xi).
\end{aligned}
```

跨行括号用固定尺寸的`\Bigl`、`\Bigr`，避免把一对`\left`、`\right`拆到不同行。
公式中的文字用`\text{...}`，运算符用`\operatorname{...}`或`\mathrm{...}`；
绝对值和范数用`\lvert...\rvert`、`\lVert...\rVert`，避免与Markdown表格的分隔符冲突。
论文的公式编号使用`\tag{1}`等显式标签，并与正文引用保持一致。

## 界面验收

在GitHub的**Preview**页等待公式完成渲染，再检查正文、表格与编号。
重点查看损失函数、链式法则、GAE和多卡批量公式；同时检查一般视窗及窄视窗。
窄视窗下长公式可以在公式区域内横向滚动，正文不应被挤宽；不要用整段缩小字号掩盖过长公式。
