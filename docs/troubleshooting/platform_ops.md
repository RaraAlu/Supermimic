# 平台、远程服务器与运维

- 范围：Windows/Linux共用仓库、远程A100服务器的连接与文件传输、代码同步、训练与诊断的运行管理。
- 机器：Rara（Windows，RTX 4070 Ti SUPER 16 GB，仓库`D:\Train\Supermimic`）；bitahub2（2×A100-SXM4-80GB，Linux）。
- 本页记录这些机器当时的环境与操作约定；主机别名、绝对路径、远程名和同步方式不代表其他克隆的当前配置，操作前以当地`git remote -v`等实际状态为准。

## 1. Windows与Linux共用仓库

| 问题 | 原因 | 解决 |
| --- | --- | --- |
| Linux侧`uv run`删掉了Windows的`.venv` | 两边共用一个目录，`.venv`按平台不同；挂载盘不支持符号链接，Linux侧也重建不了 | 所有执行（`uv`、测试、训练）都经`ssh Rara`在Windows上跑；Linux侧只读写文件。Linux若要独立环境，用`UV_PROJECT_ENVIRONMENT`指到共享目录之外 |
| 额外worktree没有`.venv` | `uv run`会在worktree里新建环境 | 在worktree目录下直接用`D:\Train\Supermimic\.venv\Scripts\python.exe` |
| Windows默认编码读写日志、`args.json`出错 | Windows默认非UTF-8 | 显式以UTF-8打开（`733ebe2`） |
| 默认数据目录断言在Windows上假失败 | 路径分隔符为反斜杠 | 用`Path.as_posix()`比较（`ecfee4d`） |
| `MotionLib.from_args`调用报`TypeError` | 引入`from_source`时`@classmethod`装饰器错位 | 补回装饰器并加回归测试（`e1f0771`） |
| venv里没有pytest | 项目依赖不含pytest | 用`python -m unittest discover -s tests` |

### cmd.exe的坑

- `if exist ... & git format-patch ...`会吞掉后面的命令：分开执行。
- `git log --format=%h...`里的`%`被cmd当作变量：改用`git show -s <提交>`看完整提交信息。
- `findstr`没匹配到时返回1，接在`&&`链里会中断后续命令；错误信息按系统代码页（GBK）输出，经ssh显示为乱码。

## 2. 连接bitahub2

服务器只接受密码登录；本机`~/.ssh/config`给别名`bitahub2`配了`ControlMaster auto`，登录一次后其余命令复用主连接。

| 问题 | 原因 | 解决 |
| --- | --- | --- |
| 对话框里`!ssh bitahub2`直接断开 | `!`执行的命令没有TTY，无法确认主机密钥，也无法输入密码 | 在终端标签页里交互登录 |
| 新端口的主机密钥未知 | 服务器换了端口 | 核对新端口的密钥与之前已信任的相同（同一指纹）后加入`known_hosts` |
| `Connection closed`（认证之前） | sshd报`Exceeded MaxStartups`：同时握手的未认证连接太多，部分来自连续测试 | 稍等重试；不要密集发起新连接 |
| 用`ssh -p 端口 root@...`登录后其他命令仍要密码 | 直接写地址不走别名，主连接不在约定的`ControlPath` | 一律用`ssh bitahub2`登录；去掉了无用的`ProxyJump` |
| 主连接几小时后失效 | 预期12 h，实测约4 h后被中途断开（可能是网关） | 监督脚本ssh连续失败10次（约10分钟）后退出并提示重新登录 |
| 从Rara连不上 | Rara的公钥未加到服务器；Rara ssh配置里带密钥的那条指向旧端口 | 不加公钥，仍由本机终端登录后复用主连接 |

密码只由用户在终端里输入，代理不代输。

## 3. 文件传输

| 问题 | 解决 |
| --- | --- |
| 大文件`scp`中途断开，主连接随之失效（71.78 MB的权重传到40 MB停止） | 优先把诊断放到服务器上跑，只传小文件；必须传时用`scp -l 8000`限速或分块，传后核对SHA256 |
| 小检查点`ckpt_*.pt`里没有critic | 只含`pi`、`wm`与元数据；读critic或优化器状态用`last.pt` |
| 分箱状态不在检查点里 | 已改：`last.pt`保存各rank的`scores`、`visits`、`ema`，世界大小与划分相同时恢复（之前的检查点仍没有，恢复时课程从零开始）；训练中终止集中在哪些片段由`termwin_log.csv`与`term_bins.csv`记录 |
| Windows上`torchrun`报`use_libuv was requested but PyTorch was built without libuv support` | Windows版PyTorch未编译libuv（异步IO库，与uv包管理器无关），`torchrun`的rendezvous要用它；本机多进程测试改为直接起N个进程，设置`RANK`、`LOCAL_RANK`、`WORLD_SIZE`、`MASTER_ADDR`、`MASTER_PORT`与`USE_LIBUV=0` |

## 4. 代码同步

- 服务器worktree：`/root/work/train/Supermimic-tailbin`，分支与Rara相同。
- 做法：Rara上`git format-patch -1 HEAD`，`scp`到服务器后
  `git -c user.name=... -c user.email=... am --committer-date-is-author-date PATCH`。
  作者、提交者与时间都相同，两边哈希一致；同步后用`git rev-parse HEAD`核对。
- 只在用户要求时push；GitHub远程在Rara上叫`github`。

## 5. 训练与诊断的运行管理

| 事项 | 做法 |
| --- | --- |
| 运行留档 | 每个run一个tmux会话，会话名与run目录同名；启动脚本复制为`RUN/run_train.sh`，`console.log`首行记录开始时间与提交 |
| 停止训练 | 先在run目录写`user-stop.json`（原因、时间、最后迭代），再向tmux会话发`C-c`；`exit.json`记录退出码 |
| 不碰别人的训练 | 不是自己启动的训练，未经同意不停、不改、不在其GPU上跑诊断 |
| 在训练卡上跑诊断 | 训练会从2.6降到约1.7 it/s；全量数据的诊断可能挤爆显存，改用子集（`scripts/make_motion_subset.py`） |
| 等待长任务 | 后台`until`循环等完成标志或文件，不轮询、不串联`sleep` |
| 远端命令可追溯 | `smrun`：经命名tmux会话执行，写`tmux/<session>.log`与`tmux/registry.tsv`（`809c66c`） |
