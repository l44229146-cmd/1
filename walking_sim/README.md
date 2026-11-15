# 平地行走仿真初始平台

## 肌骨骼人体模型与髋部辅助接口

另已接入 [MyoSuite 3.0.0 的 myoLegWalk-v0](https://myosuite.readthedocs.io/en/latest/suite.html)。它使用以 [Rajagopal 等（2016）人体步态肌骨骼模型](https://doi.org/10.1109/TBME.2016.2586891)为重要参考的腿部和躯干模型，包含左右髋、膝、踝以及 80 个肌肉执行器。MyoSuite 模型结构及其参考来源有研究文献支撑，但这不意味着本项目产生的步态或新加入的辅助系统已经通过人体实验验证。具体模型、转换方法和控制器都需要独立核查与评估。

由于肌肉模型和原先简化人形的动作空间不同，原来的 Humanoid-v5 策略不能直接用于它。当前测试以零肌肉激励运行，人体很快失稳；这只证明模型和数据接口可执行，**不是正常行走演示**。

脚本在两侧髋屈伸关节施加给定的恒定广义力矩，用作外骨骼执行器的理想化占位接口。它**不是完整外骨骼**：没有外骨骼实体、质量、穿戴接触、传感器、延迟或实时智能响应。不同模式暂不能作为辅助效果结论；未找到可直接整合并经过验证的髋部设备模型，后续需选定并耦合具体设备，再配合适用于肌肉模型的行走控制器及真实步态数据进行验证。

独立环境重建（不修改原来的 `.venv`）：

```powershell
python -m venv .venv-myosuite
.\.venv-myosuite\Scripts\python.exe -m pip install -r walking_sim\requirements_myo.txt
```

运行接口检查：

```powershell
.\.venv-myosuite\Scripts\python.exe walking_sim\myo_hip.py --steps 30 --torque-nm 0 --output walking_sim\results_myo\no_assistance.csv
.\.venv-myosuite\Scripts\python.exe walking_sim\myo_hip.py --steps 30 --torque-nm 2 --output walking_sim\results_myo\hip_torque_2nm.csv
```

CSV 保存仿真时间、髋部外加力矩、两侧髋膝踝关节角度和角速度，以及环境终止状态。角度单位为 rad，角速度为 rad/s，力矩单位为 N·m。两侧力矩采用各自模型关节的正方向；相同数值不应未经检查就解释成相同的人体解剖运动方向。该测试力矩被限制在每侧 ±10 N·m 内，这只是软件测试边界，不代表设备安全上限。不要把仿真力矩直接用于真人或硬件。

这是一个简化的、无外骨骼的人形控制基线，不是经生物力学验证的正常人模型。Gymnasium Humanoid-v5 的目标是尽快前进，不限制正常步速，也没有独立踝关节或肌肉模型，因此不能用它直接声称真实人类步态、肌肉力矩或临床效果。

策略来自 Farama Minari 发布的 [Humanoid-v5 SAC expert](https://huggingface.co/farama-minari/Humanoid-v5-SAC-expert)，固定版本 `f9130b25c70584670ceac33eb1d06fbc418d691a`。下载的 checkpoint SHA-256 为 `5a7b38be61afb41cfe37acdc70e9997f29415a833684d91773c6049951a18cd6`。策略使用 SAC 而非 PPO；这里先复现已有策略，不代表在本机从零训练完成。原模型页面没有明确列明许可证，复用或分发前需确认许可。SB3 checkpoint 可能包含可执行的序列化数据，只加载可信来源。

## 较慢的候选策略

另比较了同一来源的 [SAC medium](https://huggingface.co/farama-minari/Humanoid-v5-SAC-medium)（版本 `3c2263f7923ccbdb659493e49a6240b429462802`，SHA-256 `3b92da2a5bd8b4e7024f7f971ac5ef2fdcfc49caf75e3edc3b826d216317c0c5`）及 [SAC simple](https://huggingface.co/farama-minari/Humanoid-v5-SAC-simple)（版本 `97d27b0552f4d9176b564532e80c32f8fdaf54b1`，SHA-256 `e7706432558e22d3a5930efd5f9b42640dc4d1962897f4182ac13d363b1735b5`）。它们同样是预训练模型，并未在本机重新训练。对种子 0–4 的初步筛选中，medium 较 expert 慢且 5 次均达到时限，simple 多数时候几乎不向前。需要在不同种子上重新评估，仍不能视作经验证的人体步态。

```powershell
.\.venv\Scripts\python.exe walking_sim\compare_policies.py
.\.venv\Scripts\python.exe walking_sim\run_walking.py --model walking_sim\models\humanoid-v5-sac-medium.zip --output walking_sim\results_medium --gif
```

比较脚本默认用未参与初选的种子 100–109，输出逐次实验及汇总。1.3 m/s 只是对照用的示例目标速度，并未强加给任何策略，也不是证明步态正常的标准。时间上限内未提前终止不代表安全或人体步态正常。

## 运行

在当前文件夹打开 PowerShell：

```powershell
.\.venv\Scripts\python.exe walking_sim\run_walking.py --gif
```

输出在 `walking_sim\results`：`joint_data.csv`、`summary.json`、`hip_curves.png` 和 `walking.gif`。使用 `--render` 可尝试打开交互式窗口（不要同时使用 `--gif`）；使用 `--output` 指定另一个输出目录。

可选：在已有模型上继续训练，并保存为新 checkpoint，训练可能很慢，也可能使性能下降：

```powershell
.\.venv\Scripts\python.exe walking_sim\train.py --steps 10000
.\.venv\Scripts\python.exe walking_sim\run_walking.py --model walking_sim\models\sac_finetuned.zip --output walking_sim\results_finetuned --gif
```

## 数据说明

每行是一个环境步骤结束后的状态，时间间隔为 0.015 秒。角度单位为 rad，角速度为 rad/s；`actuator_torque_nm` 是 MuJoCo 在该时刻针对给定状态和控制输入计算的**广义执行器力矩**，不是归一化的动作命令，也不是人体肌肉力矩或总关节净力矩。左右脚接触值仅表示脚与地面发生几何接触；并非经过验证的 heel-strike 或 toe-off 标签。

`terminated_unhealthy` 代表环境不健康导致提前终止，不等同于经过验证的摔倒判定；`truncated_time_limit` 是 Gymnasium 的 1000 步时限。平均速度为净前进距离除以仿真时长。要论证“正常人行走”，仍须与真实步态速度、关节轨迹和步态事件数据比较，并选择包含足踝的合适人体模型。

重建环境（Python 3.11）：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r walking_sim\requirements.txt
```

Windows 下模型 XML 临时复制至系统临时目录，以避免 MuJoCo 无法读取包含中文的路径。
