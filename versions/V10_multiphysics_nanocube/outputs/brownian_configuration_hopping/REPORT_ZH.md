# 颗粒整体转动引起的接触构型跃迁：20 s 能量图

本版依据用户澄清，将 Brownian 运动处理成不同等价接触构型之间的跃迁，
不是之前被中止的 coupled_axial_brownian_neel 固定链接轴扭转模型。
不计算或叠加 dipole 取向熵、整体转动熵、弛豫差值或任何能量 penalty。
主曲线仅为 Udd，face vdW 与 kBT 独立绘制。

## 构型和状态

保留旧模型的 16 nm 核心、3 nm 表面间距、Ms=2.85e5 A/m、CV=0、
tau0=0.98 ns、TB=250 K（100 s 标定）及 hexane 黏度模型。
face 中心距 19 nm，tip 中心距 28.51666 nm，仍采用旧的圆角几何定义。
每个温度从均匀 64 态独立开始，计算 0..20 s 的平均。

一个立方体存在 24 个立方对称的整体姿态，可由绕 <100> 轴正负90度转动连接。
端点形状相同，但朝向邻居的是不同的材料面或材料顶角。
若体坐标中的磁矩锁定在某条111易轴，整体姿态变化会改变实验室中的磁矩。
24 姿态对应 8 个实验室易轴方向，每个方向有 3 个等能对称姿态。
在立方对称、形状和接触势不区分材料面、所选路径对称的近似下，可合并这3重简并。
两颗粒的能量与跃迁过程因此可投影到原有 8x8=64 个方向状态。

这里增加的是同一 face 分支或同一 tip 分支内部的构型变换，
不是 face 与 tip 两个分支之间的结构转变。
也不再采用 face twist +/-5度的硬限制，否则不能跳到另一材料面的接触构型。

## 速率

概率方程及绘图量为

    dp/dt = p (Q_N + Q_B)
    Udd_mean = integral[0,tw] sum_i p_i(t) Udd_i dt / tw

Q_N 沿用内部易轴翻转速率，仍含内部各向异性势垒。
Q_B 来自整体刚体转动，磁矩与晶格共同旋转，因此不加内部各向异性翻转势垒。
所有占据态的各向异性最低能均置零，未包含阱内连续偏离易轴。

    tau_B = 3 eta(T) Vh / (kBT) = 1/(2 Dr)
    Vh = pi (19 nm)^3 / 6

使用球等效自由颗粒转动阻力。tau_B 是一阶取向相关衰减时间，不是某次构型逃逸时间。
选择每个颗粒6条有向正负90度路径。在无磁耦合时，这些路径满足

    sum_moves (R_move - I) m = -4 m

故每条路径的零势垒速率为 k0=1/(4 tau_B)，使 <m(t)> 按 exp(-t/tau_B) 衰减。
该校准只匹配一阶相关时间，不意味着完整连续 Brownian 扩散被精确离散化。

沿路径，另一个颗粒暂时固定，内部磁矩保持锁定于体坐标易轴。

    Edd(theta) = C [m1(theta).m2 - 3(m1(theta).rhat)(m2.rhat)]
    E_path_max = max_[0 <= theta <= pi/2] Edd(theta)
    k_B,path = [1/(4 tau_B)] exp[-(E_path_max - E_i + B_contact)/(kBT)]

Edd(theta) 是 A cos(theta)+B sin(theta)+D，代码解析求端点和区间内极值。
反向路径共享相同最大能量，满足详细平衡。两条机制的速率相加，而不是能量相加。
本次 B_contact=0 是未额外添加接触势垒的基准，不是接触势垒的实验测量值。

## 关键限制

端点是合法对称构型，不保证中间90度刚体转动在固定中心距下没有核碰撞或配体排斥。
尤其 face 更换接触面可能需要先分离，再转动，再接触。
本版没有计算这些平移、配体重排、接触摩擦及多粒子约束，因而是快速可达性的基准。
tau_B 不能单独确定这些真实接触跃迁速率。
黏度低于250 K仍为继承的外推。M_s(T)未实测，保持常数。
vdW沿用未收敛的4^3体素名义姿态值，仅作独立参考，不参与转动路径势垒。

## 结果

20 s 的新平均偶极能（每对颗粒，单位 kBT）为

| T / K | face | tip |
|---|---:|---:|
| 200 | -9.51422 | -2.81257 |
| 250 | -7.50928 | -1.83648 |
| 300 | -6.13395 | -1.28776 |

tip 在200 K不再接近零。Brownian 自由时间在200、250、300 K分别约为
4.083、1.601、0.759微秒。偶极路径势垒进一步改变具体跃迁时间。
200 K联合生成矩阵的最慢非稳态模态时间约 face=0.0599 s、tip=48.5微秒。
因此20 s磁能接近平衡，实线与虚线在图上重合。未直接赋值为平衡态，
而是用完整联合生成矩阵的谱传播和解析时间平均得到。
此结论不代表真实溶液的组装或玻璃结构在20 s内完成弛豫。
没有出现所期待的face/tip相交，不能通过调画法声称存在相变。

## 参考与运行

转动摩擦及tau_B定义参考：
Dynamics of interacting magnetic nanoparticles: effective behavior from competition
between Brownian and Neel relaxation, PCCP (2020), DOI:10.1039/D0CP04377J.
https://pubs.rsc.org/en/content/articlehtml/2020/cp/d0cp04377j

六路径构型网络及路径最大能量速率为本次粗粒化模型选择，并非上述文献的原公式。

    python versions/V10_multiphysics_nanocube/code/brownian_configuration_hopping.py
    python -m unittest discover -s versions/V10_multiphysics_nanocube/code -p test_brownian_configuration_hopping.py
