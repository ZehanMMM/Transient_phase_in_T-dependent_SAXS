# 从300 K继承磁矩分布的降温历史模型

本次关闭颗粒整体转动，只保留旧的64态Neel生成矩阵。
不加入取向熵、转动熵、Brownian修正、未弛豫penalty或其他能量项。
face/tip主曲线为两粒子的平均Udd，face vdW和kBT各自独立绘制。

## 实验程序与初态

300 K时仅初始化一次 p_i=1/64，首先在300 K恒温采集。
随后依次降到290、280、...、200 K。每降10 K用120 s线性降温，
降温过程中也推进磁矩概率，之后在目标温度恒温采集。
输出每点20 s（与上一版对照）及150 s（用户此前SAXS积分时长）两种协议。
两种协议各自独立运行，不能把其数据混合成同一条冷却轨迹。
150 s采集是整个0..150 s平均，不以75 s时刻值代替。

    dp(t)/dt = p(t) Q_N[T(t)]
    p_next,start = p_previous,end 经整个降温段传播后的分布
    Udd_frame = integral_hold p(t).Edd dt / exposure_duration

降温采用0.25 K子步、子步中点温度的矩阵指数传播，恒温段采用谱传播及解析时间平均。
概率在任意相邻时间段连续，不重置，不从平衡态初始化后续温度。
绘图中11个实际温度点的采集平均值用PCHIP连接，不显示点标记。
插值仅用于显示，不是额外温度实验或动力学计算。
完整降温和恒温时间轨迹另存time_trajectory.csv。

## 保留参数与限制

16 nm Fe3O4，Ms=2.85e5 A/m保持常数，表面间距3 nm。
固定假想face pair中心距19 nm，tip pair中心距28.51666 nm。
磁偶极耦合在300 K初始时就存在，不计算从远距离胶体到接触的过程。
CV=0，tau0=0.98 ns，TB=250 K由100 s条件标定内部各向异性势垒。
磁矩占据8个易轴最低点，最低各向异性能置零，势垒只进入Neel跃迁速率。
没有连续阱内收窄、颗粒转动或face与tip分支之间的结构转换。
两个分支是分别推进的假想固定构型，不是一个体系自发切换构型的轨迹。
vdW沿用旧4^3体素名义几何结果，未收敛、不随姿态变化，仅为参考。

## 数值结果：每对粒子，单位为相应温度的kBT

| T / K | face 20 s | tip 20 s | face 150 s | tip 150 s |
|---|---:|---:|---:|---:|
|300|-6.08499|-1.22001|-6.12743|-1.27873|
|250|-7.50892|-1.80909|-7.50924|-1.82147|
|200|-9.47540|-2.38113|-9.48481|-2.43002|

低温tip不再接近零。这里并没有加快低温Neel翻转，而是保留了较高温度建立的相关性。
低温分布变化减慢后，J单位磁能逐渐冻结在非零负值。
同一负能量除以更小的kBT，会在kBT单位图中继续向下，因此需同时查看J单位图。
历史模型不必达到每个温度的平衡：200 K tip平衡值为-2.81258 kBT，
而20 s历史值约-2.38113 kBT。face在此模型下仍更低，未得到face/tip交叉。
这些是固定构型条件磁能，不能据此独立认定组装相或玻璃相稳定性。

## 文件

- magnetic_energy_history_20s_kBT.png/pdf及J版本
- magnetic_energy_history_150s_kBT.png/pdf及J版本
- frame_energies.csv：各曝光起止、平均、平衡和同窗口随机重置对照能量
- state_probabilities.csv：每个温度前一曝光末态、本次曝光初态、平均与末态
- time_trajectory.csv：整个时间轨迹（恒温期间也记录瞬态，不只是平均值）

    python versions/V10_multiphysics_nanocube/code/neel_cooling_history.py
    python -m unittest discover -s versions/V10_multiphysics_nanocube/code -p test_neel_cooling_history.py
