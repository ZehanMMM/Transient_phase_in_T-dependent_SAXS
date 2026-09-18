# 当前20 s加转动熵模型的完整表达式

## 曲线定义

g表示face或tip。图中实线为每对颗粒、相对于两个无耦合且自由取向颗粒的
部分接触自由能，不是纯磁内能，也不是完整组装Gibbs自由能。

\[
\overline{\Delta F_g}^{20s}(T)=
\frac{1}{t_w}\int_0^{t_w}\left[
\sum_{i=1}^{64}p_{g,i}(t,T)E_{g,i}
+k_BT\sum_{i=1}^{64}p_{g,i}(t,T)\ln(64p_{g,i}(t,T))
\right]dt-2k_BT\ln f_g+U_{\mathrm{vdW},g},\qquad t_w=20\ \mathrm{s}.
\]

按顺序分别为平均磁内能、相对于均匀64态的磁取向熵代价、两个颗粒的
局部整体取向熵代价、vdW。磁取向熵与整体取向熵是不同自由度，但将两者
相加需要当前模型的因子化近似，忽略整体转动对瞬时偶极能和速率的反馈。

图中的vdW曲线只显示face组分。face和tip的总曲线已经包含各自的vdW，
不可再把图中蓝线重复加到总曲线上。kBT是参照线，不是被额外相加的项。

## 1. 磁态能级

两个颗粒各有8条有向<111>易轴，构成64个状态。

\[
E_{g,i}=\frac{\mu_0\mu^2}{4\pi r_g^3}
\left[\mathbf s_{1,i}\cdot\mathbf s_{2,i}
-3(\mathbf s_{1,i}\cdot\hat{\mathbf r}_g)
(\mathbf s_{2,i}\cdot\hat{\mathbf r}_g)\right],
\qquad \mu=M_sV.
\]

s为磁矩方向单位向量，r为中心距离，rhat为中心连线单位向量。
V=(16 nm)^3，Ms=2.85e5 A/m，因此mu=1.16736e-18 A m2。
固定该Ms及几何，所以本版能级本身不随T变化。
face中心距19 nm、中心连线沿[100]，tip中心距28.51666 nm、沿[111]。
表面间隙均为3 nm，来源是1.5 nm配体厚度和旧圆角几何函数。

一般的立方各向异性能相对最低点为

\[
\Delta E_{ani}(\mathbf s)=K_cV
\left[\frac13-s_x^2s_y^2-s_y^2s_z^2-s_z^2s_x^2\right].
\]

在所有64个易轴组合中，两粒子的此项都为零。因此本版占据态磁能等于
偶极能。Kc只通过跃迁势垒起作用。未计算连续易轴内偏转能，不使用q(T)^2。
不要将此版本与rotational_entropy_model.py的连续磁矩平衡模型混淆。

## 2. 概率与Néel跃迁

每个温度从p_i(0)=1/64重新开始，不继承降温历史。

\[
\frac{dp_i}{dt}=\sum_{j\ne i}(p_jk_{j\to i}-p_ik_{i\to j}),
\qquad \mathbf p(t)=\mathbf p(0)e^{Q(T)t}.
\]

每条边只改变一个颗粒的一个方向符号，连接相邻<111>易轴。

\[
k_{i\to j}=\frac{1}{3\tau_0}
\exp\left[-\frac{\max(\Delta E_{ani}^{\ddagger}
+E_{dd,ij}^{\ddagger}-E_{dd,i},0)}{k_BT}\right].
\]

tau0=0.98 ns，1/3是将每个颗粒总逃逸尝试频率分配到其3条相邻路径，
不是整对颗粒仅有3条路径。
鞍点由正在跳跃的磁矩置一个分量为0再归一化，得到<110>方向，另一个
磁矩保持原方向，代入相同偶极公式得到Edd(saddle)。

\[
\Delta E_{ani}^{\ddagger}=k_BT_B\ln(t_{ZFC}/\tau_0)
=K_cV/12.
\]

TB=250 K、tZFC=100 s，得到势垒8.7494e-20 J与Kc=2.5633e5 J/m3。
这仍假设有效阻塞势垒完全对应纯立方各向异性，尚未由样品独立验证。
基础势垒与tau0取常数，但跃迁速率随T及磁矩组合改变。

## 3. 磁取向熵

\[
S_{mag}[p]=-k_B\sum_i p_i\ln p_i,\qquad
S_{mag,free}=k_B\ln64,
\]
\[
-T\Delta S_{mag}=k_BT\sum_i p_i\ln(64p_i).
\]

0 ln0取0。均匀分布时此项为零，概率分布相对于均匀态集中时此项非负。
它不是额外的势能或能垒。窗口平均是在每一时刻计算p ln p后积分，
不是用平均概率代入一次自由能。

## 4. 整体转动熵

链接轴允许在半角theta=2度的圆锥内摆动。face绕该轴的twist在[-delta,delta]
内，主图delta=5度。tip绕链接轴twist自由。
SO(3)测度相对于完整自由取向归一化。

\[
c(\theta)=\frac{1-\cos\theta}{2},\quad
f_f=24c(\theta)\frac{\delta}{\pi},\quad f_t=8c(\theta).
\]

24是理想立方体等价face配准姿态的数目，8是其有向体对角线数目。
计算delta/pi时使用弧度。两粒子独立局部约束的旋转自由能为

\[
\Delta F_{rot,g}=-2k_BT\ln f_g.
\]

主图Frot,face=17.00404 kBT，Frot,tip=12.03423 kBT。
相对于自由取向颗粒，两者均有正的约束代价，tip的代价更小。
两者之差为-4.9698133 kBT，不重复按配位数增加。
摆动角未测定，仅为试算参数。假设局部角度自由度已在允许域内平衡，
不是Brownian真实轨迹。若冻结只是有限时间下的转动缓慢，此熵近似需重新审查。

## 5. vdW

\[
U_{vdW,g}=-\frac{A_H}{\pi^2}\sum_{a,b}
\frac{\Delta V_a\Delta V_b}{\left[\max(d_{ab}^2,d_{floor}^2)\right]^3},
\quad d_{ab}=|\mathbf r_g+\mathbf x_b-\mathbf x_a|.
\]

a,b标记两个颗粒的体素，AH=2e-20 J，每颗粒4^3个体素，
DeltaV=(4 nm)^3，代码dfloor^2=1e-19 m2作为数值保护。
结果为face -4.88344e-21 J，tip -2.45320e-22 J。
以J表示恒定，以kBT表示随1/T变化。
此分辨率已知未收敛，本版为保持20 s基线只加熵而沿用，尚未包含
整体转动导致的vdW变化、真实圆角体积分或配体/溶剂有效作用。

## 6. 平衡态虚线

\[
p_i^{eq}=\frac{e^{-E_i/k_BT}}{\sum_j e^{-E_j/k_BT}},
\]
\[
\Delta F_g^{eq}=-k_BT\ln\left[\frac1{64}\sum_i e^{-E_i/k_BT}\right]
-2k_BT\ln f_g+U_{vdW,g}.
\]

平衡虚线与20 s实线使用完全相同的能级、转动约束和vdW，只替换磁态分布。
由此实线与同构型虚线之差为

\[
\overline{\Delta F_g}^{20s}-\Delta F_g^{eq}
=\frac{k_BT}{t_w}\int_0^{t_w}D_{KL}(p_g(t)\Vert p_g^{eq})dt\geq0.
\]

此量已包含在实线，不再作为penalty额外相加。代码测试检查了这一恒等式。

## 7. kBT与实验背景

kB=1.380649e-23 J/K。J图中的黑线为kBT，归一化图中为1。
该线仅提供热能尺度，不能当作组装熵或额外加到F上。

灰色实验范围233.15至293.15 K，即-40至20摄氏度。
黄色transient aggregation范围253.15至273.15 K，即-20至0摄氏度。
两者都来自用户给出的实验区间，不决定模型参数或数值交点。

本式是当前代码所有已计算项的完整表达式，但不是完整的组装自由能。
未包括整体转动与磁矩动力学反馈、配体/溶剂自由能、平移熵及SC多粒子约束。
