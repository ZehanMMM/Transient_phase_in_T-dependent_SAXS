# 旧模型三阶段分项图

本次以用户指定的 contact_free_energy_kBT.png 对应模型为基准。
原图保留，新增 separated_stage1/2/3 的 kBT、J、PNG、PDF 版本。

1. 主曲线为 Udd_time_mean，虚线为 Udd_eq。
2. 主曲线为 delta_Fmag_time_mean，虚线为 delta_Fmag_eq。
3. 主曲线为 Fmag_plus_rotation_time_mean，虚线为相应 equilibrium 值。

所有主曲线都不含 vdW。face vdW 与 kBT 各自作为独立参考线。
第三图严格等于原 contact 图各主曲线减去各自的 vdW，不改变磁动力学。
前两图使用相同纵轴范围。

恢复的旧假设：固定几何，64 个易轴磁态，每个温度从均匀初态独立演化，
在 0 到 20 s 平均。内部各向异性最低值置零，势垒进入跃迁速率。
自由能为 U + kBT sum(p ln(64p)) 的时间平均。未再添加弛豫差值 penalty。
旧程序以 F_eq + kBT KL 恒等式评价该自由能，并非在实际自由能上叠加差值。

转动熵恢复旧图定义，tilt 容差 2 度，face twist 为 +/-5 度，
tip twist 绕链接轴完全自由。tip 并非三维完全自由。
单颗粒 SO(3) 允许域比例为

    c = (1-cos(2 deg))/2
    f_face = 24 c (5 deg in radians)/pi
    f_tip = 8 c
    F_rotation = -2 kBT ln(f)

它只是因子化几何熵修正，不改变固定几何的磁能和跃迁矩阵。
所以低温 tip 仍可因随机初态与慢 Neel 重排而接近零。
这不代表已计算了颗粒整体旋转对齐。不能同时声称恢复旧模型又解决了该局限。
此前 axial_face_free_tip 分支属于不同模型，不能与本次当作仅改画法的版本比较。

生成命令：

    python versions/V10_multiphysics_nanocube/code/finite_window_rotational_entropy.py --separated-only
