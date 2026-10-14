# #!/usr/bin/env python3
# import os
# import argparse
# import random
# import numpy as np
# import pandas as pd
# from contextlib import contextmanager
# from joblib import Parallel, delayed, parallel as joblib_parallel
# from tqdm import tqdm
# from compound_model import compounding


# def process_sample(pc_path: str, dest_path: str, noise_model) -> None:
#     """读取原始点云，做模拟，写入目标路径"""
#     pc = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)
#     xyz, intensity = pc[:, :3], pc[:, 3]
#     if intensity.max() < 1.0:
#         intensity *= noise_model.lidar.max_intensity

#     sim_xyz, sim_intensity = noise_model.simulate(xyz, intensity)
#     sim_pc = np.concatenate([sim_xyz, sim_intensity[:, None]], axis=1).astype(np.float32)
#     os.makedirs(os.path.dirname(dest_path), exist_ok=True)
#     sim_pc.tofile(dest_path)


# def _is_provided(x: str) -> bool:
#     return isinstance(x, str) and x.strip() != ""


# def _first_existing(paths):
#     for p in paths:
#         if p and os.path.isfile(p):
#             return p
#     return None


# def count_points_fast(bin_path: str) -> int:
#     """通过文件大小快速计算点数（每点16字节: 4×float32）。"""
#     size = os.path.getsize(bin_path)
#     if size % 16 != 0:
#         raise ValueError(f"{bin_path} 大小 {size} 非 16 的整数倍，可能不是有效KITTI点云")
#     return size // 16


# @contextmanager
# def tqdm_joblib(tqdm_object):
#     """让 joblib 的 Parallel 显示 tqdm 进度条。"""
#     class TqdmBatchCallback(joblib_parallel.BatchCompletionCallBack):
#         def __call__(self, *args, **kwargs):
#             tqdm_object.update(n=self.batch_size)
#             return super().__call__(*args, **kwargs)
#     old_callback = joblib_parallel.BatchCompletionCallBack
#     joblib_parallel.BatchCompletionCallBack = TqdmBatchCallback
#     try:
#         yield tqdm_object
#     finally:
#         joblib_parallel.BatchCompletionCallBack = old_callback
#         tqdm_object.close()


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         description="基于 CSV 筛选原始点云并生成指定天气模拟数据（自动路径；扁平化输出；去重；进度显示；drop评估）"
#     )
#     # CSV 筛选参数
#     parser.add_argument("--csv_path",     type=str, required=True, help="包含 name,weather,severity 列的 CSV 文件路径")
#     parser.add_argument("--csv_weather",  type=str, required=True, help="CSV 中用于筛选的 weather 值（如 Fog）")
#     parser.add_argument("--csv_severity", type=str, required=True, help="CSV 中用于筛选的 severity 值（如 light）")

#     # 可选覆盖原始/输出目录及 split
#     parser.add_argument("--raw_lidar_root", type=str, default=None, help="原始 raw_lidar 根目录（默认取 CSV 所在目录）")
#     parser.add_argument("--simulated_root", type=str, default=None, help="模拟点云输出根目录（默认 raw_lidar_root+'_simulated'）")
#     parser.add_argument("--split", type=str, choices=['train','val','test'], default=None, help="数据切分名称（默认取 CSV 文件名，如 'val'）")

#     # 模拟模型及配置
#     parser.add_argument("--sim_weather", type=str, required=True, help="模拟天气模型名（如 fog,rain,snow,haze 等）")
#     parser.add_argument("--noise", type=str, default="", help="内部噪声模型名（如 calib_error），可选（留空表示无噪声）")

#     # 传感器配置
#     parser.add_argument("--sensor_cfg",  type=str, required=True, help="激光雷达传感器配置文件路径")

#     # 其他
#     parser.add_argument("--seed",   type=int, default=42, help="随机种子（默认 42）")
#     parser.add_argument("--n_jobs", type=int, default=8,  help="并行作业数（默认 8）")
#     parser.add_argument("--eval_k", type=int, default=5,  help="sim 后随机抽样评估的帧数（默认 5）")
#     args = parser.parse_args()

#     # 设置随机性
#     random.seed(args.seed)
#     np.random.seed(args.seed)

#     # 自动推断路径
#     csv_dir, csv_file = os.path.split(args.csv_path)
#     csv_base = os.path.splitext(csv_file)[0]
#     raw_lidar_root = args.raw_lidar_root if _is_provided(args.raw_lidar_root) else csv_dir
#     split = args.split if _is_provided(args.split) else csv_base
#     simulated_root = args.simulated_root if _is_provided(args.simulated_root) else raw_lidar_root.rstrip('/\\') + '_simulated'

#     print(f"🔀 raw_lidar_root: {raw_lidar_root}")
#     print(f"🔀 split: {split}")
#     print(f"🔀 simulated_root: {simulated_root}")

#     # 1) 读取并过滤 CSV
#     df = pd.read_csv(args.csv_path)
#     df = df[(df['weather'] == args.csv_weather) & (df['severity'] == args.csv_severity)]
#     print(f"🚀 CSV 匹配记录: {len(df)} (weather={args.csv_weather}, severity={args.csv_severity})")

#     # 2) 去重 & 构造扁平化输出
#     samples = []
#     unique_map = {}  # bin_name -> pc_path
#     dup_count = 0

#     for _, row in df.iterrows():
#         name = row['name']
#         try:
#             prefix, suffix = name.split('_image_')
#         except ValueError:
#             print(f"[Skip] name 字段格式异常：{name}")
#             continue
#         frame_id = suffix.split('_')[-1]
#         bin_name = f"{prefix}_velodyne_points_{frame_id}.bin"

#         pc_path = os.path.join(raw_lidar_root, split, bin_name)
#         if not os.path.isfile(pc_path):
#             print(f"[Skip] 找不到原始文件: {pc_path}")
#             continue

#         if bin_name in unique_map:
#             dup_count += 1
#             continue
#         unique_map[bin_name] = pc_path

#     # 扁平化输出：<simulated_root>/<split>/<bin_name>
#     for bin_name, pc_path in unique_map.items():
#         dest_path = os.path.join(simulated_root, split, bin_name)
#         samples.append((pc_path, dest_path))

#     print(f"ℹ️ 去重后唯一 LiDAR 帧: {len(samples)}，其中去重 {dup_count}")
#     print(f"✅ 将模拟并写出 {len(samples)} 个点云文件（扁平化输出）")

#     # 3) 自动定位参数文件
#     base_dir = os.path.dirname(os.path.abspath(__file__))

#     weather_params = _first_existing([
#         os.path.join(base_dir, 'configs', 'weather_params', f"{args.sim_weather}.yaml"),
#         os.path.join(base_dir, 'configs', 'weather_params', f"{args.sim_weather}_params.yaml"),
#         # 兼容旧目录结构（如天气参数也放在 noise_params 下）
#         os.path.join(base_dir, 'configs', 'noise_params',   f"{args.sim_weather}.yaml"),
#         os.path.join(base_dir, 'configs', 'noise_params',   f"{args.sim_weather}_params.yaml"),
#     ])
#     if not weather_params:
#         raise FileNotFoundError(
#             f"未找到天气参数文件，请检查 configs/weather_params 或 noise_params 目录下是否存在\n"
#             f"  - {args.sim_weather}.yaml 或 {args.sim_weather}_params.yaml"
#         )

#     noise_params = ""
#     if _is_provided(args.noise):
#         noise_params = _first_existing([
#             os.path.join(base_dir, 'configs', 'noise_params', f"{args.noise}.yaml"),
#             os.path.join(base_dir, 'configs', 'noise_params', f"{args.noise}_params.yaml"),
#         ]) or ""
#         if noise_params == "":
#             raise FileNotFoundError(
#                 f"指定了噪声模型 '{args.noise}' 但未找到其参数文件 ({args.noise}.yaml 或 {args.noise}_params.yaml)"
#             )

#     print(f"🌥 使用天气模拟模型: {args.sim_weather} (params: {weather_params})")
#     if _is_provided(args.noise):
#         print(f"🔊 使用噪声模型: {args.noise} (params: {noise_params})")

#     # 4) 初始化模拟器
#     noise_model = compounding(
#         args.sim_weather, weather_params,
#         args.noise if _is_provided(args.noise) else "",
#         noise_params if _is_provided(noise_params) else "",
#         args.sensor_cfg
#     )

#     # 5) 并行处理 + 进度条
#     with tqdm_joblib(tqdm(total=len(samples), desc=f"Simulating {args.sim_weather}", unit="pc")):
#         Parallel(n_jobs=args.n_jobs)(
#             delayed(process_sample)(pc, dst, noise_model) for pc, dst in samples
#         )

#     print("🎉 全部样本处理完成。")

#     # 6) 随机抽样评估 drop 百分比
#     eval_k = min(args.eval_k, len(samples))
#     if eval_k <= 0:
#         raise SystemExit(0)

#     eval_samples = random.sample(samples, eval_k)
#     print(f"\n📊 随机抽样 {eval_k} 帧进行 drop 评估：")

#     retain_list, drop_list = [], []
#     for pc_path, dst_path in eval_samples:
#         try:
#             before = count_points_fast(pc_path)
#             after  = count_points_fast(dst_path)
#             retain = after / before if before > 0 else 0.0
#             drop   = 1.0 - retain
#             retain_list.append(retain)
#             drop_list.append(drop)
#             print(f"  - {os.path.basename(dst_path)} | before={before:7d} after={after:7d} | retain={retain*100:6.2f}% drop={drop*100:6.2f}%")
#         except Exception as e:
#             print(f"  - {os.path.basename(dst_path)} | 评估失败：{e}")

#     if retain_list:
#         avg_retain = np.mean(retain_list) * 100
#         avg_drop   = np.mean(drop_list)   * 100
#         print(f"\n📈 平均保留率: {avg_retain:.2f}%  | 平均 drop: {avg_drop:.2f}%")

#!/usr/bin/env python3
import os
import argparse
import random
import numpy as np
import pandas as pd
from contextlib import contextmanager
from joblib import Parallel, delayed, parallel as joblib_parallel
from tqdm import tqdm
from compound_model import compounding


def process_sample(pc_path: str, dest_path: str, noise_model) -> bool:
    """读取原始点云，做模拟，写入目标路径。成功返回 True，失败返回 False。"""
    try:
        pc = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)
        xyz, intensity = pc[:, :3], pc[:, 3]
        if intensity.max() < 1.0:
            intensity *= noise_model.lidar.max_intensity

        sim_xyz, sim_intensity = noise_model.simulate(xyz, intensity)
        sim_pc = np.concatenate([sim_xyz, sim_intensity[:, None]], axis=1).astype(np.float32)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        sim_pc.tofile(dest_path)
        return True
    except Exception as e:
        print(f"[Fail] {os.path.basename(dest_path)}: {e}")
        return False


def _is_provided(x: str) -> bool:
    return isinstance(x, str) and x.strip() != ""


def _first_existing(paths):
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None


def count_points_fast(bin_path: str) -> int:
    """通过文件大小快速计算点数（每点16字节: 4×float32）。"""
    size = os.path.getsize(bin_path)
    if size % 16 != 0:
        raise ValueError(f"{bin_path} 大小 {size} 非 16 的整数倍，可能不是有效KITTI点云")
    return size // 16


@contextmanager
def tqdm_joblib(tqdm_object):
    """让 joblib 的 Parallel 显示 tqdm 进度条。"""
    class TqdmBatchCallback(joblib_parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)
    old_callback = joblib_parallel.BatchCompletionCallBack
    joblib_parallel.BatchCompletionCallBack = TqdmBatchCallback
    try:
        yield tqdm_object
    finally:
        joblib_parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="基于 CSV 筛选原始点云并生成指定天气模拟数据（自动路径；扁平化输出；去重；进度显示；drop评估；汇总统计）"
    )
    # CSV 筛选参数
    parser.add_argument("--csv_path",     type=str, required=True, help="包含 name,weather,severity 列的 CSV 文件路径")
    parser.add_argument("--csv_weather",  type=str, required=True, help="CSV 中用于筛选的 weather 值（如 Fog）")
    parser.add_argument("--csv_severity", type=str, required=True, help="CSV 中用于筛选的 severity 值（如 light）")

    # 可选覆盖原始/输出目录及 split
    parser.add_argument("--raw_lidar_root", type=str, default=None, help="原始 raw_lidar 根目录（默认取 CSV 所在目录）")
    parser.add_argument("--simulated_root", type=str, default=None, help="模拟点云输出根目录（默认 raw_lidar_root+'_simulated'）")
    parser.add_argument("--split", type=str, choices=['train','val','test'], default=None, help="数据切分名称（默认取 CSV 文件名，如 'val'）")

    # 模拟模型及配置
    parser.add_argument("--sim_weather", type=str, required=True, help="模拟天气模型名（如 fog,rain,snow,haze 等）")
    parser.add_argument("--noise", type=str, default="", help="内部噪声模型名（如 calib_error），可选（留空表示无噪声）")

    # 传感器配置
    parser.add_argument("--sensor_cfg",  type=str, required=True, help="激光雷达传感器配置文件路径")

    # 其他
    parser.add_argument("--seed",   type=int, default=42, help="随机种子（默认 42）")
    parser.add_argument("--n_jobs", type=int, default=8,  help="并行作业数（默认 8）")
    parser.add_argument("--eval_k", type=int, default=100,  help="sim 后随机抽样评估的帧数（默认 5）")
    args = parser.parse_args()

    # 设置随机性
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 自动推断路径
    csv_dir, csv_file = os.path.split(args.csv_path)
    csv_base = os.path.splitext(csv_file)[0]
    raw_lidar_root = args.raw_lidar_root if _is_provided(args.raw_lidar_root) else csv_dir
    split = args.split if _is_provided(args.split) else csv_base
    simulated_root = args.simulated_root if _is_provided(args.simulated_root) else raw_lidar_root.rstrip('/\\') + '_simulated'

    print(f"🔀 raw_lidar_root: {raw_lidar_root}")
    print(f"🔀 split: {split}")
    print(f"🔀 simulated_root: {simulated_root}")

    # 1) 读取并过滤 CSV
    df = pd.read_csv(args.csv_path)
    df = df[(df['weather'] == args.csv_weather) & (df['severity'] == args.csv_severity)]
    print(f"🚀 CSV 匹配记录: {len(df)} (weather={args.csv_weather}, severity={args.csv_severity})")

    # 2) 去重 & 构造扁平化输出
    samples = []
    unique_map = {}  # bin_name -> pc_path
    dup_count = 0

    for _, row in df.iterrows():
        name = row['name']
        try:
            prefix, suffix = name.split('_image_')
        except ValueError:
            print(f"[Skip] name 字段格式异常：{name}")
            continue
        frame_id = suffix.split('_')[-1]
        bin_name = f"{prefix}_velodyne_points_{frame_id}.bin"

        pc_path = os.path.join(raw_lidar_root, split, bin_name)
        if not os.path.isfile(pc_path):
            print(f"[Skip] 找不到原始文件: {pc_path}")
            continue

        if bin_name in unique_map:
            dup_count += 1
            continue
        unique_map[bin_name] = pc_path

    # 扁平化输出：<simulated_root>/<split>/<bin_name>
    for bin_name, pc_path in unique_map.items():
        dest_path = os.path.join(simulated_root, split, bin_name)
        samples.append((pc_path, dest_path))

    print(f"ℹ️ 去重后唯一 LiDAR 帧: {len(samples)}，其中去重 {dup_count}")

    # # 3) 自动定位参数文件
    # base_dir = os.path.dirname(os.path.abspath(__file__))

    # weather_params = _first_existing([
    #     os.path.join(base_dir, 'configs', 'weather_params', f"{args.sim_weather}.yaml"),
    #     os.path.join(base_dir, 'configs', 'weather_params', f"{args.sim_weather}_params.yaml"),
    #     # 兼容旧目录结构（如天气参数也放在 noise_params 下）
    #     os.path.join(base_dir, 'configs', 'noise_params',   f"{args.sim_weather}.yaml"),
    #     os.path.join(base_dir, 'configs', 'noise_params',   f"{args.sim_weather}_params.yaml"),
    # ])
    # if not weather_params:
    #     raise FileNotFoundError(
    #         f"未找到天气参数文件，请检查 configs/weather_params 或 noise_params 目录下是否存在\n"
    #         f"  - {args.sim_weather}.yaml 或 {args.sim_weather}_params.yaml"
    #     )

    # noise_params = ""
    # if _is_provided(args.noise):
    #     noise_params = _first_existing([
    #         os.path.join(base_dir, 'configs', 'noise_params', f"{args.noise}.yaml"),
    #         os.path.join(base_dir, 'configs', 'noise_params', f"{args.noise}_params.yaml"),
    #     ]) or ""
    #     if noise_params == "":
    #         raise FileNotFoundError(
    #             f"指定了噪声模型 '{args.noise}' 但未找到其参数文件 ({args.noise}.yaml 或 {args.noise}_params.yaml)"
    #         )

    # print(f"🌥 使用天气模拟模型: {args.sim_weather} (params: {weather_params})")
    # if _is_provided(args.noise):
    #     print(f"🔊 使用噪声模型: {args.noise} (params: {noise_params})")

    # 3) 自动定位参数文件
    base_dir = os.path.dirname(os.path.abspath(__file__))

    def _normalize_sim_weather(sim_weather_raw: str):
        """从 sim_weather 中提取主天气名，并指示是否包含湿地面附加模块。"""
        w = (sim_weather_raw or "").strip().lower().replace("-", "_")
        tokens = [t for t in w.split("_") if t]
        # 这些 token 代表湿地面附加模块
        wet_tokens = {"wet", "wetground", "ground"}
        has_wet = any(t in wet_tokens for t in tokens)
        # 主天气 = 去掉 wet 相关 token 后的剩余部分（可能为空）
        main_tokens = [t for t in tokens if t not in wet_tokens]
        main_weather = "_".join(main_tokens)
        return main_weather, has_wet

    main_weather, has_wet = _normalize_sim_weather(args.sim_weather)

    # —— 主天气参数文件：按“主天气名”去找，而不是用原始 sim_weather —— #
    weather_params = None
    if _is_provided(main_weather):
        weather_params = _first_existing([
            os.path.join(base_dir, 'configs', 'weather_params', f"{main_weather}.yaml"),
            os.path.join(base_dir, 'configs', 'weather_params', f"{main_weather}_params.yaml"),
            # 兼容旧目录结构
            os.path.join(base_dir, 'configs', 'noise_params',   f"{main_weather}.yaml"),
            os.path.join(base_dir, 'configs', 'noise_params',   f"{main_weather}_params.yaml"),
        ])
        if not weather_params:
            raise FileNotFoundError(
                "未找到天气参数文件，请检查 configs/weather_params 或 noise_params 目录下是否存在\n"
                f"  - {main_weather}.yaml 或 {main_weather}_params.yaml"
            )

    # 噪声参数维持原逻辑
    noise_params = ""
    if _is_provided(args.noise):
        noise_params = _first_existing([
            os.path.join(base_dir, 'configs', 'noise_params', f"{args.noise}.yaml"),
            os.path.join(base_dir, 'configs', 'noise_params', f"{args.noise}_params.yaml"),
        ]) or ""
        if noise_params == "":
            raise FileNotFoundError(
                f"指定了噪声模型 '{args.noise}' 但未找到其参数文件 ({args.noise}.yaml 或 {args.noise}_params.yaml)"
            )

    # 友好日志
    if _is_provided(main_weather):
        print(f"🌥 主天气模型: {main_weather} (params: {weather_params})")
    if has_wet:
        print(f"💧 叠加湿地面模型: wet_ground (params: configs/noise_params/wet_ground*.yaml 由 compound_model 自动加载)")

    # 4) 初始化模拟器
    noise_model = compounding(
        args.sim_weather,  # 仍然传原始串，如 "rain_wet_ground"
        weather_params,    # 这是按主天气名找到的 rain.yaml
        args.noise if _is_provided(args.noise) else "",
        noise_params if _is_provided(noise_params) else "",
        args.sensor_cfg
    )

    # 5) 并行处理 + 进度条
    with tqdm_joblib(tqdm(total=len(samples), desc=f"Simulating {args.sim_weather}", unit="pc")):
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(process_sample)(pc, dst, noise_model) for pc, dst in samples
        )

    # 6) 统计汇总：计划/成功/失败
    total_planned = len(samples)
    total_success = sum(1 for r in results if r)
    total_failed  = total_planned - total_success
    print(f"\n📦 汇总 | 计划处理: {total_planned} | ✅ 成功生成: {total_success} | ❌ 失败: {total_failed}")

    # 7) 随机抽样评估 drop 百分比（仅在成功样本上）
    success_pairs = [(pc, dst) for (pc, dst), ok in zip(samples, results) if ok and os.path.isfile(dst)]
    if not success_pairs:
        print("⚠️ 无成功生成的样本，跳过 drop 评估。")
        raise SystemExit(0)

    eval_k = min(args.eval_k, len(success_pairs))
    if eval_k <= 0:
        raise SystemExit(0)

    eval_samples = random.sample(success_pairs, eval_k)
    print(f"\n📊 随机抽样 {eval_k}/{len(success_pairs)} 帧进行 drop 评估：")

    retain_list, drop_list = [], []
    for pc_path, dst_path in eval_samples:
        try:
            before = count_points_fast(pc_path)
            after  = count_points_fast(dst_path)
            retain = after / before if before > 0 else 0.0
            drop   = 1.0 - retain
            retain_list.append(retain)
            drop_list.append(drop)
            print(f"  - {os.path.basename(dst_path)} | before={before:7d} after={after:7d} | retain={retain*100:6.2f}% drop={drop*100:6.2f}%")
        except Exception as e:
            print(f"  - {os.path.basename(dst_path)} | 评估失败：{e}")

    if retain_list:
        avg_retain = np.mean(retain_list) * 100
        avg_drop   = np.mean(drop_list)   * 100
        print(f"\n📈 平均保留率: {avg_retain:.2f}%  | 平均 drop: {avg_drop:.2f}%")
