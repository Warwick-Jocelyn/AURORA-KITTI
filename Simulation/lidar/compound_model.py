# import importlib
# import os
# from lidar import Lidar
# from util import load_yaml

# def _is_provided(x):
#     return isinstance(x, str) and x.strip() != ""

# def _first_existing(paths):
#     for p in paths:
#         if p and os.path.isfile(p):
#             return p
#     return None

# class compounding:
#     def __init__(self, weather, weather_params_path, noise, noise_params_path, sensor_cfg_path):
#         # ---- wet ground 处理 ----
#         wet_ground_cls_path = None
#         wet_ground_params = None
#         if _is_provided(weather) and 'wet' in weather:
#             wet_ground_cls_path = "weather_models.wet_ground"
#             # remove 'wet' and '_' from the weather name
#             weather = weather.replace("wet", "").replace("_", "").strip()

#         # ---- 规范化（允许 None / ""）----
#         weather = weather if _is_provided(weather) else ""
#         noise   = noise   if _is_provided(noise)   else ""

#         # ---- 解析参数文件路径（自动回退）----
#         # 天气参数优先级：显式 -> configs/weather_params/{weather}.yaml -> {weather}_params.yaml -> 兼容旧路径 noise_params/...
#         if _is_provided(weather):
#             weather_params_path = (
#                 weather_params_path if _is_provided(weather_params_path) else
#                 _first_existing([
#                     os.path.join("configs", "weather_params", f"{weather}.yaml"),
#                     os.path.join("configs", "weather_params", f"{weather}_params.yaml"),
#                     # 兼容旧项目结构（如果历史上把天气参数也放在 noise_params）
#                     os.path.join("configs", "noise_params",   f"{weather}.yaml"),
#                     os.path.join("configs", "noise_params",   f"{weather}_params.yaml"),
#                 ])
#             )
#         else:
#             weather_params_path = None

#         # 噪声参数优先级：显式 -> configs/noise_params/{noise}.yaml -> {noise}_params.yaml
#         if _is_provided(noise):
#             noise_params_path = (
#                 noise_params_path if _is_provided(noise_params_path) else
#                 _first_existing([
#                     os.path.join("configs", "noise_params", f"{noise}.yaml"),
#                     os.path.join("configs", "noise_params", f"{noise}_params.yaml"),
#                 ])
#             )
#         else:
#             noise_params_path = None

#         # wet_ground 参数（如需）
#         if wet_ground_cls_path is not None:
#             wet_ground_params = load_yaml(
#                 _first_existing([
#                     os.path.join("configs", "noise_params", "wet_ground.yaml"),
#                     os.path.join("configs", "noise_params", "wet_ground_params.yaml"),
#                 ])
#             )

#         # 传感器配置：显式 -> 默认 ouster
#         if _is_provided(sensor_cfg_path):
#             sensor_cfg = load_yaml(sensor_cfg_path)
#         else:
#             sensor_cfg = load_yaml(os.path.join("configs", "sensor_params", "lidar_ouster_config.yaml"))

#         # 实例化 Lidar
#         self.lidar = Lidar(sensor_cfg)

#         # ---- 实例化天气模型（非空才导入）----
#         if _is_provided(weather):
#             weather_cls_path = "weather_models." + weather
#             weather_params = load_yaml(weather_params_path) if weather_params_path else None
#             self.weather_class = getattr(
#                 importlib.import_module(weather_cls_path),
#                 f"lidar_{weather}"
#             )(weather_params, self.lidar)
#         else:
#             self.weather_class = None

#         # ---- 实例化噪声模型（非空才导入）----
#         if _is_provided(noise):
#             noise_cls_path = "internal_noise_models." + noise
#             noise_params = load_yaml(noise_params_path) if noise_params_path else None
#             self.noise_class = getattr(
#                 importlib.import_module(noise_cls_path),
#                 f"lidar_{noise}"
#             )(noise_params, self.lidar)
#         else:
#             self.noise_class = None

#         # ---- 实例化 wet_ground（如需）----
#         if wet_ground_cls_path is not None:
#             self.wet_ground_class = getattr(
#                 importlib.import_module(wet_ground_cls_path),
#                 "lidar_wet_ground"
#             )(wet_ground_params, self.lidar)
#         else:
#             self.wet_ground_class = None

#     def simulate(self, pcl, intensity):
#         # filter out zero depth points
#         pcl, intensity, _ = self.lidar.filter_out_zero_depth(pcl, intensity)
#         if self.weather_class is not None and self.noise_class is not None:
#             if self.noise_class.order == 1:
#                 pcl, intensity = self.noise_class.simulate(pcl, intensity)
#                 pcl, intensity = self.weather_class.simulate(pcl, intensity)
#             elif self.noise_class.order == 2:
#                 pcl, intensity = self.weather_class.simulate(pcl, intensity)
#                 pcl, intensity = self.noise_class.simulate(pcl, intensity)
#             else:
#                 raise ValueError("Not implemented")
#         else:
#             if self.weather_class is not None:
#                 pcl, intensity = self.weather_class.simulate(pcl, intensity)
#             elif self.noise_class is not None:
#                 pcl, intensity = self.noise_class.simulate(pcl, intensity)
#         if self.wet_ground_class is not None:
#             pcl, intensity = self.wet_ground_class.simulate(pcl, intensity)
#         return pcl, intensity

# compound_model.py

import importlib
import os
from lidar import Lidar
from util import load_yaml

def _is_provided(x):
    return isinstance(x, str) and x.strip() != ""

def _first_existing(paths):
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None

class compounding:
    def __init__(self, weather, weather_params_path, noise, noise_params_path, sensor_cfg_path):
        # ---- 规范化输入 ----
        weather = (weather or "").strip().lower().replace("-", "_")
        noise   = (noise   or "").strip().lower().replace("-", "_")

        # ---- 识别 wet_ground 作为附加模块，而非主天气 ----
        wet_ground_cls_path = None
        wet_ground_params = None
        has_wet_ground = False

        if _is_provided(weather) and ("wet" in weather):
            # 兼容几种写法：wet / wet_ground / rain_wet / rain_wet_ground
            tokens = [t for t in weather.split("_") if t]   # e.g. ["rain","wet","ground"]
            tokens_wo_wet = [t for t in tokens if t not in ("wet", "wetground", "ground")]
            # 只要出现过 wet/wet_ground，就加载 wet_ground 附加模块
            has_wet_ground = True
            wet_ground_cls_path = "weather_models.wet_ground"
            # 剩余 token 作为“主天气”名称（可能为空）
            weather = "_".join(tokens_wo_wet)

        # ---- 解析参数文件路径（主天气与噪声）----
        if _is_provided(weather):
            weather_params_path = (
                weather_params_path if _is_provided(weather_params_path) else
                _first_existing([
                    os.path.join("configs", "weather_params", f"{weather}.yaml"),
                    os.path.join("configs", "weather_params", f"{weather}_params.yaml"),
                    # 兼容旧项目结构
                    os.path.join("configs", "noise_params",   f"{weather}.yaml"),
                    os.path.join("configs", "noise_params",   f"{weather}_params.yaml"),
                ])
            )
        else:
            weather_params_path = None

        if _is_provided(noise):
            noise_params_path = (
                noise_params_path if _is_provided(noise_params_path) else
                _first_existing([
                    os.path.join("configs", "noise_params", f"{noise}.yaml"),
                    os.path.join("configs", "noise_params", f"{noise}_params.yaml"),
                ])
            )
        else:
            noise_params_path = None

        # ---- wet_ground 参数 ----
        if has_wet_ground:
            wet_ground_params = load_yaml(
                _first_existing([
                    os.path.join("configs", "noise_params", "wet_ground.yaml"),
                    os.path.join("configs", "noise_params", "wet_ground_params.yaml"),
                ])
            )

        # ---- 传感器配置 ----
        if _is_provided(sensor_cfg_path):
            sensor_cfg = load_yaml(sensor_cfg_path)
        else:
            sensor_cfg = load_yaml(os.path.join("configs", "sensor_params", "lidar_ouster_config.yaml"))

        # ---- 实例化 Lidar ----
        self.lidar = Lidar(sensor_cfg)

        # ---- 实例化主天气（只有在有合法主天气名时才加载）----
        if _is_provided(weather):
            weather_cls_path = "weather_models." + weather
            weather_params = load_yaml(weather_params_path) if weather_params_path else None
            self.weather_class = getattr(
                importlib.import_module(weather_cls_path),
                f"lidar_{weather}"
            )(weather_params, self.lidar)
        else:
            self.weather_class = None

        # ---- 实例化内部噪声 ----
        if _is_provided(noise):
            noise_cls_path = "internal_noise_models." + noise
            noise_params = load_yaml(noise_params_path) if noise_params_path else None
            self.noise_class = getattr(
                importlib.import_module(noise_cls_path),
                f"lidar_{noise}"
            )(noise_params, self.lidar)
        else:
            self.noise_class = None

        # ---- 实例化 wet_ground 附加模块 ----
        if has_wet_ground:
            self.wet_ground_class = getattr(
                importlib.import_module(wet_ground_cls_path),
                "lidar_wet_ground"
            )(wet_ground_params, self.lidar)
        else:
            self.wet_ground_class = None

    def simulate(self, pcl, intensity):
        # filter out zero depth points
        pcl, intensity, _ = self.lidar.filter_out_zero_depth(pcl, intensity)

        # 噪声与天气的先后次序受 noise_class.order 控制
        if self.weather_class is not None and self.noise_class is not None:
            if self.noise_class.order == 1:
                pcl, intensity = self.noise_class.simulate(pcl, intensity)
                pcl, intensity = self.weather_class.simulate(pcl, intensity)
            elif self.noise_class.order == 2:
                pcl, intensity = self.weather_class.simulate(pcl, intensity)
                pcl, intensity = self.noise_class.simulate(pcl, intensity)
            else:
                raise ValueError("Not implemented")
        else:
            if self.weather_class is not None:
                pcl, intensity = self.weather_class.simulate(pcl, intensity)
            elif self.noise_class is not None:
                pcl, intensity = self.noise_class.simulate(pcl, intensity)

        # 最后叠加湿地面
        if self.wet_ground_class is not None:
            pcl, intensity = self.wet_ground_class.simulate(pcl, intensity)

        return pcl, intensity
