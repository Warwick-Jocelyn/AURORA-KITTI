import numpy as np

## Initialise Classes for camera and weather
class Cam_Param:
    # Default parameters for Carissma Lucid Triton TRI028S-CC + Universe Optics BL060C (Lucid UC060-5M lens)
    def __init__(self, etime = float(0.03), fov = np.array([80.8, 61.6]), res = np.array([1936, 1464]), pix = np.array([4.5, 4.5]), flen = float(6.0), f_dot = float(2.1), min_d = float(0.1), height = float(1)):
        self.etime = etime              # in s
        self.fov = np.deg2rad(fov)      # [hFov, vFov]
        self.res = res                  # [hRes, vRes]
        self.pix = pix                  # Pixel size in um [H V]
        self.flen = flen                # Focal length in mm
        self.f_dot = f_dot              # Aperture
        self.min_d = min_d              # Min Focus Distance in m
        self.height = height            # Height of sensor above ground

'''
        # Camera Parameters - based on Lucid triton TRI054-S, Kowa LM3JC10M lens
        self.etime = float(0.003)                # in s
        self.fov = np.array([100.2, 83.7])      # [hFoV, vFoV]
        self.res = np.array([2880, 1860])       # [hRes, vRes]
        self.pix = np.array([3, 3])             # Pixel size in um [H V]
        self.flen = float(3.7)               # Focal length in mm
        self.mind = float(0.1)               # Min focus distance in m
'''
class Cam_Sek_Param:
    # Based on Sekonix SF3325 RCCB (NA6062 lens)
    def __init__(self, etime = float(0.002), fov = np.array([60.0, 38.0]), res = np.array([1920, 1208]), pix = np.array([0, 0]), flen = float(5.49), min_d = float(0.1), height = float(1)):
        self.etime = etime              # in s
        self.fov = np.deg2rad(fov)      # [hFov, vFov]
        self.res = res                  # [hRes, vRes]
        self.pix = pix                  # Pixel size in um [H V]
        self.flen = flen                # Focal length in mm
        self.min_d = min_d              # Min Focus Distance in m
        self.height = height            # Height of sensor above ground

class Cam_ACDC_Param:
    # Parameters based on GOPro hero 5 used in ACDC dataset
    def __init__(self, etime = float(0.03), fov = np.array([118.2, 69.5]), res = np.array([1920, 1080]), pix = np.array([4.5, 4.5]), flen = float(14.0), f_dot = float(2.1), min_d = float(0.1), height = float(1)):
        self.etime = etime              # in s
        self.fov = np.deg2rad(fov)      # [hFov, vFov]
        self.res = res                  # [hRes, vRes]
        self.pix = pix                  # Pixel size in um [H V]
        self.flen = flen                # Focal length in mm
        self.f_dot = f_dot              # Aperture
        self.min_d = min_d              # Min Focus Distance in m
        self.height = height            # Height of sensor above ground

        sensor_size = np.array([6.17, 4.55])    # Sensor size for GoPro (not needed!)
        self.pix = self.res/sensor_size

class Cam_Kitti:
    # Default parameters for Carissma Lucid Triton TRI028S-CC + Universe Optics BL060C (Lucid UC060-5M lens)
    def __init__(self, etime = float(0.03), fov = np.array([90, 38.0]), res = np.array([1242, 375]), pix = np.array([0, 0]), flen = float(6.0), f_dot = float(2.1), min_d = float(0.1), height = float(1)):

    # def __init__(self, etime = float(0.003), fov = np.array([60.0, 38.0]), res = np.array([1242, 375]), pix = np.array([0, 0]), flen = float(3.0), f_dot = float(2.1), min_d = float(0.1), height = float(1)):
    # def __init__(self,ß
    #              etime=float(0.002),                     # 曝光时间，最大 2 ms :contentReference[oaicite:0]{index=0}
    #              fov=np.deg2rad(np.array([81.6, 29.2])), # [hFov, vFov]，分别约 81.6° 和 29.2°
    #              res=np.array([1242, 375]),              # 裁剪后分辨率 [宽, 高]
    #              pix=np.array([4.65, 4.65]),             # 像素大小 4.65 μm :contentReference[oaicite:1]{index=1}
    #              flen=float(3.35),                       # 焦距 ≈ fx·pix = 718.856·4.65μm ≈ 3.35mm :contentReference[oaicite:2]{index=2}
    #              f_dot=float(1.4),                       # 镜头最大光圈 f/1.4（NT59-917 标配） 
    #              min_d=float(0.1),                       # 最短对焦距离，取典型值 0.1m
    #              height=float(1.68)                      # 相机中心距地高度 ≈ 1.68m :contentReference[oaicite:3]{index=3}
    #              ):
        self.etime = etime              # in s
        self.fov = np.deg2rad(fov)      # [hFov, vFov]
        self.res = res                  # [hRes, vRes]
        self.pix = pix                  # Pixel size in um [H V]
        self.flen = flen                # Focal length in mm
        self.f_dot = f_dot              # Aperture
        self.min_d = min_d              # Min Focus Distance in m
        self.height = height            # Height of sensor above ground