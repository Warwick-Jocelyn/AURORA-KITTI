import numpy as np
import cv2
import os
import importlib


from colour_demosaicing import (
    demosaicing_CFA_Bayer_Malvar2004,
)

###############################################################################################
class compounding:
    def __init__(self, sensor = 'cam', weather = 'cam_snow', noise = 'cam_occlu', mask_weather = True, mask_noise = True):
        self.weather = weather              # Weather model
        self.noise = noise                  # Model for compounding
        self.mask_w = mask_weather
        self.mask_n = mask_noise

        self.weather_class = getattr(importlib.import_module("noise_class"), self.weather)
        self.noise_class = getattr(importlib.import_module("noise_class"), self.noise) 


###############################################################################################
def BGR2Bay(colour_img):
    # Extract colour intensity per pixel based on a CFA pattern
    (h, w) = colour_img.shape[:2]
    (B, G, R) = cv2.split(colour_img)

    bay_img = np.empty((h, w), np.uint8)
    # Pattern of RGGB
    bay_img[0::2, 0::2] = R[0::2, 0::2]
    bay_img[1::2, 0::2] = G[1::2, 0::2]
    bay_img[0::2, 1::2] = G[0::2, 1::2]
    bay_img[1::2, 1::2] = B[1::2, 1::2]

    cv2.imwrite(os.path.join(os.getcwd(), 'debug_img/01_img_gray.png'), bay_img)
    return bay_img

def load_img(loc):
    # Load Image
    return BGR2Bay(cv2.imread(os.path.join(os.getcwd(), loc)))

def Bay2BGR(bay):
    #BGR_img = cv2.cvtColor(bay, cv2.COLOR_BayerBG2BGR)
    BGR_img = cv2.cvtColor(np.clip(demosaicing_CFA_Bayer_Malvar2004(bay), 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    return BGR_img

###############################################################################################
if __name__ == '__main__':
    import numpy as np
    
    # img_name = '03_car01_day01_clear_weather01_28m_45deg_frame_201'
    # img_bayer = load_img('input/'+ img_name + '.png')
    img_bayer = load_img('/path/to/aurora-root/Code/Cam_Weather_perf-optimization_numba/input_depth_anything/0000000006_clear.png')
    (img_h, img_w) = img_bayer.shape[:2]

    #depth = np.load(f"input/{img_name}.npy")
    # depth = np.loadtxt('input/' + img_name +'.txt')
    depth = np.loadtxt('/path/to/aurora-root/Code/Cam_Weather_perf-optimization_numba/input_depth_anything/depth.txt')
    print(depth.shape,img_bayer.shape)

    modelling = compounding(weather='cam_snow', noise='cam_wind')

    # weather = modelling.weather_class(res = np.array([img_w, img_h]))
    # 
    noise = modelling.noise_class(res = np.array([img_w, img_h]), weather_rate = 1)                                                                                      
    # noise = modelling.noise_class(res = np.array([img_w, img_h]), weather_rate = 3)

    compounded_img_bayer = noise.application(img_bayer, depth, modelling.weather)
    compounded_img_colour = Bay2BGR(compounded_img_bayer)

    cv2.imwrite(os.path.join('debug','01_Compounded_Colour.png'), compounded_img_colour)

    cv2.imwrite(os.path.join(os.getcwd(), 'debug','01_Compounded_Gray.png'), compounded_img_bayer)
    cv2.imwrite(os.path.join(os.getcwd(), 'debug','02_Compounded_Colour.png'), compounded_img_colour)