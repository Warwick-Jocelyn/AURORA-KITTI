import numpy as np
import cv2
import os
import importlib
import numpy as np
import cv2
from scipy.sparse import csr_matrix
from numba import jit, prange
from time import time


def split_stk(speed, angle):
    vel_vert = speed * np.cos(angle)
    vel_hor = speed * np.sin(angle)
    return vel_vert, vel_hor


def lookup_tbl(weather, cam, wind_speed):
    # Create the lookup tables
    rain_dia = np.linspace(weather.dia_range[0], weather.dia_range[1], num = weather.dia_step_num)                      # in mm array of sizes to test
    rain_dis = np.linspace(weather.dis_range[0], weather.dis_range[1], num = weather.dis_step_num)

    # Initialise look-up tables
    tbl_stk = np.zeros(shape = [rain_dis.shape[0], rain_dia.shape[0]])
    tbl_hor = tbl_stk.copy()
    tbl_time = tbl_stk.copy()

    for idx_i, dia in enumerate(rain_dia):
        # Set parameters per Diameter of rain
        # Speed of Raindrop
        #vel_rain = weather.velocity(Diameter = dia)
        vel_rain, angle = weather.terminal_velocity(1e-3 * dia / 2, wind_speed)

        for idx_j, dis in enumerate(rain_dis):
            pix_height = ((2 * dis * np.tan(cam.fov[1] / 2)) / cam.res[1])

            num_pix_vert = dia * 1e-3 / pix_height
            vel_rain_pix = vel_rain / pix_height

            tbl_stk[idx_j, idx_i] = (vel_rain_pix * cam.etime) + num_pix_vert
            tbl_hor[idx_j, idx_i] = dia * 1e-3 / ((2 * dis * np.tan(cam.fov[0] / 2)) / cam.res[0])
            if tbl_stk[idx_j, idx_i] < 1:
                tbl_time[idx_j, idx_i] = cam.etime
            elif tbl_stk[idx_j, idx_i] < num_pix_vert + 1:
                tbl_time[idx_j, idx_i] = tbl_stk[idx_j, idx_i] / vel_rain_pix
            # if tbl_stk[idx_j,idx_i] < num_pix_vert * 1.5:
                # tbl_time[idx_j, idx_i] = cam.etime
            else:
                tbl_time[idx_j, idx_i] = (1 + num_pix_vert) / vel_rain_pix

    tbl_stk_vert, tbl_stk_hor = split_stk(tbl_stk, angle)

    return tbl_stk_vert, tbl_stk_hor, tbl_hor, tbl_time, rain_dia


def integration_trap(size, weather):
    # Integrate each step size for number of drops
    step = weather.dia_step_size
    n_drop_size = np.zeros(len(size))           # Drops array for all size
    n_drop = int(0)                             # Total drop number
    for idx in range(len(size)):
        n_drop_size[idx] = step * ((weather.drops_dist(drop_size = (size[idx] - (step / 2))) + weather.drops_dist(drop_size = (size[idx] + (step / 2)))) / 2)
        n_drop = n_drop + n_drop_size[idx]
    drop_ratio = np.asarray(n_drop_size / n_drop)
    return n_drop, drop_ratio


def fov_cuboid(range, cam):
    # Compute Volume to create rain drop
    # Assumes ground flat and parallel
    vol = (2 * range * np.tan(cam.fov[0] / 2)) * ((range * np.tan(cam.fov[1] / 2)) + cam.height) * range 

    return vol


def rnd_loc_cuboid(num_rain_size, max_d, cam, weather):
    # Random location generation for cuboid volume
    rain_drops = np.sum(num_rain_size, dtype=int)
    rain_loc = np.zeros(shape= [rain_drops, 4], dtype=np.double)                           # Stores pointers to steps in lookup table
    half_width = max_d * np.tan(cam.fov[0] / 2) * 1000                                       # At max range in mm half due to centerpoint
    half_height = max_d * np.tan(cam.fov[1] / 2) * 1000                                       # At max range in mm half due to centerpoint
    idx_j = int(0)
    for idx_i in range(len(num_rain_size)):
        rain_loc[idx_j:(idx_j + num_rain_size[idx_i]), 0] = np.full((num_rain_size[idx_i]), idx_i)                                                            # Drop size pointer
        rain_loc[idx_j:(idx_j + num_rain_size[idx_i]), 1] = (np.random.default_rng().integers(-half_width, half_width, size=(num_rain_size[idx_i]))) / 1000   # Horizontal location Back to meters
        rain_loc[idx_j:(idx_j + num_rain_size[idx_i]), 2] = (np.random.default_rng().integers(-cam.height * 1000, half_height, size=(num_rain_size[idx_i]))) / 1000  # Vertical starting locationback to meters
        rain_loc[idx_j:(idx_j + num_rain_size[idx_i]), 3] = np.random.default_rng().integers(0, weather.dis_step_num, size=(num_rain_size[idx_i]))                                 # uniform distribution for "Parallel" distance
        idx_j = idx_j + num_rain_size[idx_i]
    return rain_loc


def frame2pix(rain_loc, cam, weather):
    # Rain_Loc in [Drop size pointer, H_loc, V_loc, Distance pointer], please use fov_cuboid for this
    # Need to be updated if pixels are not even!
    # Takes rain coordinates frame and convert them into pixels. Assumes rain frame aligned to cam frame. Principle point in the centre of image
    # Convert rain pointer to real distance with hard coding
    distance = weather.dis_range[0] + (rain_loc[:, 3] * weather.dis_step_size)

    w_factor = (2 * np.tan(cam.fov[0] / 2)) / cam.res[0]                  # Horizonal length per pixel based on distance
    v_factor = (2 * np.tan(cam.fov[1] / 2)) / cam.res[1]                  # Vertical length per pixel based on distance
    w_pix = rain_loc[:, 1] / (distance * w_factor)                      # Location horizontally based on principle point
    v_pix = rain_loc[:, 2] / (distance * v_factor)                      # Location vertically based on principle point

    # Flip y axis to match camera pixel
    v_pix = np.negative(v_pix)

    # Find the principle point which is at the centre
    w_principle = cam.res[0] / 2
    v_principle = cam.res[1] / 2

    # Convert principle pixel to Image pixel, round down for pixel location, does not matter which part of pixel it is on
    rain_loc[:, 1] = np.floor(w_pix + w_principle)
    rain_loc[:, 2] = np.floor(v_pix + v_principle)

    # Remove drops outisde of H FoV and then V FoV
    rain_loc = np.delete(rain_loc, np.nonzero(np.bitwise_or(rain_loc[:, 1] < 0, rain_loc[:, 1] >= cam.res[0])), 0)
    rain_loc = np.delete(rain_loc, np.nonzero(np.bitwise_or(rain_loc[:, 2] < 0, rain_loc[:, 2] >= cam.res[1])), 0)

    rain_loc = rain_loc.astype(int)

    return rain_loc


@jit(nopython=True)
def bresenham_line_thick(x0, y0, x1, y1, thickness):
    """
    Bresenham's line algorithm with thickness support
    Returns arrays of x and y coordinates for the thick line
    """
    points_x = []
    points_y = []

    # Basic Bresenham's algorithm
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    x, y = x0, y0

    while True:
        # Add thickness around each point
        half_thick = thickness // 2
        for dy_offset in range(-half_thick, half_thick + 1):
            for dx_offset in range(-half_thick, half_thick + 1):
                points_x.append(x + dx_offset)
                points_y.append(y + dy_offset)

        if x == x1 and y == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

    return np.array(points_x), np.array(points_y)


@jit(nopython=True, parallel=True)
def optimized_rain_mask_v2_numba(rain_loc, depth_map, tbl_hor, tbl_stk_vert, tbl_stk_hor, tbl_time, etime, h, w):
    print(rain_loc.shape)
    att_mask = np.zeros((h, w), dtype=np.double)
    for idx_i in prange(rain_loc.shape[0]):
        # Depth check
        if rain_loc[idx_i, 3] > depth_map[rain_loc[idx_i, 2], rain_loc[idx_i, 1]]:
            continue

        hor = int(np.ceil(tbl_hor[rain_loc[idx_i, 3], rain_loc[idx_i, 0]]))
        vert = int(np.ceil(tbl_stk_vert[rain_loc[idx_i, 3], rain_loc[idx_i, 0]]))
        rel_time = tbl_time[rain_loc[idx_i, 3], rain_loc[idx_i, 0]] / etime 

        if tbl_hor[rain_loc[idx_i, 3], rain_loc[idx_i, 0]] <= 1:
            rel_time = pow(tbl_hor[rain_loc[idx_i, 3], rain_loc[idx_i, 0]],2) 

        start_x = rain_loc[idx_i, 1]
        start_y = rain_loc[idx_i, 2]

        end_x = start_x + int(np.floor(tbl_stk_hor[rain_loc[idx_i, 3], rain_loc[idx_i, 0]]))
        end_y = start_y + int(np.floor(vert)) - 1

        line_x, line_y = bresenham_line_thick(start_x, start_y, end_x, end_y, hor)

        for x, y in zip(line_x, line_y):
            if 0 <= x < w and 0 <= y < h:
                att_mask[y, x] += rel_time  # / (len(line_x))

    return att_mask


def attenuation_mask(raw_img, rain_loc, tbl_stk_vert, tbl_stk_hor, tbl_hor, tbl_time, depth_map, cam, weather, windspeed):
    # Create a mask based on raindrop to apply noise
    # Assumes rain at the centre of the pixel
    # Initialise the mask

    (h, w) = raw_img.shape[:2]
    att_mask = np.zeros((h, w), np.double)

    # Create attenuation matrix based on image to apply noise on right colour channel
    att_mat = weather.attenuation(img=raw_img)

    filter_perc = 0.0000                                                      # Percent of drop in decimals covered to ignore
    print("starting optimized loop", time())
    att_mask = optimized_rain_mask_v2_numba(rain_loc, depth_map, tbl_hor, tbl_stk_vert, tbl_stk_hor, tbl_time, cam.etime, h, w)
    print("ended optimized loop", time())

    # Apply the Atteunation
    # Blur
    att_mask = np.clip(att_mask, 0, 1)
    att_mask = cv2.GaussianBlur(att_mask, [3, 3], cv2.BORDER_TRANSPARENT)
    #raw_img = contrast(raw_img, 1)

    #att_mask = np.clip(att_mask, 0, 1)
    raw_img = np.clip((att_mask * att_mat) + ((1 - att_mask) * raw_img), 0, 255)

    # Cast array from float due to mask back to uint8
    raw_img = raw_img.astype(np.uint8)

    # Expand and save the mask image for Debugging
    #np.savetxt('./debug_img/00_snow.txt', att_mask)
    att_mask = np.ceil(att_mask * 255)
    att_mask = att_mask.astype(np.uint8)
    cv2.imwrite(os.path.join(os.getcwd(), 'debug', '04_att_mask.png'), att_mask)

    return raw_img


def wind_application(img, depth, weather_type, windspeed, weather_rate):
    np.random.seed(seed=1234)
    if weather_type == "cam_fog":
        fog_class = getattr(importlib.import_module("noise_class"), weather_type)
        weather = fog_class(visibility = 50, lum = 255)
        noisy_img = weather.application(img, depth)
        return noisy_img

    # cam_class = getattr(importlib.import_module("models.00_cam_class"), "Cam_Param")
    cam_class = getattr(importlib.import_module("models.00_cam_class"), "Cam_Kitti")
    weather_class = getattr(importlib.import_module("models.00_weather_class"), weather_type)
    cam = cam_class(etime=0.004)
    weather = weather_class(rate = weather_rate, dis_range=[1.5, 30.0])

    tbl_stk_vert, tbl_stk_hor, tbl_hor, tbl_time, dsize = lookup_tbl(weather, cam, windspeed)

    number_particle_unit, drop_ratio = integration_trap(dsize, weather)            # Drops per unit volume and ratio of drops in certain sizes
    number_particle = number_particle_unit * fov_cuboid(weather.dis_range[1], cam)

    # Determine location of particles in frame
    poi_particle = np.random.poisson(number_particle, 1)
    num_particle_size = poi_particle * drop_ratio
    num_particle_size = num_particle_size.astype(int)
    particle_loc = rnd_loc_cuboid(num_particle_size, weather.dis_range[1], cam, weather)
    particle_loc = frame2pix(particle_loc, cam, weather)

    noisy_img = attenuation_mask(img, particle_loc, tbl_stk_vert, tbl_stk_hor, tbl_hor, tbl_time, depth, cam, weather, windspeed)

    return noisy_img
