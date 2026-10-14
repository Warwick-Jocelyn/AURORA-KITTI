import numpy as np
import sys


class cam_rain:
    # Default parameters for rain
    def __init__(self, rate = float(10), dia_range = np.array([0.5, 6.0]), dia_step_size = float(0.1), dis_range = np.array([1.00, 30.00]), dis_step_size = float(0.01)):
        self.rate = rate                    # Rainrate in mm/h
        self.dia_range = dia_range          # Range of raindrop size [min max], in mm
        self.dia_step_size = dia_step_size  # Step size for raindrop in lookup table
        self.dis_range = dis_range          # Range of distances to spawn raindrops [min max], in m
        self.dis_step_size = dis_step_size  # Step size for distance in lookup table

        self.dia_step_num = int(1 + ((dia_range[1] - dia_range[0]) / dia_step_size))    # Number of steps for linspace in lookup table
        self.dis_step_num = int(1 + ((dis_range[1] - dis_range[0]) / dis_step_size))    # Number of steps for linspace in lookup table

    def velocity(self, Diameter):
        # Based on empirical power law 
        # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8936913 or https://journals.ametsoc.org/view/journals/apme/16/12/1520-0450_1977_016_1322_paairm_2_0_co_2.xml?tab_body=pdf 
        return 3.78 * (Diameter**0.67)

    def terminal_velocity(self, radius, wind):
        # Define terminal velocity of rain including the 
        # Approximate raindrop as circular and use the terminal velocity equation based on the two forces to dertermine absolute speed and direction.
        water_rho = 1000
        m = (4 / 3) * np.pi * np.power(radius, 3) * water_rho
        g = 9.81        # Meters per second
        rho = 1.204     # Density of environment medium, kg/m3
        XSA = np.pi * np.power(radius, 2)
        C_d = 0.47      # Drag coefficient of a sphere

        acc = np.power(g * g + wind * wind, 0.5)

        angle = np.arctan(wind / g)

        return np.power((2 * m * acc) / (rho * XSA * C_d), 0.5), angle

    def drops_dist(self, drop_size):
        return 8000 * np.exp(-4.1 * np.power(self.rate, -0.21) * drop_size)

    def attenuation(self, img):
        # Create attenuation matrix based on image to apply noise on right colour channel
        # return 255
        return ((0.94 * self.att_mask_bayer(img)) + (0.06 * self.att_mask_lum(img)))

    def att_mask_bayer(self, img):
        # For rain, average colour of the frame
        # Crop frame due to the limited rain area and enclosed
        (h, w) = img.shape[:2]
        cr_left = 0
        cr_right = img.shape[1]
        # Reduce the image by ignoring the bottom 2/5 of the image
        cr_bot = int(img.shape[0] - (2 * (img.shape[0]) / 5))
        img = img[0:cr_bot, cr_left:cr_right]

        # Compute the average colour in of each channel for the frame
        av_r = int(img[0::2, 0::2].mean())
        av_g = int((img[0::2, 1::2].mean() + img[1::2, 0::2].mean()) / 2)
        av_b = int(img[1::2, 1::2].mean())

        # Build the attenuation matrix
        att_mat = np.empty(shape = ([h, w]), dtype=np.double)
        att_mat[0::2, 0::2] = av_r
        att_mat[1::2, 0::2] = av_g
        att_mat[0::2, 1::2] = av_g
        att_mat[1::2, 1::2] = av_b

        return att_mat

    def att_mask_lum(self, img):
        # Comupte the Luminosity of frame
        # https://www.pixelsham.com/2020/12/26/exposure-value-measurements/
        # att_mat = 6000      # Roughly for a sunny day
        # att_mat = 737       # Heavy Overcast
        att_mat = 3237

        return att_mat


class cam_snow:
    # Default parameters for snow
    # Use same variables as rain so the framework code can be used as it is
    # Assume snow are spherical initially - to be updated to oval 3.03?
    def __init__(self, rate = float(9.03), dia_range = np.array([0.1, 10]), dia_step_size = float(0.1), dis_range = np.array([0.50, 30.09]), dis_step_size = float(0.01)):
        self.rate = rate                    # Rainrate in mm/h
        self.dia_range = dia_range          # Range of raindrop size [min max], in mm
        self.dia_step_size = dia_step_size  # Step size for raindrop in lookup table
        self.dis_range = dis_range          # Range of distances to spawn raindrops [min max], in m
        self.dis_step_size = dis_step_size  # Step size for distance in lookup table

        self.dia_step_num = int(1 + ((dia_range[1] - dia_range[0]) / dia_step_size))    # Number of steps for linspace in lookup table
        self.dis_step_num = int(1 + ((dis_range[1] - dis_range[0]) / dis_step_size))    # Number of steps for linspace in lookup table

    def velocity(self, Diameter):
        # Velocity based on "Fast Reactive Control for Illumination Through Rain and Snow"

        # Another accounting for snow properties can be found eq.9 https://journals.ametsoc.org/view/journals/apme/46/5/jam2489.1.xml
        velocity = 2.07 * np.power(Diameter/100, 0.31)

        return velocity

    def terminal_velocity(self, radius, wind):
        # Define terminal velocity of rain including the 
        # Approximate raindrop as circular and use the terminal velocity equation based on the two forces to dertermine absolute speed and direction.
        snow_rho = 70
        m = (4 / 3) * np.pi * np.power(radius, 3) * snow_rho
        g = 9.81        # Meters per second
        rho = 1.204     # Density of environment medium, kg/m3
        XSA = np.pi * np.power(radius, 2)
        C_d = 0.47      # Drag coefficient of a sphere

        acc = np.power(g * g + wind * wind, 0.5)

        angle = np.arctan(wind / g)

        return np.power((2 * m * acc) / (rho * XSA * C_d), 0.5), angle

    def attenuation(self, img):
        # Create attenuation matrix based on image to apply noise on right colour channel
        return 255

    def drops_dist(self, drop_size):
        # DSD based on "Fast Reactive Control for Illumination Through Rain and Snow"
        #n_drop = 3800*np.power(self.rate,-0.87)*np.exp(-25.5*np.power(self.rate,-0.48))

        # DSD based on A Statistical and Physical Description of Hydrometeor Distributions in Colorado Snowstorms Using a Video Disdrometer
        # Table 1 - Snowfall liquid equivanlent rate of 3.03 mm/h, using the Gamma model
        if self.rate == 3.03:
            N_O = 2.33e4
            mu = 1.23
            slope_lambda = 2.25
            gamma = True
        elif self.rate == 2.70:
            N_O = 1.39e3
            mu = -0.90
            slope_lambda = 1.40
            gamma = True
        elif self.rate == 2.59:
            N_O = 7.18e1
            mu = -0.78
            slope_lambda = 0.35
            gamma = True
        else:
            # From generalised colorado
            Temp_O = 273.15 # Fixed 0C in Kelvins 
            Temp = 271.15 # Needs to be sub zero, 271.5K = -2C

            N_O = 5000 * np.power(self.rate, -1.2)
            slope_lambda = 2.27*np.power(Temp_O - Temp,0.18)
            mu = -0.00499 * slope_lambda*slope_lambda + 0.798* slope_lambda - 0.666

            gamma = True
        if gamma:
            n_drop = N_O * np.power(drop_size, mu) * np.exp(-slope_lambda * drop_size)
        else:
            n_drop = N_O * np.exp(slope_lambda * drop_size)

        return n_drop


class Fog_Param:
    # Default parameters for fog
    def __init__(self, rate = int(10)):
        self.rate = rate                 # Fog Visibility in meters

    def drops_dist(self, drop_size):
        # DSD based on log normal law
        radius = drop_size * 1e6            # Equation uses micrometers

        # Parameters from CE:
        if self.rate == 10:
            N = 6108.486
            sigma = 0.74651233
            ln_rt = 0.484235252
        elif self.rate == 20:
            N = 3559.641
            sigma = 0.777099786
            ln_rt = 0.362382125
        elif self.rate == 30:
            N = 2754.154
            sigma = 0.742155699
            ln_rt = 0.334042672
        elif self.rate == 50:
            N = 3189.711
            sigma = 0.656240841
            ln_rt = 0.118386081
        else:
            print('Please parameterise fog DSD for this rate')
            sys.exit()

        # Equation based on cm-3, need to convert to m-3
        return (N / (radius * sigma * np.power(2 * np.pi, 0.5))) * np.exp(-np.power(np.log(radius) - ln_rt, 2) / (2 * np.power(sigma, 2))) * 1e6
