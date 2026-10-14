# import torch
# import math
# import numpy as np

# lg_e_10 = math.log(10)


# def log10(x):
#     """Convert a new tensor with the base-10 logarithm of the elements of x. """
#     return torch.log(x) / lg_e_10


# class Result(object):
#     def __init__(self):
#         self.irmse = 0
#         self.imae = 0
#         self.mse = 0
#         self.rmse = 0
#         self.mae = 0
#         self.absrel = 0
#         self.squared_rel = 0
#         self.lg10 = 0
#         self.delta1 = 0
#         self.delta2 = 0
#         self.delta3 = 0
#         self.data_time = 0
#         self.gpu_time = 0
#         self.silog = 0  # Scale invariant logarithmic error [log(m)*100]
#         self.photometric = 0

#     def set_to_worst(self):
#         self.irmse = np.inf
#         self.imae = np.inf
#         self.mse = np.inf
#         self.rmse = np.inf
#         self.mae = np.inf
#         self.absrel = np.inf
#         self.squared_rel = np.inf
#         self.lg10 = np.inf
#         self.silog = np.inf
#         self.delta1 = 0
#         self.delta2 = 0
#         self.delta3 = 0
#         self.data_time = 0
#         self.gpu_time = 0

#     def update(self, irmse, imae, mse, rmse, mae, absrel, squared_rel, lg10, \
#             delta1, delta2, delta3, gpu_time, data_time, silog, photometric=0):
#         self.irmse = irmse
#         self.imae = imae
#         self.mse = mse
#         self.rmse = rmse
#         self.mae = mae
#         self.absrel = absrel
#         self.squared_rel = squared_rel
#         self.lg10 = lg10
#         self.delta1 = delta1
#         self.delta2 = delta2
#         self.delta3 = delta3
#         self.data_time = data_time
#         self.gpu_time = gpu_time
#         self.silog = silog
#         self.photometric = photometric

#     def evaluate(self, output, target, photometric=0):
#         valid_mask = target > 0.1  # 过滤掉零像素

#         if valid_mask.sum() == 0:
#             print("[WARN] No valid pixels in this sample, skipping evaluation")
#             self.irmse = float('nan')
#             self.imae = float('nan')
#             self.rmse = float('nan')
#             self.mae = float('nan')
#             self.silog = float('nan')
#             self.absrel = float('nan')
#             self.squared_rel = float('nan')
#             self.lg10 = float('nan')
#             self.delta1 = 0
#             self.delta2 = 0
#             self.delta3 = 0
#             self.photometric = float(photometric)
#             return

#         output_valid = output[valid_mask]
#         target_valid = target[valid_mask]

#         abs_diff = (output_valid - target_valid).abs()

#         self.mse = float((abs_diff**2).mean())
#         self.rmse = float(torch.sqrt(torch.tensor(self.mse)))
#         self.mae = float(abs_diff.mean())
#         self.lg10 = float((torch.log10(output_valid) - torch.log10(target_valid)).abs().mean())
#         self.absrel = float((abs_diff / target_valid).mean())
#         self.squared_rel = float(((abs_diff / target_valid)**2).mean())

#         maxRatio = torch.max(output_valid / target_valid, target_valid / output_valid)
#         self.delta1 = float((maxRatio < 1.25).float().mean())
#         self.delta2 = float((maxRatio < 1.25**2).float().mean())
#         self.delta3 = float((maxRatio < 1.25**3).float().mean())

#         # silog 使用 meters
#         err_log = torch.log(target_valid) - torch.log(output_valid)
#         normalized_squared_log = (err_log**2).mean()
#         log_mean = err_log.mean()
#         self.silog = math.sqrt(normalized_squared_log - log_mean * log_mean) * 100

#         # inverse depth metrics
#         inv_mask = output_valid > 0  # 避免除零
#         if inv_mask.sum() == 0:
#             self.irmse = float('nan')
#             self.imae = float('nan')
#         else:
#             inv_output = (1e-3 * output_valid[inv_mask])**(-1)
#             inv_target = (1e-3 * target_valid[inv_mask])**(-1)
#             abs_inv_diff = (inv_output - inv_target).abs()
#             self.irmse = float(torch.sqrt((abs_inv_diff**2).mean()))
#             self.imae = float(abs_inv_diff.mean())

#         self.data_time = 0
#         self.gpu_time = 0
#         self.photometric = float(photometric)



# class AverageMeter(object):
#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.count = 0.0
#         self.sum_irmse = 0
#         self.sum_imae = 0
#         self.sum_mse = 0
#         self.sum_rmse = 0
#         self.sum_mae = 0
#         self.sum_absrel = 0
#         self.sum_squared_rel = 0
#         self.sum_lg10 = 0
#         self.sum_delta1 = 0
#         self.sum_delta2 = 0
#         self.sum_delta3 = 0
#         self.sum_data_time = 0
#         self.sum_gpu_time = 0
#         self.sum_photometric = 0
#         self.sum_silog = 0

#     def update(self, result, gpu_time, data_time, n=1):
#         self.count += n
#         self.sum_irmse += n * result.irmse
#         self.sum_imae += n * result.imae
#         self.sum_mse += n * result.mse
#         self.sum_rmse += n * result.rmse
#         self.sum_mae += n * result.mae
#         self.sum_absrel += n * result.absrel
#         self.sum_squared_rel += n * result.squared_rel
#         self.sum_lg10 += n * result.lg10
#         self.sum_delta1 += n * result.delta1
#         self.sum_delta2 += n * result.delta2
#         self.sum_delta3 += n * result.delta3
#         self.sum_data_time += n * data_time
#         self.sum_gpu_time += n * gpu_time
#         self.sum_silog += n * result.silog
#         self.sum_photometric += n * result.photometric

#     def average(self):
#         avg = Result()
#         if self.count > 0:
#             avg.update(
#                 self.sum_irmse / self.count, self.sum_imae / self.count,
#                 self.sum_mse / self.count, self.sum_rmse / self.count,
#                 self.sum_mae / self.count, self.sum_absrel / self.count,
#                 self.sum_squared_rel / self.count, self.sum_lg10 / self.count,
#                 self.sum_delta1 / self.count, self.sum_delta2 / self.count,
#                 self.sum_delta3 / self.count, self.sum_gpu_time / self.count,
#                 self.sum_data_time / self.count, self.sum_silog / self.count,
#                 self.sum_photometric / self.count)
#         return avg

import torch
import math
import numpy as np

lg_e_10 = math.log(10)

def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x."""
    return torch.log(x) / lg_e_10


class Result(object):
    def __init__(self):
        self.irmse = 0
        self.imae = 0
        self.mse = 0
        self.rmse = 0
        self.mae = 0
        self.absrel = 0
        self.squared_rel = 0
        self.lg10 = 0
        self.delta1 = 0
        self.delta2 = 0
        self.delta3 = 0
        self.data_time = 0
        self.gpu_time = 0
        self.silog = 0
        self.photometric = 0

    def set_to_worst(self):
        self.irmse = np.inf
        self.imae = np.inf
        self.mse = np.inf
        self.rmse = np.inf
        self.mae = np.inf
        self.absrel = np.inf
        self.squared_rel = np.inf
        self.lg10 = np.inf
        self.silog = np.inf
        self.delta1 = 0
        self.delta2 = 0
        self.delta3 = 0
        self.data_time = 0
        self.gpu_time = 0

    def update(self, irmse, imae, mse, rmse, mae, absrel, squared_rel, lg10,
               delta1, delta2, delta3, gpu_time, data_time, silog, photometric=0):
        self.irmse = irmse
        self.imae = imae
        self.mse = mse
        self.rmse = rmse
        self.mae = mae
        self.absrel = absrel
        self.squared_rel = squared_rel
        self.lg10 = lg10
        self.delta1 = delta1
        self.delta2 = delta2
        self.delta3 = delta3
        self.data_time = data_time
        self.gpu_time = gpu_time
        self.silog = silog
        self.photometric = photometric

    def evaluate(self, output, target, photometric=0):
        epsilon = 1e-6  # 避免 log(0) 或倒数除零

        valid_mask = target > 0.1
        if valid_mask.sum() == 0:
            # 没有有效像素，指标保持 nan
            self.irmse = float('nan')
            self.imae = float('nan')
            self.rmse = float('nan')
            self.mae = float('nan')
            self.silog = float('nan')
            self.absrel = float('nan')
            self.squared_rel = float('nan')
            self.lg10 = float('nan')
            self.delta1 = 0
            self.delta2 = 0
            self.delta3 = 0
            self.photometric = float(photometric)
            return

        output_valid = output[valid_mask]
        target_valid = target[valid_mask]

        # 基本误差
        abs_diff = (output_valid - target_valid).abs()
        self.mse = float((abs_diff**2).mean())
        self.rmse = float(torch.sqrt(torch.tensor(self.mse)))
        self.mae = float(abs_diff.mean())
        self.lg10 = float((log10(output_valid + epsilon) - log10(target_valid + epsilon)).abs().mean())
        self.absrel = float((abs_diff / target_valid).mean())
        self.squared_rel = float(((abs_diff / target_valid)**2).mean())

        # delta指标
        maxRatio = torch.max(output_valid / target_valid, target_valid / output_valid)
        self.delta1 = float((maxRatio < 1.25).float().mean())
        self.delta2 = float((maxRatio < 1.25**2).float().mean())
        self.delta3 = float((maxRatio < 1.25**3).float().mean())

        # silog
        err_log = torch.log(target_valid + epsilon) - torch.log(output_valid + epsilon)
        normalized_squared_log = (err_log**2).mean()
        log_mean = err_log.mean()
        self.silog = math.sqrt(normalized_squared_log - log_mean * log_mean) * 100

        # 倒数深度指标
        inv_mask = output_valid > 0
        if inv_mask.sum() == 0:
            self.irmse = float('nan')
            self.imae = float('nan')
        else:
            inv_output = 1.0 / (output_valid[inv_mask] * 1e-3)
            inv_target = 1.0 / (target_valid[inv_mask] * 1e-3)
            abs_inv_diff = (inv_output - inv_target).abs()
            self.irmse = float(torch.sqrt((abs_inv_diff**2).mean()))
            self.imae = float(abs_inv_diff.mean())

        self.data_time = 0
        self.gpu_time = 0
        self.photometric = float(photometric)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0
        self.sum_irmse = 0
        self.sum_imae = 0
        self.sum_mse = 0
        self.sum_rmse = 0
        self.sum_mae = 0
        self.sum_absrel = 0
        self.sum_squared_rel = 0
        self.sum_lg10 = 0
        self.sum_delta1 = 0
        self.sum_delta2 = 0
        self.sum_delta3 = 0
        self.sum_data_time = 0
        self.sum_gpu_time = 0
        self.sum_photometric = 0
        self.sum_silog = 0

    def update(self, result, gpu_time, data_time, n=1):
        self.count += n
        self.sum_irmse += n * result.irmse
        self.sum_imae += n * result.imae
        self.sum_mse += n * result.mse
        self.sum_rmse += n * result.rmse
        self.sum_mae += n * result.mae
        self.sum_absrel += n * result.absrel
        self.sum_squared_rel += n * result.squared_rel
        self.sum_lg10 += n * result.lg10
        self.sum_delta1 += n * result.delta1
        self.sum_delta2 += n * result.delta2
        self.sum_delta3 += n * result.delta3
        self.sum_data_time += n * data_time
        self.sum_gpu_time += n * gpu_time
        self.sum_silog += n * result.silog
        self.sum_photometric += n * result.photometric

    def average(self):
        avg = Result()
        if self.count > 0:
            avg.update(
                self.sum_irmse / self.count, self.sum_imae / self.count,
                self.sum_mse / self.count, self.sum_rmse / self.count,
                self.sum_mae / self.count, self.sum_absrel / self.count,
                self.sum_squared_rel / self.count, self.sum_lg10 / self.count,
                self.sum_delta1 / self.count, self.sum_delta2 / self.count,
                self.sum_delta3 / self.count, self.sum_gpu_time / self.count,
                self.sum_data_time / self.count, self.sum_silog / self.count,
                self.sum_photometric / self.count)
        return avg
