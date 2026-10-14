import argparse
import logging
import os
import sys
import warnings
import time
from glob import glob

import diffusers
import numpy as np
import torch
from diffusers import DDIMScheduler, MarigoldDepthPipeline
from PIL import Image
from tqdm import tqdm

warnings.simplefilter(action="ignore", category=FutureWarning)
diffusers.utils.logging.disable_progress_bar()


class MarigoldDepthCompletionPipeline(MarigoldDepthPipeline):
    """
    Pipeline for Marigold Depth Completion.
    Extends the MarigoldDepthPipeline to include depth completion functionality.
    """

    def __call__(
        self,
        image: Image.Image,
        sparse_depth: np.ndarray,
        num_inference_steps: int = 50,
        ensemble_size: int = 1,
        processing_resolution: int = 768,
        seed: int = 2024,
    ) -> np.ndarray:
        """
        Args:
            image (PIL.Image.Image): Input RGB image [H,W,3].
            sparse_depth (np.ndarray): Sparse depth map [H,W] in METERS.
            num_inference_steps (int): Diffusion denoising steps. Default 50.
            ensemble_size (int): #samples for median ensemble. Default 1.
            processing_resolution (int): Working resolution for the model. Default 768.
            seed (int): Random seed for generator.

        Returns:
            np.ndarray: Dense metric depth [H,W] in meters.
        """

        device = self._execution_device
        generator = torch.Generator(device=device).manual_seed(seed)

        # Sanity checks
        if not isinstance(num_inference_steps, int) or num_inference_steps < 1:
            raise ValueError("Invalid num_inference_steps")
        if type(sparse_depth) is not np.ndarray or sparse_depth.ndim != 2:
            raise ValueError("Sparse depth should be a 2D numpy ndarray with zeros at missing positions")
        if ensemble_size < 1:
            raise ValueError("Ensemble size must be at least 1")

        # Prepare empty text conditioning once
        with torch.no_grad():
            if self.empty_text_embedding is None:
                text_inputs = self.tokenizer(
                    "",
                    padding="do_not_pad",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids.to(device)
                self.empty_text_embedding = self.text_encoder(text_input_ids)[0]  # [1,2,1024]

        # Preprocess RGB
        image, padding, original_resolution = self.image_processor.preprocess(
            image,
            processing_resolution=processing_resolution,
            device=device,
            dtype=self.dtype,
        )  # [N,3,PPH,PPW]

        # Check sparse depth dimensions
        if sparse_depth.shape != original_resolution:
            raise ValueError(
                f"Sparse depth dimensions ({sparse_depth.shape}) must match that of the image ({image.shape[-2:]})"
            )

        # Encode the image into latents
        with torch.no_grad():
            image_latent, pred_latent = self.prepare_latents(
                image, None, generator, ensemble_size, 1
            )  # [N*E,4,h,w], [N*E,4,h,w]
        del image

        # Prepare sparse depth & mask
        sparse_depth = torch.from_numpy(sparse_depth)[None, None].float().to(device)
        sparse_mask = sparse_depth > 0
        logging.debug(f"Using {sparse_mask.int().sum().item()} guidance points")

        def affine_to_metric(depth: torch.Tensor) -> torch.Tensor:
            # Convert affine-invariant depth to metric depth using learnable scale/shift (Eq. 2 in paper)
            return (scale**2) * sparse_range * depth + (shift**2) * sparse_lower

        def latent_to_metric(latent: torch.Tensor) -> torch.Tensor:
            # latent -> affine-invariant depth -> metric depth -> unpad -> resize to original res
            affine_invariant_prediction = self.decode_prediction(latent)  # [E,1,PPH,PPW]
            prediction = affine_to_metric(affine_invariant_prediction)
            prediction = self.image_processor.unpad_image(prediction, padding)  # [E,1,PH,PW]
            prediction = self.image_processor.resize_antialias(
                prediction, original_resolution, "bilinear", is_aa=False
            )  # [1,1,H,W]
            return prediction

        def loss_l1l2(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            out_l1 = torch.nn.functional.l1_loss(input, target)
            out_l2 = torch.nn.functional.mse_loss(input, target)
            return out_l1 + out_l2

        self.scheduler.set_timesteps(num_inference_steps, device=device)

        ensemble_predictions = []

        for ensemble_idx in self.progress_bar(
            range(ensemble_size),
            desc="Processing ensemble members...",
            leave=False,
        ):
            current_image_latent = image_latent[ensemble_idx : ensemble_idx + 1]  # [1,4,h,w]
            current_pred_latent = pred_latent[ensemble_idx : ensemble_idx + 1]  # [1,4,h,w]

            # Per-image trainable parameters at test time
            scale = torch.nn.Parameter(torch.ones(1, device=device))
            shift = torch.nn.Parameter(torch.ones(1, device=device))
            current_pred_latent = torch.nn.Parameter(current_pred_latent)

            # Sparse stats (for metric alignment)
            sparse_range = (
                sparse_depth[sparse_mask].max() - sparse_depth[sparse_mask].min()
            ).item()  # cmax - cmin
            sparse_lower = (sparse_depth[sparse_mask].min()).item()  # cmin

            # Optimizer for these per-image params
            optimizer = torch.optim.Adam(
                [
                    {"params": [scale, shift], "lr": 0.005},
                    {"params": [current_pred_latent], "lr": 0.05},
                ]
            )

            # Denoising + guidance loop
            for _, t in enumerate(
                self.progress_bar(
                    self.scheduler.timesteps,
                    desc=f"Marigold-DC steps ({str(device)})...",
                    leave=False,
                )
            ):
                optimizer.zero_grad()

                # Concatenate latents and forward U-Net
                batch_latent = torch.cat(
                    [current_image_latent, current_pred_latent], dim=1
                )  # [1,8,h,w]
                noise = self.unet(
                    batch_latent,
                    t,
                    encoder_hidden_states=self.empty_text_embedding,
                    return_dict=False,
                )[0]  # [1,4,h,w]

                # For gradient scaling
                with torch.no_grad():
                    alpha_prod_t = self.scheduler.alphas_cumprod[t]
                    beta_prod_t = 1 - alpha_prod_t
                    pred_epsilon = (alpha_prod_t**0.5) * noise + (beta_prod_t**0.5) * current_pred_latent

                step_output = self.scheduler.step(
                    noise, t, current_pred_latent, generator=torch.Generator(device=device)
                )

                pred_original_sample = step_output.pred_original_sample  # Tweedie clean latent

                # Decode to metric depth, compute guidance loss, backprop
                current_metric_estimate = latent_to_metric(pred_original_sample)
                loss = loss_l1l2(
                    current_metric_estimate[sparse_mask],
                    sparse_depth[sparse_mask],
                )
                loss.backward()

                # Gradient rescale
                with torch.no_grad():
                    pred_epsilon_norm = torch.linalg.norm(pred_epsilon).item()
                    depth_latent_grad_norm = torch.linalg.norm(
                        current_pred_latent.grad
                    ).item()
                    scaling_factor = pred_epsilon_norm / max(
                        depth_latent_grad_norm, 1e-8
                    )
                    current_pred_latent.grad *= scaling_factor

                # Optim update
                optimizer.step()

                # DDIM step for current_pred_latent
                with torch.no_grad():
                    current_pred_latent.data = self.scheduler.step(
                        noise,
                        t,
                        current_pred_latent,
                        generator=torch.Generator(device=device),
                    ).prev_sample

                del (
                    pred_original_sample,
                    current_metric_estimate,
                    step_output,
                    pred_epsilon,
                    noise,
                )
                torch.cuda.empty_cache()

            # Decode final dense depth for this ensemble member
            with torch.no_grad():
                current_prediction = latent_to_metric(current_pred_latent.detach())
                ensemble_predictions.append(current_prediction)

        del image_latent

        # Ensemble (median across ensemble_size)
        if ensemble_size > 1:
            ensemble_tensor = torch.cat(ensemble_predictions, dim=0)  # [E,1,H,W]
            prediction = ensemble_tensor.median(dim=0, keepdim=True).values  # [1,1,H,W]
        else:
            prediction = ensemble_predictions[0]

        # Convert to numpy [H,W]
        prediction = self.image_processor.pt_to_numpy(prediction)  # [N,H,W,1]
        self.maybe_free_model_hooks()
        return prediction.squeeze()


def count_model_params(pipe: MarigoldDepthCompletionPipeline):
    """
    Estimate parameter counts by summing params of known nn.Modules inside the pipeline.
    Returns:
        total_params: all params in {unet, vae, text_encoder} [int]
        trainable_params: subset with requires_grad=True [int]
    Note:
        During Marigold-DC test-time, scale/shift/current_pred_latent are created per image,
        so they're NOT persistent in this pipeline and not counted here.
    """
    modules_to_check = []
    # some of these may not exist depending on tiny_vae or checkpoint
    if hasattr(pipe, "unet"):
        modules_to_check.append(pipe.unet)
    if hasattr(pipe, "vae"):
        modules_to_check.append(pipe.vae)
    if hasattr(pipe, "text_encoder"):
        modules_to_check.append(pipe.text_encoder)

    total_params = 0
    trainable_params = 0

    for m in modules_to_check:
        # m should be a torch.nn.Module
        for p in m.parameters():
            n = p.numel()
            total_params += n
            if p.requires_grad:
                trainable_params += n

    return total_params, trainable_params

def save_outputs(pred_depth_m: np.ndarray, out_path_npy: str, pipe, rgb_stem: str,
                 depth16_dir: str):
    """
    Save:
    - pred_depth_m (float meters) -> npy
    - visualization jpg
    - 16-bit PNG in depth16_dir (same basename as rgb, not *_pred)
    """
    os.makedirs(os.path.dirname(out_path_npy), exist_ok=True)

    # 1) save dense depth .npy (meters)
    np.save(out_path_npy, pred_depth_m)

    # 2) save color visualization .jpg
    vis = pipe.image_processor.visualize_depth(
        pred_depth_m,
        val_min=pred_depth_m.min(),
        val_max=pred_depth_m.max(),
    )[0]
    vis.save(os.path.splitext(out_path_npy)[0] + "_vis.jpg")

    # 3) save 16-bit PNG (depth * 256 -> uint16) to depth16_dir
    if depth16_dir is not None:
        os.makedirs(depth16_dir, exist_ok=True)

        depth_uint16 = np.clip(pred_depth_m * 256.0, 0, 65535).astype(np.uint16)
        depth16_path = os.path.join(depth16_dir, rgb_stem + ".png")
        Image.fromarray(depth_uint16).save(depth16_path, mode="I;16")  # enforce 16bit

    return


def guess_sparse_from_rgb(rgb_filename: str) -> str:
    """
    Given RGB filename like:
      2011_09_29_drive_0026_sync_image_0000000008_image_02.png
    The correct sparse filename should be:
      2011_09_29_drive_0026_sync_velodyne_raw_0000000008_image_02.npy

    Rule:
      only replace the FIRST occurrence of '_image_' with '_velodyne_raw_'
      and keep the trailing '_image_02' part unchanged.
    """
    base = os.path.basename(rgb_filename)
    stem, _ = os.path.splitext(base)

    # 只替换第一次出现的 "_image_" → "_velodyne_raw_"
    stem_sparse = stem.replace("_image_", "_velodyne_raw_", 1)

    return stem_sparse + ".npy"

def run_single_inference(
    pipe,
    rgb_path,
    sparse_path,
    out_path_npy,
    num_inference_steps,
    ensemble_size,
    processing_resolution,
    depth16_dir=None,
):
    """
    Run Marigold-DC on a single (rgb, sparse_depth); time it; save outputs.
    Returns elapsed time in seconds.
    """
    rgb_img = Image.open(rgb_path).convert("RGB")
    sparse_depth = np.load(sparse_path)  # float meters, HxW, 0 = invalid

    start_t = time.time()

    pred = pipe(
        image=rgb_img,
        sparse_depth=sparse_depth,
        num_inference_steps=num_inference_steps,
        ensemble_size=ensemble_size,
        processing_resolution=processing_resolution,
    )

    elapsed = time.time() - start_t

    # stem for naming
    rgb_stem = os.path.splitext(os.path.basename(rgb_path))[0]

    save_outputs(
        pred_depth_m=pred,
        out_path_npy=out_path_npy,
        pipe=pipe,
        rgb_stem=rgb_stem,
        depth16_dir=depth16_dir,
    )

    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Marigold-DC Batch Runner with timing & 16bit export")

    DEPTH_CHECKPOINT = "prs-eth/marigold-depth-v1-0"

    # mode 1: single
    parser.add_argument("--in-image", type=str, default=None, help="Single RGB image path")
    parser.add_argument("--in-depth", type=str, default=None, help="Single sparse depth .npy path (meters)")
    parser.add_argument("--out-depth", type=str, default=None, help="Single output dense depth .npy path")

    # mode 2: batch
    parser.add_argument("--image-dir", type=str, default=None, help="Folder of RGB .png images")
    parser.add_argument("--sparse-dir", type=str, default=None, help="Folder of sparse depth .npy files")
    parser.add_argument("--out-dir", type=str, default=None, help="Folder to save per-frame *_pred.npy and *_pred_vis.jpg")

    # NEW: where to save 16-bit PNG depth maps (pred * 256 as uint16), same basename as RGB
    parser.add_argument("--depth16-dir", type=str, default=None,
                        help="Folder to save 16-bit depth PNGs (prediction[m]*256 cast to uint16).")

    parser.add_argument("--num_inference_steps", type=int, default=50, help="Denoising steps")
    parser.add_argument("--ensemble_size", type=int, default=1, help="Median-ensemble size")
    parser.add_argument("--processing_resolution", type=int, default=768, help="Working resolution")
    parser.add_argument("--checkpoint", type=str, default=DEPTH_CHECKPOINT, help="HuggingFace checkpoint")
    parser.add_argument("--use_full_precision", action="store_true", help="Use float32 instead of bfloat16")
    parser.add_argument("--use_tiny_vae", action="store_true", help="Use Tiny VAE for faster inference (lower quality)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # choose device + adjust if no CUDA
    num_inference_steps = args.num_inference_steps
    ensemble_size = args.ensemble_size
    processing_resolution = args.processing_resolution

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        # Fallback heuristics for slow backends
        processing_resolution_non_cuda = 512
        num_inference_steps_non_cuda = 10
        ensemble_size_non_cuda = 1

        if processing_resolution > processing_resolution_non_cuda:
            logging.warning(f"CUDA not found: Reducing processing_resolution to {processing_resolution_non_cuda}")
            processing_resolution = processing_resolution_non_cuda
        if num_inference_steps > num_inference_steps_non_cuda:
            logging.warning(f"CUDA not found: Reducing num_inference_steps to {num_inference_steps_non_cuda}")
            num_inference_steps = num_inference_steps_non_cuda
        if ensemble_size > ensemble_size_non_cuda:
            logging.warning(f"CUDA not found: Reducing ensemble_size to {ensemble_size_non_cuda}")
            ensemble_size = ensemble_size_non_cuda

    torch_dtype = torch.float32 if args.use_full_precision else torch.bfloat16
    logging.info(f"Using {torch_dtype} precision. Using full precision (float32) is discouraged unless needed.")

    # build pipeline
    pipe = MarigoldDepthCompletionPipeline.from_pretrained(
        args.checkpoint,
        prediction_type="depth",
    ).to(device, dtype=torch_dtype)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    if not torch.cuda.is_available():
        logging.warning("CUDA not found: Consider using tiny VAE for faster processing")

    if args.use_tiny_vae:
        logging.info("Using a lightweight Tiny VAE")
        del pipe.vae
        pipe.vae = diffusers.AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device, dtype=torch_dtype)

    # report parameter counts
    total_params, trainable_params = count_model_params(pipe)
    logging.info(f"Model total parameters: {total_params/1e6:.2f} M")
    logging.info(
        f"Parameters with requires_grad=True in pipeline: {trainable_params/1e6:.2f} M "
        "(global trainable params; test-time also optimizes per-image latent + scale/shift)"
    )

    # MODE 1: single image mode
    if args.in_image and args.in_depth and args.out_depth:
        logging.info("Running in single-image mode")

        elapsed = run_single_inference(
            pipe=pipe,
            rgb_path=args.in_image,
            sparse_path=args.in_depth,
            out_path_npy=args.out_depth,
            num_inference_steps=num_inference_steps,
            ensemble_size=ensemble_size,
            processing_resolution=processing_resolution,
            depth16_dir=args.depth16_dir,
        )

        logging.info(f"Single image processing time: {elapsed:.4f} seconds")
        logging.info("Done.")
        return

    # MODE 2: batch mode
    if args.image_dir and args.sparse_dir and args.out_dir:
        logging.info("Running in batch mode")
        os.makedirs(args.out_dir, exist_ok=True)
        if args.depth16_dir is not None:
            os.makedirs(args.depth16_dir, exist_ok=True)

        rgb_list = sorted(glob(os.path.join(args.image_dir, "*.png")))

        per_image_times = []
        for rgb_path in tqdm(rgb_list, desc="Batch inference"):
            # derive sparse depth filename using naming rule
            sparse_name = guess_sparse_from_rgb(os.path.basename(rgb_path))
            sparse_path = os.path.join(args.sparse_dir, sparse_name)

            if not os.path.exists(sparse_path):
                logging.warning(f"[SKIP] no sparse depth for {rgb_path} -> {sparse_path}")
                continue

            stem = os.path.splitext(os.path.basename(rgb_path))[0]
            out_path_npy = os.path.join(args.out_dir, stem + "_pred.npy")

            elapsed = run_single_inference(
                pipe=pipe,
                rgb_path=rgb_path,
                sparse_path=sparse_path,
                out_path_npy=out_path_npy,
                num_inference_steps=num_inference_steps,
                ensemble_size=ensemble_size,
                processing_resolution=processing_resolution,
                depth16_dir=args.depth16_dir,
            )

            per_image_times.append(elapsed)

        if len(per_image_times) > 0:
            avg_time = float(np.mean(per_image_times))
        else:
            avg_time = float("nan")

        logging.info(f"Processed {len(per_image_times)} images.")
        logging.info(f"Average processing time per image: {avg_time:.4f} seconds")
        logging.info("Batch mode complete.")
        return

    # Neither mode matched
    logging.error(
        "You must either provide --in-image/--in-depth/--out-depth (single mode) "
        "OR --image-dir/--sparse-dir/--out-dir (batch mode)."
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
