"""Fixed drop-in replacement for
``sglang.srt.hardware_backend.npu.modules.qwen_vl_processor``.

This module is injected into ``sys.modules`` under the original sglang path by
``patches/qwen_vl_processor.py`` so that ``base_processor``'s lazy
``from ...qwen_vl_processor import npu_apply_qwen_image_preprocess_patch`` picks
up this fixed version instead of the original module (whose top-level import of
``group_images_by_shape``/``reorder_images`` from
``transformers.image_processing_utils_fast`` fails on the target transformers
version).

Differences from the original sglang module:
  1. Import ``group_images_by_shape``/``reorder_images`` from
     ``transformers.image_transforms``.
  2. The image/video preprocess wrappers use ``*args``/``**kwargs`` so they are
     compatible across transformers versions (handles NPU's 8-dim tensor limit).
  3. ``npu_apply_qwen_image_preprocess_patch`` guards each ``apply_module_patch``
     with ``try/except ModuleNotFoundError`` and also patches the ``*Fast``
     processor variants.
"""

import torch
import torchvision.transforms.v2.functional as tvF
from transformers.image_processing_utils import BatchFeature
from transformers.image_transforms import group_images_by_shape, reorder_images
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    SizeDict,
    get_image_size,
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
    smart_resize as smart_resize_video,
)
from transformers.utils import TensorType
from transformers.video_utils import group_videos_by_shape, reorder_videos

from sglang.srt.utils import apply_module_patch


def transform_patches_to_flatten(
    patches: torch.Tensor,
    batch_size: int,
    grid_t: int,
    temporal_patch_size: int,
    channel: int,
    grid_h: int,
    grid_w: int,
    patch_size: int,
    merge_size: int,
) -> torch.Tensor:
    patches = patches.view(
        batch_size * grid_t,
        temporal_patch_size * channel,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.permute(0, 1, 2, 5, 3, 6, 4, 7)
    patches = patches.reshape(
        batch_size,
        grid_t,
        temporal_patch_size,
        channel,
        grid_h * grid_w,
        patch_size,
        patch_size,
    )
    patches = patches.permute(0, 1, 4, 3, 2, 5, 6)
    flatten_patches = patches.reshape(
        batch_size,
        grid_t * grid_h * grid_w,
        -1,
    )
    return flatten_patches


# Func refers to transformers.models.qwen2_vl.image_processing_qwen2_vl.py
# Qwen2VLImageProcessor._preprocess
def npu_wrapper_preprocess(func):
    """
    Universal wrapper for Qwen2VL/Qwen3VL image preprocess that handles NPU's 8-dim tensor limit.
    Uses *args/**kwargs to be compatible with different transformers versions.
    """

    def _preprocess(self, *args, **kwargs):
        # Extract parameters from kwargs with defaults from self attributes
        images = kwargs.get("images", args[0] if len(args) > 0 else None)
        do_resize = kwargs.get("do_resize", getattr(self, "do_resize", True))
        size = kwargs.get("size", getattr(self, "size", SizeDict(shortest_edge=3136, longest_edge=1003520)))
        resample = kwargs.get("resample", getattr(self, "resample", PILImageResampling.BICUBIC))
        do_rescale = kwargs.get("do_rescale", getattr(self, "do_rescale", True))
        rescale_factor = kwargs.get("rescale_factor", getattr(self, "rescale_factor", 1 / 255.0))
        do_normalize = kwargs.get("do_normalize", getattr(self, "do_normalize", True))
        image_mean = kwargs.get("image_mean", getattr(self, "image_mean", None))
        image_std = kwargs.get("image_std", getattr(self, "image_std", None))
        patch_size = kwargs.get("patch_size", getattr(self, "patch_size", 14))
        temporal_patch_size = kwargs.get("temporal_patch_size", getattr(self, "temporal_patch_size", 2))
        merge_size = kwargs.get("merge_size", getattr(self, "merge_size", 2))
        disable_grouping = kwargs.get("disable_grouping", None)
        return_tensors = kwargs.get("return_tensors", "pt")
        # Remove already extracted parameters from kwargs to avoid duplication
        for key in ["images", "do_resize", "size", "resample", "do_rescale",
                    "rescale_factor", "do_normalize", "image_mean", "image_std",
                    "patch_size", "temporal_patch_size", "merge_size",
                    "disable_grouping", "return_tensors"]:
            kwargs.pop(key, None)

        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(
            images, disable_grouping=disable_grouping
        )
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            height, width = stacked_images.shape[-2:]
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=size.shortest_edge,
                    max_pixels=size.longest_edge,
                )
                stacked_images = self.resize(
                    image=stacked_images,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
                )
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(
            resized_images, disable_grouping=disable_grouping
        )
        processed_images_grouped = {}
        processed_grids = {}
        for shape, stacked_images in grouped_images.items():
            resized_height, resized_width = stacked_images.shape[-2:]
            # Fused rescale and normalize
            patches = self.rescale_and_normalize(
                stacked_images,
                do_rescale,
                rescale_factor,
                do_normalize,
                image_mean,
                image_std,
            )
            if patches.ndim == 4:
                # add a temporal dimension if we have images
                patches = patches.unsqueeze(1)
            if patches.shape[1] % temporal_patch_size != 0:
                repeats = patches[:, -1:].repeat(1, temporal_patch_size - 1, 1, 1, 1)
                patches = torch.cat([patches, repeats], dim=1)
            batch_size, grid_t, channel = patches.shape[:3]
            grid_t = grid_t // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            flatten_patches = transform_patches_to_flatten(
                patches,
                batch_size,
                grid_t,
                temporal_patch_size,
                channel,
                grid_h,
                grid_w,
                patch_size,
                merge_size,
            )

            processed_images_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_images = reorder_images(
            processed_images_grouped, grouped_images_index
        )
        processed_grids = reorder_images(processed_grids, grouped_images_index)
        pixel_values = torch.cat(processed_images, dim=0)
        image_grid_thw = torch.tensor(processed_grids)

        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
            tensor_type=return_tensors,
        )

    return _preprocess


def npu_wrapper_video_preprocess(func):
    """
    Universal wrapper for Qwen3VL video preprocess that handles NPU's 8-dim tensor limit.
    Uses *args/**kwargs to be compatible with different transformers versions.
    """

    def _preprocess(self, *args, **kwargs):
        # Extract parameters from kwargs with defaults from self attributes
        videos = kwargs.get("videos", args[0] if len(args) > 0 else None)
        do_convert_rgb = kwargs.get("do_convert_rgb", True)
        do_resize = kwargs.get("do_resize", getattr(self, "do_resize", True))
        size = kwargs.get("size", getattr(self, "size", SizeDict(shortest_edge=3136, longest_edge=1003520)))
        resample = kwargs.get("resample", getattr(self, "resample", PILImageResampling.BICUBIC))
        do_rescale = kwargs.get("do_rescale", getattr(self, "do_rescale", True))
        rescale_factor = kwargs.get("rescale_factor", getattr(self, "rescale_factor", 1 / 255.0))
        do_normalize = kwargs.get("do_normalize", getattr(self, "do_normalize", True))
        image_mean = kwargs.get("image_mean", getattr(self, "image_mean", None))
        image_std = kwargs.get("image_std", getattr(self, "image_std", None))
        patch_size = kwargs.get("patch_size", getattr(self, "patch_size", 14))
        temporal_patch_size = kwargs.get("temporal_patch_size", getattr(self, "temporal_patch_size", 2))
        merge_size = kwargs.get("merge_size", getattr(self, "merge_size", 2))
        return_tensors = kwargs.get("return_tensors", "pt")

        # Remove already extracted parameters from kwargs
        for key in ["videos", "do_convert_rgb", "do_resize", "size", "resample",
                    "do_rescale", "rescale_factor", "do_normalize", "image_mean",
                    "image_std", "patch_size", "temporal_patch_size", "merge_size",
                    "return_tensors"]:
            kwargs.pop(key, None)
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}

        for shape, stacked_videos in grouped_videos.items():
            B, T, C, H, W = stacked_videos.shape
            num_frames, height, width = T, H, W
            if do_resize:
                resized_height, resized_width = smart_resize_video(
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    temporal_factor=temporal_patch_size,
                    factor=patch_size * merge_size,
                    min_pixels=size.shortest_edge,
                    max_pixels=size.longest_edge,
                )
                stacked_videos = stacked_videos.view(B * T, C, H, W)
                stacked_videos = self.resize(
                    stacked_videos,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
                )
                stacked_videos = stacked_videos.view(
                    B, T, C, resized_height, resized_width
                )
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

        # Group videos by size for further processing
        # Needed in case do_resize is False, or resize returns videos with different sizes
        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        processed_grids = {}
        for shape, stacked_videos in grouped_videos.items():
            resized_height, resized_width = get_image_size(
                stacked_videos[0], channel_dim=ChannelDimension.FIRST
            )

            # Fused rescale and normalize
            stacked_videos = self.rescale_and_normalize(
                stacked_videos,
                do_rescale,
                rescale_factor,
                do_normalize,
                image_mean,
                image_std,
            )
            patches = stacked_videos

            # Check that videos have `num_frames` divisible by `temporal_patch_size`
            T = patches.shape[1]
            if pad := -T % temporal_patch_size:
                repeats = patches[:, -1:].expand(-1, pad, -1, -1, -1)
                patches = torch.cat((patches, repeats), dim=1)
            batch_size, grid_t, channel = patches.shape[:3]
            grid_t = grid_t // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            flatten_patches = transform_patches_to_flatten(
                patches,
                batch_size,
                grid_t,
                temporal_patch_size,
                channel,
                grid_h,
                grid_w,
                patch_size,
                merge_size,
            )

            processed_videos_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_videos = reorder_videos(
            processed_videos_grouped, grouped_videos_index
        )
        processed_grids = reorder_videos(processed_grids, grouped_videos_index)
        pixel_values_videos = torch.cat(processed_videos, dim=0)
        video_grid_thw = torch.tensor(processed_grids)
        data = {
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        }

        return BatchFeature(data=data, tensor_type=return_tensors)

    return _preprocess


_npu_preprocess_patched = False


def npu_apply_qwen_image_preprocess_patch():
    global _npu_preprocess_patched
    if _npu_preprocess_patched:
        return

    # Apply patches with try-except to handle modules that may not exist
    try:
        apply_module_patch(
            "transformers.models.qwen2_vl.image_processing_qwen2_vl.Qwen2VLImageProcessor",
            "_preprocess",
            [npu_wrapper_preprocess],
        )
    except ModuleNotFoundError:
        pass

    try:
        apply_module_patch(
            "transformers.models.qwen2_vl.image_processing_qwen2_vl_fast.Qwen2VLImageProcessorFast",
            "_preprocess",
            [npu_wrapper_preprocess],
        )
    except ModuleNotFoundError:
        pass

    try:
        apply_module_patch(
            "transformers.models.qwen3_vl.video_processing_qwen3_vl.Qwen3VLVideoProcessor",
            "_preprocess",
            [npu_wrapper_video_preprocess],
        )
    except ModuleNotFoundError:
        pass

    try:
        apply_module_patch(
            "transformers.models.qwen3_vl.video_processing_qwen3_vl_fast.Qwen3VLVideoProcessorFast",
            "_preprocess",
            [npu_wrapper_video_preprocess],
        )
    except ModuleNotFoundError:
        pass

    _npu_preprocess_patched = True
