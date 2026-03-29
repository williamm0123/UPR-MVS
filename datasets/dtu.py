from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ResizeCropTransform:
    scale: float
    resized_h: int
    resized_w: int
    crop_y: int
    crop_x: int
    target_h: int
    target_w: int


@dataclass(frozen=True)
class DTUSampleMeta:
    scan_name: str
    ref_view: int
    src_views: tuple[int, ...]
    light_id: int
    pair_file: str | None = None


def load_scan_list(list_file: str | Path) -> list[str]:
    path = Path(list_file)
    if not path.is_file():
        raise FileNotFoundError(f"DTU split file not found: {path}")
    scan_names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not scan_names:
        raise ValueError(f"DTU split file is empty: {path}")
    return scan_names


def read_pair_file(pair_file: str | Path) -> dict[int, list[int]]:
    path = Path(pair_file)
    if not path.is_file():
        raise FileNotFoundError(f"DTU pair file not found: {path}")

    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"DTU pair file is empty: {path}")

    num_viewpoints = int(lines[0])
    pairs: dict[int, list[int]] = {}
    cursor = 1
    for _ in range(num_viewpoints):
        if cursor + 1 >= len(lines):
            break
        ref_view = int(lines[cursor])
        pair_tokens = lines[cursor + 1].split()
        num_src = int(pair_tokens[0]) if pair_tokens else 0
        src_views: list[int] = []
        for src_idx in range(num_src):
            token_index = 1 + src_idx * 2
            if token_index < len(pair_tokens):
                src_views.append(int(pair_tokens[token_index]))
        pairs[ref_view] = src_views
        cursor += 2
    return pairs


def _find_section_index(lines: Sequence[str], prefix: str) -> int:
    normalized_prefix = prefix.lower()
    for idx, line in enumerate(lines):
        if line.lower().startswith(normalized_prefix):
            return idx
    raise ValueError(f"Could not find section '{prefix}' in camera file.")


def read_camera_file(cam_file: str | Path) -> tuple[np.ndarray, np.ndarray, float, float | None]:
    path = Path(cam_file)
    if not path.is_file():
        raise FileNotFoundError(f"Camera file not found: {path}")

    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    extr_idx = _find_section_index(lines, "extrinsic")
    intr_idx = _find_section_index(lines, "intrinsic")

    extrinsics = np.fromstring(" ".join(lines[extr_idx + 1 : extr_idx + 5]), sep=" ", dtype=np.float32).reshape(4, 4)
    intrinsics = np.fromstring(" ".join(lines[intr_idx + 1 : intr_idx + 4]), sep=" ", dtype=np.float32).reshape(3, 3)

    depth_min: float | None = None
    depth_max: float | None = None
    if intr_idx + 4 < len(lines):
        depth_meta = np.fromstring(lines[intr_idx + 4], sep=" ", dtype=np.float32)
        if depth_meta.size >= 1:
            depth_min = float(depth_meta[0])
        if depth_meta.size >= 4:
            depth_interval = float(depth_meta[1])
            depth_count = int(depth_meta[2])
            depth_max = depth_min + depth_interval * max(depth_count - 1, 1)
        elif depth_meta.size >= 2:
            depth_max = float(depth_meta[1])
    if depth_min is None:
        raise ValueError(f"Camera file missing depth range: {path}")
    return intrinsics, extrinsics, depth_min, depth_max


def read_pfm(pfm_file: str | Path) -> np.ndarray:
    path = Path(pfm_file)
    with path.open("rb") as handle:
        header = handle.readline().decode("latin-1").strip()
        if header not in {"PF", "Pf"}:
            raise ValueError(f"Unsupported PFM header in {path}: {header}")
        color = header == "PF"

        dims_line = handle.readline().decode("latin-1").strip()
        while dims_line.startswith("#"):
            dims_line = handle.readline().decode("latin-1").strip()
        width_str, height_str = dims_line.split()
        width, height = int(width_str), int(height_str)

        scale = float(handle.readline().decode("latin-1").strip())
        endian = "<" if scale < 0 else ">"
        data = np.fromfile(handle, endian + "f")
        channels = 3 if color else 1
        shape = (height, width, channels) if color else (height, width)
        data = np.reshape(data, shape)
        return np.flipud(data).astype(np.float32)


def compute_resize_crop_transform(orig_h: int, orig_w: int, target_h: int, target_w: int) -> ResizeCropTransform:
    scale = max(target_h / float(orig_h), target_w / float(orig_w))
    resized_h = int(round(orig_h * scale))
    resized_w = int(round(orig_w * scale))
    crop_y = max((resized_h - target_h) // 2, 0)
    crop_x = max((resized_w - target_w) // 2, 0)
    return ResizeCropTransform(
        scale=scale,
        resized_h=resized_h,
        resized_w=resized_w,
        crop_y=crop_y,
        crop_x=crop_x,
        target_h=target_h,
        target_w=target_w,
    )


def resize_and_crop_intrinsics(intrinsics: np.ndarray, transform: ResizeCropTransform) -> np.ndarray:
    scaled = intrinsics.copy().astype(np.float32)
    scaled[0, :] *= transform.scale
    scaled[1, :] *= transform.scale
    scaled[0, 2] -= float(transform.crop_x)
    scaled[1, 2] -= float(transform.crop_y)
    return scaled


def apply_transform_to_image(image: Image.Image, transform: ResizeCropTransform) -> Image.Image:
    resized = image.resize((transform.resized_w, transform.resized_h), resample=Image.BILINEAR)
    return resized.crop(
        (transform.crop_x, transform.crop_y, transform.crop_x + transform.target_w, transform.crop_y + transform.target_h)
    )


def apply_transform_to_array(array: np.ndarray, transform: ResizeCropTransform, mode: str) -> np.ndarray:
    tensor = torch.from_numpy(array).float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    else:
        raise ValueError(f"Unsupported array shape for resize/crop: {array.shape}")

    interpolate_kwargs = {
        "input": tensor,
        "size": (transform.resized_h, transform.resized_w),
        "mode": mode,
    }
    if mode in {"bilinear", "bicubic"}:
        interpolate_kwargs["align_corners"] = False
    resized = F.interpolate(**interpolate_kwargs)

    cropped = resized[
        :,
        :,
        transform.crop_y : transform.crop_y + transform.target_h,
        transform.crop_x : transform.crop_x + transform.target_w,
    ]
    if array.ndim == 2:
        return cropped.squeeze(0).squeeze(0).cpu().numpy()
    return cropped.squeeze(0).permute(1, 2, 0).cpu().numpy()


def image_to_tensor(image: Image.Image) -> Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def load_depth(depth_path: Path, depth_scale: float) -> np.ndarray:
    if depth_path.suffix.lower() == ".pfm":
        depth = read_pfm(depth_path)
    else:
        depth = np.asarray(Image.open(depth_path), dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth_scale != 1.0:
        depth = depth / depth_scale
    return depth.astype(np.float32)


def load_mask(mask_path: Path | None, depth: np.ndarray) -> np.ndarray:
    if mask_path is None:
        return (depth > 0.0).astype(np.float32)
    mask = np.asarray(Image.open(mask_path), dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.max() > 1.0:
        mask = mask / 255.0
    return (mask > 0.5).astype(np.float32)


def build_dtu_dataset(config: dict[str, Any], split: str) -> "DTUMVSDataset":
    list_key = f"{split}_list"
    root_key = f"{split}_root"
    layout_key = f"{split}_layout"
    gt_root_key = f"{split}_gt_root"

    if list_key not in config:
        raise KeyError(f"Dataset config missing split list path: {list_key}")

    root = config.get(root_key, config.get("root"))
    if root is None:
        raise KeyError(f"Dataset config missing root path for split '{split}'.")

    layout = str(config.get(layout_key, "dtu_test" if split == "test" else "trainval")).lower()
    gt_root = config.get(gt_root_key)

    return DTUMVSDataset(
        root=root,
        split=split,
        layout=layout,
        list_file=config[list_key],
        gt_root=gt_root,
        n_views=int(config["n_views"]),
        img_h=int(config["img_h"]),
        img_w=int(config["img_w"]),
        pair_file=config.get("pair_file", "Cameras/pair.txt"),
        camera_dir=config.get("camera_dir", "Cameras"),
        rectified_dir=config.get("rectified_dir", "DTU_origin/Rectified"),
        depth_dir=config.get("depth_dir", "Depths_raw"),
        mask_dir=config.get("mask_dir", "Depths_raw"),
        train_light_ids=config.get("train_light_ids", [0, 1, 2, 3, 4, 5, 6]),
        eval_light_ids=config.get("eval_light_ids", [3]),
        test_image_dir=config.get("test_image_dir", "images"),
        test_camera_dirs=config.get("test_camera_dirs", ["cams_1", "cams"]),
        test_pair_file=config.get("test_pair_file", "pair.txt"),
        depth_scale=float(config.get("depth_scale", 1.0)),
    )


class DTUMVSDataset(Dataset[dict[str, Any]]):
    """DTU dataset loader supporting both mvs_training and dtu_test layouts."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        layout: str,
        list_file: str | Path,
        n_views: int,
        img_h: int,
        img_w: int,
        gt_root: str | Path | None = None,
        pair_file: str | Path = "Cameras/pair.txt",
        camera_dir: str | Path = "Cameras",
        rectified_dir: str | Path = "DTU_origin/Rectified",
        depth_dir: str | Path = "Depths_raw",
        mask_dir: str | Path | None = "Depths_raw",
        train_light_ids: Sequence[int] | None = None,
        eval_light_ids: Sequence[int] | None = None,
        test_image_dir: str | Path = "images",
        test_camera_dirs: Sequence[str | Path] | None = None,
        test_pair_file: str | Path = "pair.txt",
        depth_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if n_views < 2:
            raise ValueError("DTU dataset requires at least 2 views per sample.")
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        if layout not in {"trainval", "dtu_test"}:
            raise ValueError(f"Unsupported DTU layout: {layout}")

        self.root = Path(root)
        self.split = split
        self.layout = layout
        self.n_views = n_views
        self.img_h = img_h
        self.img_w = img_w
        self.depth_scale = depth_scale

        self.pair_file = self._resolve_path(pair_file)
        self.camera_dir = self._resolve_path(camera_dir)
        self.rectified_dir = self._resolve_path(rectified_dir)
        self.depth_dir = self._resolve_path(depth_dir)
        self.mask_dir = self._resolve_path(mask_dir) if mask_dir is not None else None
        self.gt_root = self._resolve_path(gt_root) if gt_root is not None else self._default_gt_root()
        self.test_image_dir = str(test_image_dir)
        self.test_camera_dirs = [str(path_like) for path_like in (test_camera_dirs or ["cams_1", "cams"])]
        self.test_pair_file = str(test_pair_file)

        self.scan_names = load_scan_list(self._resolve_path(list_file))
        self.light_ids = list(train_light_ids or [0, 1, 2, 3, 4, 5, 6]) if split == "train" else list(eval_light_ids or [3])
        self.samples = self._build_samples()
        if not self.samples:
            raise RuntimeError(
                f"No valid DTU samples found for split={split}, layout={layout}. "
                "Check the scan list, root path, and directory layout."
            )

    def _resolve_path(self, path_like: str | Path | None) -> Path:
        if path_like is None:
            raise ValueError("Expected a valid path, but received None.")
        path = Path(path_like)
        if path.is_absolute():
            return path
        if path.exists():
            return path.resolve()
        return self.root / path

    def _default_gt_root(self) -> Path | None:
        if self.layout != "dtu_test":
            return self.root

        candidates = [
            self.root.parent / "mvs_training",
            self.root.parent / "dtu_training",
            self.root.parent / "DTU" / "mvs_training",
            self.root.parent / "DTU" / "dtu_training",
        ]
        for candidate in candidates:
            if (candidate / "Depths_raw").exists():
                return candidate
        return None

    def _build_samples(self) -> list[DTUSampleMeta]:
        if self.layout == "trainval":
            return self._build_trainval_samples()
        return self._build_dtu_test_samples()

    def _build_trainval_samples(self) -> list[DTUSampleMeta]:
        samples: list[DTUSampleMeta] = []
        view_pairs = read_pair_file(self.pair_file)
        light_ids = self.light_ids if self.split == "train" else [self.light_ids[0]]

        for scan_name in self.scan_names:
            for ref_view, src_views in view_pairs.items():
                if len(src_views) < self.n_views - 1:
                    continue
                selected_src_views = tuple(src_views[: self.n_views - 1])
                for light_id in light_ids:
                    samples.append(
                        DTUSampleMeta(
                            scan_name=scan_name,
                            ref_view=ref_view,
                            src_views=selected_src_views,
                            light_id=int(light_id),
                        )
                    )
        return samples

    def _build_dtu_test_samples(self) -> list[DTUSampleMeta]:
        samples: list[DTUSampleMeta] = []
        for scan_name in self.scan_names:
            pair_path = self.root / scan_name / self.test_pair_file
            view_pairs = read_pair_file(pair_path)
            for ref_view, src_views in view_pairs.items():
                if len(src_views) < self.n_views - 1:
                    continue
                samples.append(
                    DTUSampleMeta(
                        scan_name=scan_name,
                        ref_view=ref_view,
                        src_views=tuple(src_views[: self.n_views - 1]),
                        light_id=0,
                        pair_file=str(pair_path),
                    )
                )
        return samples

    def _first_existing_path(self, candidates: Iterable[Path], description: str) -> Path:
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        candidate_list = "\n".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Unable to locate {description}. Checked:\n{candidate_list}")

    def _camera_candidates(self, scan_name: str, view_id: int) -> list[Path]:
        if self.layout == "trainval":
            return [
                self.camera_dir / f"{view_id:08d}_cam.txt",
                self.camera_dir / f"{view_id:04d}_cam.txt",
                self.camera_dir / f"cam_{view_id:08d}.txt",
            ]

        scan_root = self.root / scan_name
        candidates: list[Path] = []
        for camera_dir in self.test_camera_dirs:
            candidates.extend(
                [
                    scan_root / camera_dir / f"{view_id:08d}_cam.txt",
                    scan_root / camera_dir / f"{view_id:04d}_cam.txt",
                    scan_root / camera_dir / f"cam_{view_id:08d}.txt",
                ]
            )
        return candidates

    def _image_candidates(self, scan_name: str, view_id: int, light_id: int) -> list[Path]:
        if self.layout == "trainval":
            one_based_view = view_id + 1
            candidates: list[Path] = []
            for suffix in (".png", ".jpg", ".jpeg"):
                candidates.extend(
                    [
                        self.rectified_dir / scan_name / f"rect_{one_based_view:03d}_{light_id}_r5000{suffix}",
                        self.rectified_dir / f"{scan_name}_train" / f"rect_{one_based_view:03d}_{light_id}_r5000{suffix}",
                    ]
                )
            return candidates

        scan_root = self.root / scan_name / self.test_image_dir
        candidates = []
        for suffix in (".jpg", ".png", ".jpeg"):
            candidates.extend(
                [
                    scan_root / f"{view_id:08d}{suffix}",
                    scan_root / f"{view_id:04d}{suffix}",
                ]
            )
        return candidates

    def _depth_candidates(self, scan_name: str, view_id: int) -> list[Path]:
        if self.gt_root is None:
            return []
        depth_root = self.gt_root / self.depth_dir.name
        return [
            depth_root / scan_name / f"depth_map_{view_id:04d}.pfm",
            depth_root / scan_name / f"depth_map_{view_id:08d}.pfm",
            depth_root / scan_name / f"depth_map_{view_id:04d}.png",
            depth_root / scan_name / f"depth_map_{view_id:08d}.png",
        ]

    def _mask_candidates(self, scan_name: str, view_id: int) -> list[Path]:
        if self.gt_root is None or self.mask_dir is None:
            return []
        mask_root = self.gt_root / self.mask_dir.name
        return [
            mask_root / scan_name / f"depth_visual_{view_id:04d}.png",
            mask_root / scan_name / f"depth_mask_{view_id:04d}.png",
            mask_root / scan_name / f"mask_{view_id:04d}.png",
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_meta = self.samples[index]
        scan_name = sample_meta.scan_name
        ref_view = sample_meta.ref_view
        src_views = list(sample_meta.src_views)
        light_id = sample_meta.light_id
        view_ids = [ref_view, *src_views]

        images: list[Tensor] = []
        intrinsics_list: list[Tensor] = []
        extrinsics_list: list[Tensor] = []
        transform: ResizeCropTransform | None = None
        depth_min: float | None = None
        depth_max: float | None = None

        for view_id in view_ids:
            cam_path = self._first_existing_path(
                self._camera_candidates(scan_name, view_id),
                description=f"camera for scan={scan_name}, view={view_id}",
            )
            intrinsics, extrinsics, candidate_depth_min, candidate_depth_max = read_camera_file(cam_path)

            image_path = self._first_existing_path(
                self._image_candidates(scan_name, view_id, light_id),
                description=f"image for scan={scan_name}, view={view_id}, light={light_id}",
            )
            image = Image.open(image_path).convert("RGB")

            if transform is None:
                orig_w, orig_h = image.size
                transform = compute_resize_crop_transform(
                    orig_h=orig_h,
                    orig_w=orig_w,
                    target_h=self.img_h,
                    target_w=self.img_w,
                )

            image = apply_transform_to_image(image, transform)
            intrinsics = resize_and_crop_intrinsics(intrinsics, transform)

            images.append(image_to_tensor(image))
            intrinsics_list.append(torch.from_numpy(intrinsics))
            extrinsics_list.append(torch.from_numpy(extrinsics.astype(np.float32)))

            if view_id == ref_view:
                depth_min = candidate_depth_min
                depth_max = candidate_depth_max

        if transform is None or depth_min is None:
            raise RuntimeError(f"Failed to construct DTU sample at index {index}")

        depth = np.zeros((self.img_h, self.img_w), dtype=np.float32)
        mask = np.zeros((self.img_h, self.img_w), dtype=np.float32)
        has_depth_gt = False

        depth_candidates = self._depth_candidates(scan_name, ref_view)
        if depth_candidates:
            try:
                depth_path = self._first_existing_path(
                    depth_candidates,
                    description=f"depth for scan={scan_name}, ref_view={ref_view}",
                )
                depth = load_depth(depth_path, depth_scale=self.depth_scale)
                mask_path: Path | None = None
                try:
                    mask_path = self._first_existing_path(
                        self._mask_candidates(scan_name, ref_view),
                        description=f"mask for scan={scan_name}, ref_view={ref_view}",
                    )
                except FileNotFoundError:
                    mask_path = None
                mask = load_mask(mask_path, depth)
                depth = apply_transform_to_array(depth, transform, mode="bilinear")
                mask = apply_transform_to_array(mask, transform, mode="nearest")
                mask = (mask > 0.5).astype(np.float32)
                has_depth_gt = True
            except FileNotFoundError:
                has_depth_gt = False

        if depth_max is None:
            valid_depth = depth[mask > 0.5]
            depth_max = float(valid_depth.max()) if valid_depth.size > 0 else depth_min + 1.0

        batch = {
            "imgs": torch.stack(images, dim=0),  # [V, 3, H, W]
            "intrinsics": torch.stack(intrinsics_list, dim=0).float(),  # [V, 3, 3]
            "extrinsics": torch.stack(extrinsics_list, dim=0).float(),  # [V, 4, 4]
            "depth_gt": torch.from_numpy(depth).unsqueeze(0).float(),  # [1, H, W]
            "mask": torch.from_numpy(mask).unsqueeze(0).float(),  # [1, H, W]
            "depth_range": torch.tensor([depth_min, depth_max], dtype=torch.float32),
            "view_ids": torch.tensor(view_ids, dtype=torch.long),
            "ref_view": torch.tensor(ref_view, dtype=torch.long),
            "src_views": torch.tensor(src_views, dtype=torch.long),
            "light_id": torch.tensor(light_id, dtype=torch.long),
            "has_depth_gt": torch.tensor(has_depth_gt, dtype=torch.bool),
            "scan_name": scan_name,
            "sample_name": f"{scan_name}_view{ref_view:03d}_light{light_id}",
        }
        return batch
