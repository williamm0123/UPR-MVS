from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..depth_anything3 import DPT, DinoV2


def apply_metric_scaling(depth: Tensor, intrinsics: Tensor, scale_factor: float = 300.0) -> Tensor:
    focal_length = (intrinsics[..., 0, 0] + intrinsics[..., 1, 1]) / 2.0
    return depth * (focal_length[..., None, None] / scale_factor)


def _gradient_magnitude(map_tensor: Tensor) -> Tensor:
    grad_x = map_tensor[..., :, 1:] - map_tensor[..., :, :-1]
    grad_y = map_tensor[..., 1:, :] - map_tensor[..., :-1, :]
    grad_x = F.pad(grad_x, (0, 1, 0, 0))
    grad_y = F.pad(grad_y, (0, 0, 0, 1))
    return torch.sqrt(grad_x.square() + grad_y.square() + 1.0e-8)


def _robust_channel_scale(values: Tensor, eps: float = 1.0e-6) -> Tensor:
    batch_size = values.shape[0]
    flat = values.reshape(batch_size, -1)
    return flat.median(dim=1).values.clamp_min(eps).view(batch_size, 1, 1, 1)


def _expand_reference(reference: str | Path) -> str:
    return os.path.expandvars(os.path.expanduser(str(reference).strip()))


def _looks_like_hf_repo_id(reference: str) -> bool:
    if not reference:
        return False
    if reference.startswith(("/", "./", "../", "~")):
        return False
    path = Path(reference)
    if path.suffix:
        return False
    parts = [part for part in reference.split("/") if part]
    return len(parts) == 2


def _resolve_local_model_artifact(model_dir: Path) -> Path | None:
    for filename in ("model.safetensors", "pytorch_model.bin", "model.bin"):
        candidate = model_dir / filename
        if candidate.is_file():
            return candidate
    return None


def _download_hf_model_artifact(repo_id: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Loading a Hugging Face DA3 model id requires `huggingface_hub`. "
            "Install it or provide a local checkpoint file."
        ) from exc

    last_error: Exception | None = None
    for filename in ("model.safetensors", "pytorch_model.bin", "model.bin"):
        try:
            return Path(hf_hub_download(repo_id=repo_id, filename=filename))
        except Exception as exc:  # pragma: no cover - depends on runtime/network
            last_error = exc
    raise FileNotFoundError(
        f"Could not resolve a loadable DA3 artifact from Hugging Face repo '{repo_id}'. "
        "Expected one of: model.safetensors, pytorch_model.bin, model.bin."
    ) from last_error


def _resolve_pretrained_artifact(reference: str | Path) -> tuple[Path, str]:
    expanded = _expand_reference(reference)
    path = Path(expanded)
    if path.is_file():
        return path, f"local checkpoint file '{path}'"

    if path.is_dir():
        artifact = _resolve_local_model_artifact(path)
        if artifact is None:
            raise FileNotFoundError(
                f"Depth Anything 3 model directory '{path}' does not contain a supported artifact. "
                "Expected one of: model.safetensors, pytorch_model.bin, model.bin."
            )
        return artifact, f"local model directory '{path}'"

    if _looks_like_hf_repo_id(expanded):
        artifact = _download_hf_model_artifact(expanded)
        return artifact, f"Hugging Face repo '{expanded}'"

    raise FileNotFoundError(
        "Depth Anything 3 checkpoint could not be resolved from "
        f"'{reference}'. Provide one of: "
        "1) a local .pth/.pt/.bin/.safetensors file, "
        "2) a local model directory containing model.safetensors, or "
        "3) a Hugging Face repo id such as 'depth-anything/da3metric-large'."
    )


def _load_state_dict_payload(path: Path) -> dict[str, Any]:
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Loading a .safetensors DA3 artifact requires `safetensors`. "
                "Install it or use a .pth checkpoint."
            ) from exc
        return load_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu")


def _extract_state_dict(state: Any) -> dict[str, Tensor]:
    if isinstance(state, dict):
        for key in ("state_dict", "model"):
            candidate = state.get(key)
            if isinstance(candidate, dict):
                state = candidate
                break
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported Depth Anything 3 checkpoint format: {type(state)!r}")
    return state


def _normalize_checkpoint_key(key: str) -> str | None:
    normalized = key
    while normalized.startswith("module."):
        normalized = normalized[len("module.") :]
    while normalized.startswith("model."):
        normalized = normalized[len("model.") :]

    if normalized.startswith("net."):
        normalized = "backbone." + normalized[len("net.") :]
    if normalized.startswith("more_mlps."):
        normalized = "backbone." + normalized[len("more_mlps.") :]
    if normalized.startswith("fc_rot."):
        normalized = "fc_qvec." + normalized[len("fc_rot.") :]

    normalized = normalized.replace(".net.", ".backbone.")
    normalized = normalized.replace(".more_mlps.", ".backbone.")
    normalized = normalized.replace(".fc_rot.", ".fc_qvec.")
    normalized = normalized.replace(".camera_token_extra", ".camera_token")
    normalized = normalized.replace("output_conv2_additional.sky_mask", "sky_output_conv2")

    if normalized.startswith("backbone.encoder.") or normalized.startswith("depth_head.depth_head."):
        return normalized
    if normalized.startswith("backbone."):
        return "backbone.encoder." + normalized[len("backbone.") :]
    if normalized.startswith("head."):
        return "depth_head.depth_head." + normalized[len("head.") :]
    return None


def _build_loadable_state_dict(state_dict: dict[str, Tensor], model_state: dict[str, Tensor]) -> dict[str, Tensor]:
    loadable: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        normalized_key = _normalize_checkpoint_key(key)
        if normalized_key is None:
            continue
        if normalized_key in model_state and model_state[normalized_key].shape == value.shape:
            loadable[normalized_key] = value
    return loadable


class DepthAnything3Backbone(nn.Module):
    STAGE_NAMES = ("stage1", "stage2", "stage3")
    STAGE_SCALES = {"stage1": 8, "stage2": 16, "stage3": 32}
    VARIANT_SPECS = {
        "vits": 384,
        "vitb": 768,
        "vitl": 1024,
        "vitg": 1536,
    }

    def __init__(
        self,
        *,
        variant: str = "vitl",
        out_layers: Sequence[int] = (4, 11, 17, 23),
        feature_dim: int = 256,
        feature_layer_indices: Sequence[int] = (0, 1, 3),
        alt_start: int = -1,
        qknorm_start: int = -1,
        rope_start: int = -1,
        cat_token: bool = False,
    ) -> None:
        super().__init__()
        variant_key = str(variant).lower()
        if variant_key not in self.VARIANT_SPECS:
            raise ValueError(f"Unsupported Depth Anything 3 variant: {variant}")
        if len(feature_layer_indices) != len(self.STAGE_NAMES):
            raise ValueError(
                f"feature_layer_indices must contain {len(self.STAGE_NAMES)} entries, got {feature_layer_indices}."
            )

        self.variant = variant_key
        self.embed_dim = int(self.VARIANT_SPECS[variant_key])
        self.feature_layer_indices = tuple(int(index) for index in feature_layer_indices)
        self.out_layers = tuple(int(index) for index in out_layers)
        self.patch_size = 14

        self.encoder = DinoV2(
            name=variant_key,
            out_layers=list(self.out_layers),
            alt_start=int(alt_start),
            qknorm_start=int(qknorm_start),
            rope_start=int(rope_start),
            cat_token=bool(cat_token),
        )
        self.out_proj = nn.ModuleDict(
            {
                stage_key: nn.Conv2d(self.embed_dim, int(feature_dim), kernel_size=1)
                for stage_key in self.STAGE_NAMES
            }
        )

    def forward_multiview(self, images: Tensor) -> tuple[dict[str, Tensor], tuple[Any, ...]]:
        if images.ndim != 5:
            raise ValueError(f"Expected images [B, V, 3, H, W], got {tuple(images.shape)}")

        batch_size, num_views, _, image_h, image_w = images.shape
        if image_h % self.patch_size != 0 or image_w % self.patch_size != 0:
            raise ValueError(
                "Depth Anything 3 requires input sizes divisible by 14. "
                f"Got H={image_h}, W={image_w}. Consider using 756x1008 for DTU."
            )

        raw_features, _ = self.encoder(images, export_feat_layers=[])
        patch_h = image_h // self.patch_size
        patch_w = image_w // self.patch_size

        feature_pyramid: dict[str, Tensor] = {}
        for stage_key, source_index in zip(self.STAGE_NAMES, self.feature_layer_indices):
            tokens = raw_features[source_index][0]  # [B, V, N, C]
            _, _, num_tokens, channels = tokens.shape
            if num_tokens != patch_h * patch_w:
                raise ValueError(
                    f"Unexpected token count for {stage_key}: expected {patch_h * patch_w}, got {num_tokens}."
                )
            fmap = tokens.reshape(batch_size * num_views, patch_h, patch_w, channels).permute(0, 3, 1, 2).contiguous()
            fmap = self.out_proj[stage_key](fmap)
            target_hw = (
                max(1, image_h // self.STAGE_SCALES[stage_key]),
                max(1, image_w // self.STAGE_SCALES[stage_key]),
            )
            fmap = F.interpolate(fmap, size=target_hw, mode="bilinear", align_corners=False)
            feature_pyramid[stage_key] = fmap.reshape(batch_size, num_views, fmap.shape[1], *target_hw)

        return feature_pyramid, raw_features


class DepthAnything3MetricHead(nn.Module):
    def __init__(
        self,
        *,
        dim_in: int,
        patch_size: int = 14,
        metric_scale_factor: float = 300.0,
        depth_conf_beta: float = 2.0,
        image_texture_weight: float = 0.35,
        use_sky_head: bool = True,
    ) -> None:
        super().__init__()
        self.depth_head = DPT(
            dim_in=int(dim_in),
            patch_size=int(patch_size),
            output_dim=1,
            features=256,
            out_channels=(256, 512, 1024, 1024),
            use_sky_head=bool(use_sky_head),
        )
        self.metric_scale_factor = float(metric_scale_factor)
        self.depth_conf_beta = float(depth_conf_beta)
        self.image_texture_weight = float(image_texture_weight)
        self.sky_threshold = 0.3

    def _build_confidence_maps(self, depth: Tensor, ref_image: Tensor, sky_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        valid_mask = torch.isfinite(depth) & (depth > 0.0)
        depth_grad = _gradient_magnitude(depth)
        depth_scale = _robust_channel_scale(depth_grad)
        reliability = torch.exp(-(depth_grad / depth_scale) * self.depth_conf_beta)

        gray = ref_image.mean(dim=1, keepdim=True)
        image_grad = _gradient_magnitude(gray)
        image_scale = _robust_channel_scale(image_grad)
        texture = torch.tanh(image_grad / image_scale)

        confidence = reliability * valid_mask.float()
        selection_score = confidence * (1.0 + self.image_texture_weight * texture)

        if sky_mask is not None:
            non_sky = (~sky_mask).float()
            confidence = confidence * non_sky
            selection_score = selection_score * non_sky

        confidence = confidence.clamp(0.0, 1.0)
        selection_score = selection_score.clamp_min(0.0)
        return confidence, selection_score

    def forward(
        self,
        *,
        raw_features: tuple[Any, ...],
        images: Tensor,
        intrinsics: Tensor,
    ) -> dict[str, Tensor]:
        if images.ndim != 5:
            raise ValueError(f"Expected images [B, V, 3, H, W], got {tuple(images.shape)}")
        if intrinsics.ndim != 4:
            raise ValueError(f"Expected intrinsics [B, V, 3, 3], got {tuple(intrinsics.shape)}")

        image_h, image_w = images.shape[-2:]
        head_outputs = self.depth_head(list(raw_features), image_h, image_w, patch_start_idx=0)

        ref_depth = head_outputs["depth"][:, :1]
        ref_intrinsics = intrinsics[:, :1]
        metric_depth = apply_metric_scaling(ref_depth, ref_intrinsics, self.metric_scale_factor)

        sky_mask = None
        if "sky" in head_outputs:
            sky_mask = head_outputs["sky"][:, :1] >= self.sky_threshold
        coarse_confidence, point_selection_score = self._build_confidence_maps(
            depth=metric_depth,
            ref_image=images[:, 0],
            sky_mask=sky_mask,
        )

        return {
            "coarse_depth": metric_depth,
            "coarse_confidence": coarse_confidence,
            "point_selection_score": point_selection_score,
            "depth_prior_raw": ref_depth,
        }


class DepthAnything3MetricPrior(nn.Module):
    def __init__(self, prior_cfg: dict[str, Any]) -> None:
        super().__init__()
        variant = str(prior_cfg.get("variant", "vitl")).lower()
        out_layers = prior_cfg.get("out_layers", [4, 11, 17, 23])
        feature_dim = int(prior_cfg.get("feature_dim", 256))

        self.backbone = DepthAnything3Backbone(
            variant=variant,
            out_layers=out_layers,
            feature_dim=feature_dim,
            feature_layer_indices=prior_cfg.get("feature_layer_indices", [0, 1, 3]),
            alt_start=int(prior_cfg.get("alt_start", -1)),
            qknorm_start=int(prior_cfg.get("qknorm_start", -1)),
            rope_start=int(prior_cfg.get("rope_start", -1)),
            cat_token=bool(prior_cfg.get("cat_token", False)),
        )
        self.depth_head = DepthAnything3MetricHead(
            dim_in=self.backbone.embed_dim,
            patch_size=self.backbone.patch_size,
            metric_scale_factor=float(prior_cfg.get("metric_scale_factor", 300.0)),
            depth_conf_beta=float(prior_cfg.get("depth_conf_beta", 2.0)),
            image_texture_weight=float(prior_cfg.get("image_texture_weight", 0.35)),
            use_sky_head=bool(prior_cfg.get("use_sky_head", True)),
        )

        pretrained = prior_cfg.get("pretrained")
        if not pretrained:
            raise ValueError(
                "model.depth_anything3.pretrained must resolve to a DA3 weight source. "
                "Supported inputs are: a local checkpoint file, a local model directory, "
                "or a Hugging Face repo id such as 'depth-anything/da3metric-large'."
            )
        self.load_pretrained(pretrained)

    def load_pretrained(self, checkpoint_reference: str | Path) -> None:
        artifact_path, source_desc = _resolve_pretrained_artifact(checkpoint_reference)
        state = _extract_state_dict(_load_state_dict_payload(artifact_path))
        model_state = self.state_dict()
        loadable = _build_loadable_state_dict(state, model_state)

        missing, unexpected = self.load_state_dict(loadable, strict=False)
        if not loadable:
            raise RuntimeError(
                "No compatible Depth Anything 3 weights were loaded from "
                f"{source_desc} ({artifact_path})."
            )
        if unexpected:
            raise RuntimeError(f"Unexpected keys while loading Depth Anything 3 checkpoint: {unexpected[:10]}")
        ignored_missing = [key for key in missing if "backbone.out_proj" in key]
        real_missing = [key for key in missing if key not in ignored_missing]
        if real_missing:
            raise RuntimeError(
                "Missing keys while loading Depth Anything 3 checkpoint: "
                + ", ".join(real_missing[:10])
            )

    def forward(self, images: Tensor, intrinsics: Tensor) -> dict[str, Tensor]:
        feature_pyramid, raw_features = self.backbone.forward_multiview(images)
        coarse_outputs = self.depth_head(raw_features=raw_features, images=images, intrinsics=intrinsics)
        return {
            "feature_pyramid": feature_pyramid,
            **coarse_outputs,
        }
