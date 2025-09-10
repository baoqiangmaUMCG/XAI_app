"""
XAI + Evaluation App (Gradio, PyTorch, Captum)
================================================
- Demo model: **DenseNet121 (ImageNet pretrained)** with human-readable labels
- Upload: TorchScript `.pt` (optionally supply a class map JSON: ["class0", "class1", ...])
- Inputs: 2D image (PNG/JPG) or 3D NIfTI slice (.nii/.nii.gz + index)
- XAI: Saliency, Integrated Gradients, DeepLIFT, Grad-CAM, Occlusion
- Evaluation: Deletion/Insertion AOPC, Pointing Game, Mask ROC-AUC (with JSON report export)
- Overlay alignment: heatmaps overlay the **exact resized+center-cropped view** used by the model

Run:  `CUDA_VISIBLE_DEVICES=0 python app.py`
"""

import io
import json
import os, shutil, time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import os, sys
if getattr(sys, 'frozen', False):
    lib_dir = os.path.join(sys._MEIPASS, "torch", "lib")
    os.environ["LD_LIBRARY_PATH"] = lib_dir + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import torch
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms import functional as TF

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True 

import gradio_client.utils as gu
_original_json_schema_to_python_type = gu._json_schema_to_python_type
def safe_json_schema_to_python_type(schema, defs=None):
    if schema is True or schema is False:  # guard against bool
        return "any"
    return _original_json_schema_to_python_type(schema, defs)
gu._json_schema_to_python_type = safe_json_schema_to_python_type

# Optional medical I/O
try:
    import SimpleITK as sitk
    HAS_SITK = True
except Exception:
    HAS_SITK = False

try:
    import nibabel as nib
    HAS_NIB = True
except Exception:
    HAS_NIB = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from captum.attr import (
    Saliency,
    IntegratedGradients,
    InputXGradient,
    GuidedBackprop,
    Deconvolution,
    DeepLift,
    DeepLiftShap,
    GradientShap,
    Occlusion,
    FeatureAblation,
    ShapleyValueSampling,
    NoiseTunnel,
)

from captum.metrics import infidelity


from sklearn.metrics import roc_auc_score
import gradio as gr

# ----------------------------
# Config
# ----------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = "mps"   # Apple Silicon GPU
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print(f"Using device: {DEVICE}")

# Preprocessing
def preprocess_both(pil_img: Image.Image):
    img_resized = TF.resize(pil_img, 256)
    img_cropped = TF.center_crop(img_resized, 224)
    x = TF.to_tensor(img_cropped)
    x = TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
    return x.unsqueeze(0).to(DEVICE), img_cropped

# ----------------------------
# I/O
# ----------------------------
def _as_3ch_pil(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGB" else img.convert("RGB")

def load_image_2d(file: gr.File) -> Image.Image:
    return _as_3ch_pil(Image.open(file.name).convert("RGB"))

def load_nifti_slice(file: gr.File, slice_index: int) -> Image.Image:
    vol = None
    if HAS_SITK:
        img = sitk.ReadImage(file.name)
        vol = sitk.GetArrayFromImage(img)
    elif HAS_NIB:
        img = nib.load(file.name)
        vol = img.get_fdata()
        if vol.ndim == 4:
            vol = vol[..., 0]
        vol = np.transpose(vol, (2, 1, 0))
    else:
        raise RuntimeError("Install SimpleITK or nibabel to read NIfTI.")
    D = vol.shape[0]
    s = int(np.clip(slice_index, 0, D - 1))
    sl = vol[s].astype(np.float32)
    if np.ptp(sl) > 0:
        sl = (sl - sl.min()) / (sl.max() - sl.min())
    pil = Image.fromarray((sl * 255).astype(np.uint8), mode="L").convert("RGB")
    return pil

# ----------------------------
# Models
# ----------------------------
@dataclass
class LoadedModel:
    model: torch.nn.Module
    kind: str
    target_layer: Optional[torch.nn.Module]
    categories: Optional[List[str]] = None

def load_demo_model() -> LoadedModel:
    weights = models.DenseNet121_Weights.DEFAULT
    model = models.densenet121(weights=weights).to(DEVICE).eval()
    target_layer = None
    for m in reversed(list(model.modules())):
        if isinstance(m, torch.nn.Conv2d):
            target_layer = m
            break
    categories = weights.meta.get("categories") if hasattr(weights, "meta") else None
    return LoadedModel(model=model, kind="demo", target_layer=target_layer, categories=categories)

def _find_last_conv(m):
    # Works for both eager (nn.Module) and TorchScript (RecursiveScriptModule)
    last_conv = None
    for sub in m.modules():
        if isinstance(sub, torch.nn.Conv2d):
            last_conv = sub
        elif hasattr(sub, "original_name") and sub.original_name == "Conv2d":
            last_conv = sub
    return last_conv



def load_torchscript(path: str, do_warmup: bool = False) -> LoadedModel:
    import shutil, os, time
    t0 = time.perf_counter()

    fast_path = "/dev/shm/upload_model.pt"
    try:
        shutil.copy2(path, fast_path)
        load_path = fast_path
    except Exception as e:
        print(f"[load_torchscript] copy failed ({e}), fallback to {path}")
        load_path = path

    # Load TorchScript
    model = torch.jit.load(load_path, map_location="cpu").eval()

    #  Force move to GPU if available
    if DEVICE == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
        print(" Model moved to CUDA")
    else:
        print(" Model kept on CPU")

    elapsed = time.perf_counter() - t0
    print(f"[load_torchscript] Finished load in {elapsed:.2f}s")

    # Pick a conv layer for Grad-CAM
    '''
    target_layer = None
    for m in reversed(list(model.modules())):
        if isinstance(m, torch.nn.Conv2d):
            target_layer = m
            break
    '''
    
    target_layer = _find_last_conv(model)
    if target_layer is None:
        print(" No Conv2d layer found in TorchScript!")
    else:
        print(f" Found Conv2d layer for Grad-CAM: {target_layer}")        

    return LoadedModel(model=model, kind="uploaded", target_layer=target_layer, categories=None)



def load_torchvision_model(model_name: str) -> LoadedModel:
    # Get the constructor dynamically
    if not hasattr(models, model_name):
        raise gr.Error(f"torchvision has no model '{model_name}'")

    # Load with pretrained weights
    constructor = getattr(models, model_name)
    try:
        weights_attr = getattr(models, f"{model_name}_Weights", None)
        weights = weights_attr.DEFAULT if weights_attr is not None else None
        model = constructor(weights=weights).to(DEVICE).eval()
        categories = weights.meta.get("categories") if weights and hasattr(weights, "meta") else None
    except Exception:
        # Fallback for older API
        model = constructor(pretrained=True).to(DEVICE).eval()
        categories = None

    # Pick a conv layer for Grad-CAM
    target_layer = None
    for m in reversed(list(model.modules())):
        if isinstance(m, torch.nn.Conv2d):
            target_layer = m
            break

    return LoadedModel(model=model, kind=f"torchvision:{model_name}", target_layer=target_layer, categories=categories)

# Collect only classification models
available_models = [
    "resnet18", "resnet50", "densenet121", "mobilenet_v2", "efficientnet_b0",
    "vit_b_16", "swin_t", "convnext_tiny"
]

from monai.networks.nets import DenseNet121, EfficientNetBN, ViT

def load_monai_model(name: str) -> LoadedModel:
    if name == "densenet121":
        model = DenseNet121(spatial_dims=2, in_channels=3, out_channels=2)  # example 2-class
    elif name == "efficientnet_b0":
        model = EfficientNetBN("efficientnet-b0", spatial_dims=2, in_channels=3, num_classes=2)
    elif name == "vit":
        model = ViT(in_channels=3, img_size=(224,224), patch_size=16, num_classes=2)
    else:
        raise gr.Error(f"MONAI has no model '{name}'")

    model = model.to(DEVICE).eval()
    target_layer = _find_last_conv(model)
    return LoadedModel(model=model, kind=f"monai:{name}", target_layer=target_layer, categories=None)

monai_models = ["densenet121", "efficientnet_b0", "vit"]


# For hugging face
from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch.nn as nn

# Wrapper so HuggingFace model returns logits as a plain tensor
class HFModelWrapper(nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, x):
        out = self.hf_model(x)
        if hasattr(out, "logits"):
            return out.logits  # return tensor for Captum
        return out

def load_transformer_model(model_name: str) -> LoadedModel:
    hf_model = AutoModelForImageClassification.from_pretrained(model_name).to(DEVICE).eval()
    model = HFModelWrapper(hf_model)  # wrap so Captum sees tensor
    processor = AutoImageProcessor.from_pretrained(model_name)

    # âš ï¸ Transformers donâ€™t have conv maps â†’ disable Grad-CAM
    target_layer = None
    print(f"[INFO] Grad-CAM disabled for Transformer model {model_name}")

    categories = hf_model.config.id2label if hasattr(hf_model.config, "id2label") else None

    return LoadedModel(model=model, kind=f"transformer:{model_name}", target_layer=target_layer, categories=categories) 

# --- New: Custom architecture loader (with optional safetensors) ---
from safetensors.torch import load_file
import importlib.util, sys

def load_custom_architecture(py_file, st_file=None) -> LoadedModel:
    spec = importlib.util.spec_from_file_location("custom_model", py_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules["custom_model"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "build_model"):
        raise gr.Error("Python file must define build_model() returning nn.Module.")
    model = module.build_model().to(DEVICE).eval()
    if st_file is not None:
        try:
            state_dict = load_file(st_file)
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded weights from {st_file}")
        except Exception as e:
            raise gr.Error(f"Failed to load SafeTensors: {e}")
    else:
        print("No SafeTensors uploaded â†’ using random weights.")
    target_layer = _find_last_conv(model)
    return LoadedModel(model=model, kind="custom_architecture", target_layer=target_layer, categories=None) 


# ----------------------------
# Prediction & XAI
# ----------------------------
def predict_topk(model, x, categories=None, k=5):
    with torch.no_grad():
        logits = model(x.to(next(model.parameters(), torch.zeros(1, device=x.device)).device))

        # HuggingFace models return an object with .logits
        if hasattr(logits, "logits"):
            logits = logits.logits

        if logits.ndim == 1:
            logits = logits[None]
        probs = F.softmax(logits, dim=-1)[0]
        topk = torch.topk(probs, min(k, probs.numel()))
        items = []
        for p, idx in zip(topk.values.tolist(), topk.indices.tolist()):
            name = categories[idx] if categories and idx < len(categories) else f"class_{idx}"
            items.append({"index": int(idx), "label": name, "prob": float(p)})
    return items

def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()

def _normalize_map(a: np.ndarray) -> np.ndarray:
    a = a - a.min()
    return a / ((a.max() - a.min()) + 1e-8)

def xai_saliency(model, inp, target):
    inp.requires_grad_()
    sal = Saliency(model)
    attr = sal.attribute(inp, target=target)
    attr = attr.abs().mean(dim=1, keepdim=True)
    return _normalize_map(_to_numpy(attr)[0, 0])

def xai_integrated_gradients(model, inp, target, steps=50):
    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(inp)
    attr = ig.attribute(inp, baseline, target=target, n_steps=steps)
    attr = attr.abs().mean(dim=1, keepdim=True)
    return _normalize_map(_to_numpy(attr)[0, 0])

def xai_deeplift(model, inp, target):
    dl = DeepLift(model)
    baseline = torch.zeros_like(inp)
    attr = dl.attribute(inp, baseline, target=target)
    attr = attr.abs().mean(dim=1, keepdim=True)
    return _normalize_map(_to_numpy(attr)[0, 0])

def xai_gradcam(model_pack: LoadedModel, inp, target):
    if model_pack.target_layer is None:
        raise RuntimeError("No conv layer found for Grad-CAM.")
    lgc = LayerGradCam(model_pack.model, model_pack.target_layer)
    attr = lgc.attribute(inp, target=target)
    attr_up = F.interpolate(attr, size=inp.shape[-2:], mode="bilinear", align_corners=False)
    attr_up = attr_up.mean(dim=1, keepdim=True)
    return _normalize_map(_to_numpy(attr_up)[0, 0])

def xai_occlusion(model, inp, target, patch=15, stride=8):
    occ = Occlusion(model)
    attr = occ.attribute(
        inp,
        strides=(1, 1, stride, stride),
        sliding_window_shapes=(1, 3, patch, patch),
        target=target,
        baselines=0.0,
    )
    attr = attr.abs().mean(dim=1, keepdim=True)
    return _normalize_map(_to_numpy(attr)[0, 0])

def xai_inputxgradient(model, inp, target):
    igx = InputXGradient(model)
    attr = igx.attribute(inp, target=target)
    attr = attr.abs().mean(dim=1, keepdim=True)
    return _normalize_map(_to_numpy(attr)[0, 0])

def xai_guidedbackprop(model, inp, target):
    gbp = GuidedBackprop(model)
    attr = gbp.attribute(inp, target=target)
    attr = attr.abs().mean(dim=1, keepdim=True)
    return _normalize_map(_to_numpy(attr)[0, 0])

def xai_deconvolution(model, inp, target):
    deconv = Deconvolution(model)
    attr = deconv.attribute(inp, target=target)
    attr = attr.abs().mean(dim=1, keepdim=True)
    return _normalize_map(_to_numpy(attr)[0, 0])

def xai_deepliftshap(model, inp, target):
    dlshap = DeepLiftShap(model)
    baselines = torch.zeros_like(inp)
    attr = dlshap.attribute(inp, baselines=baselines, target=target)
    attr = attr.abs().mean(dim=1, keepdim=True)
    return _normalize_map(_to_numpy(attr)[0, 0])

def xai_gradientshap(model, inp, target):
    gshap = GradientShap(model)
    baselines = torch.zeros_like(inp)
    attr = gshap.attribute(inp, baselines=baselines, target=target)
    attr = attr.abs().mean(dim=1, keepdim=True)
    return _normalize_map(_to_numpy(attr)[0, 0])

def xai_smoothgrad(model, inp, target):
    nt = NoiseTunnel(Saliency(model))
    attr = nt.attribute(inp, target=target, nt_type="smoothgrad", nt_samples=25, stdevs=0.2)
    attr = attr.abs().mean(dim=1, keepdim=True)
    return _normalize_map(_to_numpy(attr)[0, 0])

def xai_vargrad(model, inp, target):
    nt = NoiseTunnel(Saliency(model))
    attr = nt.attribute(inp, target=target, nt_type="vargrad", nt_samples=25, stdevs=0.2)
    attr = attr.abs().mean(dim=1, keepdim=True)
    return _normalize_map(_to_numpy(attr)[0, 0])

def xai_feature_ablation(model, inp, target):
    ablator = FeatureAblation(model)
    attr = ablator.attribute(inp, target=target)
    attr = attr.abs().mean(dim=1, keepdim=True)
    return _normalize_map(_to_numpy(attr)[0, 0])

def xai_shapley_sampling(model, inp, target):
    svs = ShapleyValueSampling(model)
    attr = svs.attribute(inp, target=target)
    attr = attr.abs().mean(dim=1, keepdim=True)
    return _normalize_map(_to_numpy(attr)[0, 0])


# ----------------------------
# Visualization
# ----------------------------
def overlay_heatmap(pil_img, heat, alpha=0.5):
    h, w = heat.shape
    img = pil_img.resize((w, h))
    fig = plt.figure(figsize=(4, 4), dpi=200)
    plt.imshow(img)
    plt.imshow(heat, cmap="jet", alpha=alpha)
    plt.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)
    
    
# ----------------------------
# Evaluation Metrics
# ----------------------------

def _topk_mask(a: np.ndarray, fraction: float) -> np.ndarray:
    flat = a.flatten()
    k = max(1, int(len(flat) * fraction))
    idx = np.argpartition(-flat, k - 1)[:k]
    mask = np.zeros_like(flat, dtype=bool)
    mask[idx] = True
    return mask.reshape(a.shape)


def _mask_image(img_t: torch.Tensor, mask: np.ndarray, mode: str = "delete") -> torch.Tensor:
    x = img_t.clone()
    m = torch.from_numpy(mask.astype(np.float32)).to(x.device)[None, None]
    if mode == "delete":
        baseline = 0.0  # simple black-out in normalized space
        x = x * (1 - m) + baseline * m
    else:  # insertion
        baseline_img = torch.zeros_like(x)
        x = baseline_img * (1 - m) + x * m
    return x


def _pred_score(model: torch.nn.Module, x: torch.Tensor, target: int) -> float:
    with torch.no_grad():
        logits = model(x)
        if logits.ndim == 1:
            logits = logits[0][None]
        prob = F.softmax(logits, dim=-1)[0, target].item()
    return float(prob)


def aopc_curve(model: torch.nn.Module, x: torch.Tensor, attr: np.ndarray, target: int, steps: int = 20, mode: str = "delete") -> Tuple[np.ndarray, np.ndarray, float, float]:
    """AOPC for deletion/insertion.
    - deletion: start at original image prob, progressively delete top-k regions
    - insertion: start at baseline image prob (all zeros), progressively insert top-k regions
    Returns: fractions, scores, AOPC, base_prob
    """
    fractions = np.linspace(0, 1, steps + 1)
    scores: List[float] = []

    if mode == "delete":
        base = _pred_score(model, x, target)
        scores.append(base)
        for f in fractions[1:]:
            mask = _topk_mask(attr, f)
            x2 = _mask_image(x, mask, mode="delete")
            scores.append(_pred_score(model, x2, target))
        diffs = (np.array(scores[:1] + scores[1:]) - np.array(scores))  # placeholder, will override below
        diffs = (scores[0] - np.array(scores))
    else:  # insertion
        baseline_img = torch.zeros_like(x)
        base = _pred_score(model, baseline_img, target)
        scores.append(base)
        for f in fractions[1:]:
            mask = _topk_mask(attr, f)
            x2 = _mask_image(x, mask, mode="insert")
            scores.append(_pred_score(model, x2, target))
        diffs = (np.array(scores) - scores[0])

    aopc = float(np.trapz(diffs, fractions))
    return fractions, np.array(scores), aopc, float(base)


def plot_aopc(fractions: np.ndarray, scores: np.ndarray, title: str) -> Image.Image:
    fig = plt.figure(figsize=(4, 4), dpi=200)
    plt.plot(fractions, scores)
    plt.xlabel("Fraction of pixels affected")
    plt.ylabel("Target class probability")
    plt.title(title)
    plt.grid(True, linestyle=":", linewidth=0.5)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def pointing_game(attr: np.ndarray, gt_mask: np.ndarray) -> float:
    y, x = np.unravel_index(np.argmax(attr), attr.shape)
    return float(gt_mask[y, x] > 0)


def roc_auc_vs_mask(attr: np.ndarray, gt_mask: np.ndarray) -> float:
    a = attr.flatten()
    m = (gt_mask.flatten() > 0).astype(np.uint8)
    if len(np.unique(m)) < 2:
        return float("nan")
    return float(roc_auc_score(m, a))

def metric_infidelity(model, inp, target, attr):
    def perturb_fn(inputs):
        noise = torch.randn_like(inputs) * 0.1
        return noise, inputs + noise
    return float(infidelity(model, inp, perturb_fn, target=target, attributions=torch.tensor(attr[None, None])))

def comprehensiveness(model, x, attr, target, fraction=0.2):
    mask = _topk_mask(attr, fraction)
    x_removed = _mask_image(x, mask, mode="delete")
    base = _pred_score(model, x, target)
    new_p = _pred_score(model, x_removed, target)
    return float(base - new_p)

def sufficiency(model, x, attr, target, fraction=0.2):
    mask = _topk_mask(attr, fraction)
    x_only = _mask_image(x, mask, mode="insert")
    base = _pred_score(model, x, target)
    new_p = _pred_score(model, x_only, target)
    return float(new_p - base)

def iou_score(attr, gt_mask, threshold=0.5):
    a_bin = (attr > threshold).astype(np.uint8)
    inter = np.logical_and(a_bin, gt_mask).sum()
    union = np.logical_or(a_bin, gt_mask).sum()
    return float(inter / (union + 1e-8))

def dice_score(attr, gt_mask, threshold=0.5):
    a_bin = (attr > threshold).astype(np.uint8)
    inter = np.logical_and(a_bin, gt_mask).sum()
    return float(2 * inter / (a_bin.sum() + gt_mask.sum() + 1e-8))

def hit_at_k(attr, gt_mask, k=0.01):
    flat = attr.flatten()
    k = max(1, int(len(flat) * k))
    idx = np.argpartition(-flat, k - 1)[:k]
    mask = np.zeros_like(flat, dtype=bool)
    mask[idx] = True
    hit = np.logical_and(mask.reshape(attr.shape), gt_mask > 0).sum()
    return float(hit > 0)



    
# ----------------------------
# (Evaluation + UI remain the same as in your version)
# ----------------------------
# keep run_xai, run_eval, export_report, UI blocks unchanged
# except inside run_xai, before inference, add lazy GPU move:

def run_xai(model_state, data_file, is_nifti, slice_idx, method, target_class, ig_steps, occ_patch, occ_stride, heat_alpha):
    if model_state is None or "kind" not in model_state:
        raise gr.Error("Load or create a model first.")
    if data_file is None:
        raise gr.Error("Please upload an image or NIfTI volume.")

    pil_raw = load_nifti_slice(data_file, slice_idx) if is_nifti else load_image_2d(data_file)
    x, pil_vis = preprocess_both(pil_raw)
    
    print ("Next step is loading nmodel")
    lm = model_state["_obj"]
    model = lm.model

    #  Lazy device move for TorchScript
    if DEVICE == "cuda" and torch.cuda.is_available():
        try:
            model.to(DEVICE)
            x = x.to(DEVICE)
        except Exception as e:
            print("Device move failed:", e)

    preds = predict_topk(model, x, categories=lm.categories, k=5)
    
    print ( "Preds:", preds )
    
    used_target = int(target_class) if (target_class is not None) else -1
    if used_target < 0 and preds:
        used_target = int(preds[0]["index"])

    if method == "Saliency":
        heat = xai_saliency(model, x, used_target)
    elif method == "Integrated Gradients":
        heat = xai_integrated_gradients(model, x, used_target, steps=ig_steps)
    elif method == "Gradient Ã— Input":
        heat = xai_inputxgradient(model, x, used_target)
    elif method == "SmoothGrad":
        heat = xai_smoothgrad(model, x, used_target)
    elif method == "VarGrad":
        heat = xai_vargrad(model, x, used_target)
    elif method == "Guided Backpropagation":
        heat = xai_guidedbackprop(model, x, used_target)
    elif method == "Deconvolution":
        heat = xai_deconvolution(model, x, used_target)
    elif method == "DeepLIFT":
        heat = xai_deeplift(model, x, used_target)
    elif method == "DeepLIFT SHAP":
        heat = xai_deepliftshap(model, x, used_target)
    elif method == "Gradient SHAP":
        heat = xai_gradientshap(model, x, used_target)
    elif method == "Grad-CAM":
        heat = xai_gradcam(lm, x, used_target)
    elif method == "Occlusion":
        heat = xai_occlusion(model, x, used_target, patch=occ_patch, stride=occ_stride)
    elif method == "Feature Ablation":
        heat = xai_feature_ablation(model, x, used_target)
    elif method == "Shapley Value Sampling":
        heat = xai_shapley_sampling(model, x, used_target)
    else:
        raise gr.Error(f"Unknown method: {method}")


    overlay = overlay_heatmap(pil_vis, heat, alpha=heat_alpha)
    return overlay, (pil_vis, x.cpu(), heat), json.dumps({"top5": preds, "used_target": used_target}, indent=2)

def run_eval(model_state: Dict[str, Any], cache_tuple, eval_type: str,
             faith_metric: str, plaus_metric: str,
             manual_target, preds_json_text, gt_file,
             is_nifti_mask: bool, slice_idx: int):

    if cache_tuple is None:
        raise gr.Error("Run an explanation first.")
    if model_state is None:
        raise gr.Error("Load or create a model first.")

    pil_vis, x_cpu, heat = cache_tuple
    x = x_cpu.to(DEVICE)
    lm = model_state["_obj"]
    model = lm.model

    # Determine class to evaluate
    try:
        t = int(manual_target)
    except Exception:
        t = -1
    if t < 0:
        try:
            meta = json.loads(preds_json_text or "{}")
            t = int(meta.get("used_target", 0))
        except Exception:
            t = 0

    # Choose metric based on type
    eval_kind = faith_metric if eval_type == "Faithfulness" else plaus_metric

    fig = None
    results = {}

    # --- Faithfulness metrics ---
    if eval_kind in ("Deletion AOPC", "Insertion AOPC"):
        mode = "delete" if eval_kind.startswith("Deletion") else "insert"
        fracs, scores, aopc, base_p = aopc_curve(model, x, heat, t, steps=20, mode=mode)
        title = f"{eval_kind} (AOPC={aopc:.3f}, base p={base_p:.3f})"
        fig = plot_aopc(fracs, scores, title)
        results["AOPC"] = aopc
        results["base_p"] = base_p

    elif eval_kind == "Infidelity":
        results["Infidelity"] = metric_infidelity(model, x, t, heat)

    elif eval_kind == "Comprehensiveness":
        results["Comprehensiveness"] = comprehensiveness(model, x, heat, t, fraction=0.2)

    elif eval_kind == "Sufficiency":
        results["Sufficiency"] = sufficiency(model, x, heat, t, fraction=0.2)

    # --- Plausibility metrics ---
    elif eval_kind in ("Pointing Game", "Mask ROC-AUC", "IOU", "Dice", "Hit@K"):
        if gt_file is None:
            raise gr.Error("Upload a ground-truth mask for this metric.")
        gt_pil = load_nifti_slice(gt_file, slice_idx) if is_nifti_mask else Image.open(gt_file.name).convert("L")
        gt = np.array(gt_pil.resize((heat.shape[1], heat.shape[0])))
        gt = (gt > 0).astype(np.uint8)

        if eval_kind == "Pointing Game":
            results["PointingGame"] = float(pointing_game(heat, gt))
        elif eval_kind == "Mask ROC-AUC":
            results["MaskROC_AUC"] = float(roc_auc_vs_mask(heat, gt))
        elif eval_kind == "IOU":
            results["IOU"] = iou_score(heat, gt)
        elif eval_kind == "Dice":
            results["Dice"] = dice_score(heat, gt)
        elif eval_kind == "Hit@K":
            results["Hit@K"] = hit_at_k(heat, gt, k=0.01)

    else:
        raise gr.Error(f"Unknown evaluation metric: {eval_kind}")

    report = json.dumps({"eval": eval_kind, "results": results}, indent=2)
    return fig, report



def export_report(report_text: str):
    if not report_text:
        raise gr.Error("No report to export. Run an evaluation first.")
    path = "xai_eval_report.json"
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_text)
    return path


# ----------------------------
# Gradio UI
# ----------------------------
with gr.Blocks(title="XAI + Evaluation (Medical imaging ready)") as demo:
    gr.Markdown("""
    # XAI + Evaluation App
    Load a model, generate attributions, and evaluate faithfulness.
    """)

    with gr.Tab("1) Model"):
        model_state = gr.State()
        with gr.Row():
            with gr.Column():

                model_src = gr.Radio(
                     ["Demo DenseNet121 (ImageNet)", "Torchvision Model", "MONAI Model", "Transformer Model", "Upload TorchScript .pt", "Custom Architecture"],
                      label="Model source",
                      value="Demo DenseNet121 (ImageNet)"
                )


                torchvision_model_name = gr.Dropdown(choices=available_models, value="densenet121", visible=False, label="Torchvision model")
                monai_model_name = gr.Dropdown(choices=monai_models, value="densenet121", visible=False, label="MONAI model")
                transformer_model_name = gr.Textbox(value="google/vit-base-patch16-224", visible=False, label="Transformer model name (HuggingFace Hub)")



                uploaded_model = gr.File(label="Upload TorchScript .pt", file_types=[".pt"], visible=False)
                arch_py_file = gr.File(label="Upload Python architecture (.py)", file_types=[".py"], visible=False)
                safetensor_file = gr.File(label="Optional: SafeTensors weights", file_types=[".safetensors"], visible=False)
                classmap_file = gr.File(label="Optional: class map JSON", file_types=[".json"], visible=True)
                load_btn = gr.Button("Load model", variant="primary")
                model_info  = gr.Textbox(label="Model info (JSON)", lines=10)
                gradcam_enabled = gr.State()

        def toggle_model_src(src):
                return (
                         gr.update(visible=(src == "Upload TorchScript .pt")),
                         gr.update(visible=(src == "Torchvision Model")),
                         gr.update(visible=(src == "MONAI Model")),
                         gr.update(visible=(src == "Transformer Model")),
                         gr.update(visible=(src == "Custom Architecture")),
                         gr.update(visible=(src == "Custom Architecture"))
                         )

        
        model_src.change(
                 toggle_model_src,
                 inputs=model_src,
                 outputs=[uploaded_model,torchvision_model_name,monai_model_name,transformer_model_name,arch_py_file,safetensor_file])


        def do_load_model(src, f, cmap, tv_name, monai_name, transformer_name, arch_py, st_file):
            if src.startswith("Demo"):
                lm = load_demo_model()
            elif src == "Torchvision Model":
                lm = load_torchvision_model(tv_name)

            elif src == "MONAI Model":
                lm = load_monai_model(monai_name)
            elif src == "Transformer Model":
                lm = load_transformer_model(transformer_name)

            elif src == "Upload TorchScript .pt":
                if f is None:
                    raise gr.Error("Please upload a TorchScript .pt model.")
                lm = load_torchscript(f.name)
            elif src == "Custom Architecture":
                if arch_py is None:
                    raise gr.Error("Upload a Python file for the architecture.")
                lm = load_custom_architecture(arch_py.name, st_file.name if st_file else None)
            else:
                raise gr.Error(f"Unknown model source: {src}")
            if cmap is not None:
                try:
                    with open(cmap.name, "r", encoding="utf-8") as jf:
                        labels = json.load(jf)
                    if isinstance(labels, list) and all(isinstance(s, str) for s in labels):
                        lm.categories = labels
                except Exception as e:
                    raise gr.Error(f"Failed to read class map JSON: {e}")
            st = {"kind": lm.kind, "device": DEVICE, "has_gradcam": lm.target_layer is not None, "_obj": lm}
            info = {
                "kind": lm.kind,
                "device": DEVICE,
                "has_gradcam": lm.target_layer is not None,
                "num_classes": None if lm.categories is None else len(lm.categories),
            }
            return st, info, lm.target_layer is not None

        load_btn.click(
            do_load_model,
            inputs=[model_src, uploaded_model, classmap_file, torchvision_model_name, monai_model_name, transformer_model_name, arch_py_file, safetensor_file],
            outputs=[model_state, model_info, gradcam_enabled]
        )


    with gr.Tab("2) Explain"):
        with gr.Row():
            with gr.Column():
                data_file = gr.File(label="Input image or NIfTI volume", file_types=[".png", ".jpg", ".jpeg", ".nii", ".nii.gz"])
                is_nifti = gr.Checkbox(False, label="This is a NIfTI volume (.nii/.nii.gz)")
                slice_idx = gr.Slider(0, 500, step=1, value=0, label="Slice index (for NIfTI)")

                #method = gr.Dropdown(["Saliency", "Integrated Gradients", "DeepLIFT", "Grad-CAM", "Occlusion"], value="Grad-CAM", label="XAI method")
                method = gr.Dropdown(
                    [
                            # Gradient-based
                            "Saliency",
                            "Integrated Gradients",
                            "Gradient Ã— Input",
                            "SmoothGrad",
                            "VarGrad",
                            "Guided Backpropagation",
                            "Deconvolution",
                            "DeepLIFT",
                            "DeepLIFT SHAP",
                            "Gradient SHAP",

                            # CAM-based
                            "Grad-CAM",

                            # Perturbation-based
                            "Occlusion",
                            "Feature Ablation",
                            "Shapley Value Sampling",
                        ],
                        value="Saliency",
                        label="XAI method"
                    )


                target_class = gr.Number(value=-1, label="Target class index (set -1 to use Top-1)")
                ig_steps = gr.Slider(8, 200, value=50, step=1, label="IG steps")
                occ_patch = gr.Slider(5, 51, value=15, step=2, label="Occlusion patch size")
                occ_stride = gr.Slider(2, 32, value=8, step=1, label="Occlusion stride")
                heat_alpha = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Heatmap alpha")
                go_btn = gr.Button("Run XAI", variant="primary")
            with gr.Column():
                out_overlay = gr.Image(label="Attribution overlay", type="pil")
                preds_json  = gr.Textbox(label="Predictions (Top-5)", lines=10)
                cache_tuple = gr.State()
        
        # ðŸ”¹ Dynamically update available XAI methods
        def update_methods(has_gradcam):
            methods = [
                "Saliency", "Integrated Gradients", "Gradient Ã— Input", "SmoothGrad", "VarGrad",
                "Guided Backpropagation", "Deconvolution",
                "DeepLIFT", "DeepLIFT SHAP", "Gradient SHAP",
                "Occlusion", "Feature Ablation", "Shapley Value Sampling"
            ]
            if has_gradcam:
                methods.insert(10, "Grad-CAM")  # add Grad-CAM at right position
            return gr.update(choices=methods, value=methods[0])


        gradcam_enabled.change(update_methods, inputs=gradcam_enabled, outputs=method)

        go_btn.click(
            run_xai,
            inputs=[model_state, data_file, is_nifti, slice_idx, method, target_class, ig_steps, occ_patch, occ_stride, heat_alpha],
            outputs=[out_overlay, cache_tuple, preds_json]
        )



    with gr.Tab("3) Evaluate"):
        with gr.Row():
            with gr.Column():
                eval_type = gr.Radio(
                    ["Faithfulness", "Plausibility"],
                    value="Faithfulness",
                    label="Evaluation type"
                )

                faith_metric = gr.Dropdown(
                    ["Deletion AOPC", "Insertion AOPC", "Infidelity", "Comprehensiveness", "Sufficiency"],
                    value="Deletion AOPC",
                    label="Faithfulness metrics",
                    visible=True
                )

                plaus_metric = gr.Dropdown(
                    ["Pointing Game", "Mask ROC-AUC", "IOU", "Dice", "Hit@K"],
                    value="Pointing Game",
                    label="Plausibility metrics",
                    visible=False
                )

                manual_target = gr.Number(value=-1, label="Target class (set -1 to use Top-1 from Explain)")
                gt_file = gr.File(label="Ground-truth mask (image or NIfTI)")
                is_nifti_mask = gr.Checkbox(False, label="Mask is NIfTI (.nii/.nii.gz)")
                slice_idx2 = gr.Slider(0, 500, step=1, value=0, label="Slice index (if mask is NIfTI)")
                eval_btn = gr.Button("Run evaluation", variant="primary")

            with gr.Column():
                eval_plot = gr.Image(label="Evaluation plot (if applicable)")
                report_text = gr.Textbox(label="JSON report", lines=10)
                save_btn = gr.Button("Export report JSON")
                save_path = gr.File(label="Download report")

        # ðŸ”¹ Toggle which dropdown is visible
        def toggle_metrics(eval_type):
            if eval_type == "Faithfulness":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)

        eval_type.change(
            toggle_metrics,
            inputs=eval_type,
            outputs=[faith_metric, plaus_metric]
        )

        eval_btn.click(
            run_eval,
            inputs=[model_state, cache_tuple, eval_type, faith_metric, plaus_metric,
                    manual_target, preds_json, gt_file, is_nifti_mask, slice_idx2],
            outputs=[eval_plot, report_text]
        )

        save_btn.click(export_report, inputs=[report_text], outputs=[save_path])


    with gr.Tab("4) Notes & TODO"):
        gr.Markdown(
            """
            **Planned extensions**
            - Full 3D support with MONAI networks and CAM/Grad-CAM++ for volumes
            - Concept-based explanations (TCAV) with concept gallery & CAV training
            - Sanity checks: parameter/input randomization tests
            - Dataset-level batch evaluation + CSV export
            - Segmentation/regression model support
            - DICOM series browser & ROI overlays
            """
        )
'''
if __name__ == "__main__":
    #demo.launch()
    demo.launch(server_name="0.0.0.0", server_port=7862, share = True )
'''    

if __name__ == "__main__":
    import sys, os, logging

    # Fix: PyInstaller --noconsole makes sys.stderr = None
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")

    # Basic logging setup so uvicorn doesnâ€™t crash
    logging.basicConfig(level=logging.INFO)

    # ðŸš€ Launch Gradio safely (quiet=True disables uvicorn logging config)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        quiet=True,            # suppress uvicorn logging setup
        inbrowser=False        # donâ€™t auto-open browser
    ) 
