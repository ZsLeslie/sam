import cv2
import torch
import numpy as np

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

model = build_sam2(
    model_cfg,
    checkpoint,
    device="cuda",
)

predictor = SAM2ImagePredictor(model)

image_bgr = cv2.imread("/home/nvidia/colmap_data/无人机/02/images_undistorted/airPont_1zdGWD4hGO_20_1787735951316_12.300000190734863.jpg")

if image_bgr is None:
    raise RuntimeError("Failed to load image")

image = cv2.cvtColor(
    image_bgr,
    cv2.COLOR_BGR2RGB,
)

print("image:", image.shape)

# 给一个前景点
h, w = image.shape[:2]

input_point = np.array([
    [w // 2, h // 2]
], dtype=np.float32)

input_label = np.array([1], dtype=np.int32)

with torch.inference_mode():

    predictor.set_image(image)

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

best = np.argmax(scores)

mask = masks[best] > 0

print("scores:", scores)
print("best:", best)
print("mask pixels:", mask.sum())

out = image_bgr.copy()

out[mask] = (
    out[mask] * 0.5 +
    np.array([0, 0, 255]) * 0.5
).astype(np.uint8)

cv2.circle(
    out,
    tuple(input_point[0].astype(int)),
    8,
    (0, 255, 0),
    -1,
)

cv2.imwrite(
    "./result/result.jpg",
    out,
)
