#!/usr/bin/env python3

import json
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from PIL import Image as PILImage
    from PIL import ImageDraw, ImageFont
except ImportError:
    PILImage = None
    ImageDraw = None
    ImageFont = None

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ======================================================================
# 配置
# ======================================================================

IMAGE_DIR = Path(
    "/home/nvidia/LoFTR/test_images/"
)

OUTPUT_DIR = Path(
    "/home/nvidia/LoFTR/"
)

MASK_DIR = OUTPUT_DIR / "masks"
VIS_DIR = OUTPUT_DIR / "visualizations"
PROMPT_DIR = OUTPUT_DIR / "prompts"


# ----------------------------------------------------------------------
# SAM2 模型
# ----------------------------------------------------------------------

# Tiny
#
# CHECKPOINT = Path(
#     "/home/nvidia/sam2/checkpoints/sam2.1_hiera_tiny.pt"
# )
#
# MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"


# Base+
CHECKPOINT = Path(
    "/home/nvidia/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
)

MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"


# ----------------------------------------------------------------------
# GUI 显示参数
#
# 注意：
# 这些参数只影响 OpenCV 窗口显示尺寸，
# 不会修改送入 SAM2 的原始图像。
# ----------------------------------------------------------------------

DISPLAY_MAX_WIDTH = 1500
DISPLAY_MAX_HEIGHT = 950

MASK_ALPHA = 0.45

WINDOW_NAME = "SAM2 Interactive Blade Segmentation"


# ----------------------------------------------------------------------
# “没有天空”按钮
# ----------------------------------------------------------------------

NO_SKY_BUTTON_TEXT = "没有天空"
NO_SKY_BUTTON_FALLBACK_TEXT = "NO SKY"
NO_SKY_BUTTON_WIDTH = 220
NO_SKY_BUTTON_HEIGHT = 64
NO_SKY_BUTTON_MARGIN = 20

# 如果系统中存在中文字体，会优先显示“没有天空”。
# 如果找不到，则自动回退为“NO SKY”。
CHINESE_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]


IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ======================================================================
# SAM2
# ======================================================================

def build_predictor():

    print("=" * 80)
    print("Loading SAM2")
    print("=" * 80)

    print("Checkpoint :", CHECKPOINT)
    print("Config     :", MODEL_CFG)
    print("Torch      :", torch.__version__)
    print("CUDA       :", torch.version.cuda)
    print("GPU        :", torch.cuda.get_device_name(0))

    model = build_sam2(
        MODEL_CFG,
        str(CHECKPOINT),
        device="cuda",
    )

    predictor = SAM2ImagePredictor(model)

    print("SAM2 loaded.")
    print()

    return predictor


# ======================================================================
# GUI
# ======================================================================

class SAM2Interactive:

    def __init__(self, predictor, image_paths):

        self.predictor = predictor
        self.image_paths = image_paths

        self.image_index = 0

        # 原始图像
        self.image_bgr = None
        self.image_rgb = None

        # 原始图像尺寸
        self.h = 0
        self.w = 0

        # GUI 显示缩放比例
        self.display_scale = 1.0

        # SAM2 prompt
        self.points = []
        self.labels = []

        # 当前最终 mask
        #
        # bool
        # shape = original_h x original_w
        #
        self.mask = None

        self.score = None

        # SAM2 原始低分辨率 mask logits
        #
        # 注意：
        # 这个不能 resize，
        # 它用于下一轮 SAM2 refinement。
        #
        self.mask_logits = None

        self.current_path = None

        # “没有天空”按钮在 GUI 显示坐标系中的矩形区域
        # (x1, y1, x2, y2)
        self.no_sky_button_rect = None

        # 缓存按钮图像，避免每一帧重复生成中文文字
        self.no_sky_button_image = self._create_no_sky_button_image()

        self.load_image(0)

    # ==================================================================
    # 载入图像
    # ==================================================================

    def load_image(self, index):

        index = max(
            0,
            min(
                index,
                len(self.image_paths) - 1
            )
        )

        self.image_index = index

        self.current_path = self.image_paths[index]

        print()
        print("=" * 80)

        print(
            f"[{index + 1}/{len(self.image_paths)}] "
            f"{self.current_path.name}"
        )

        print("=" * 80)

        # --------------------------------------------------------------
        # 直接读取原始图像
        #
        # 这里没有进行 resize
        # --------------------------------------------------------------

        image = cv2.imread(
            str(self.current_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:

            raise RuntimeError(
                f"Failed to read image: "
                f"{self.current_path}"
            )

        self.image_bgr = image

        self.image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        # --------------------------------------------------------------
        # 保存原始分辨率
        # --------------------------------------------------------------

        self.h, self.w = image.shape[:2]

        print(
            f"Original image size : "
            f"{self.w} x {self.h}"
        )

        # --------------------------------------------------------------
        # 计算 GUI 显示比例
        #
        # 注意：
        # 这里只是显示用。
        #
        # self.image_bgr
        # self.image_rgb
        #
        # 仍然保持原始分辨率。
        # --------------------------------------------------------------

        scale_w = (
            DISPLAY_MAX_WIDTH
            /
            float(self.w)
        )

        scale_h = (
            DISPLAY_MAX_HEIGHT
            /
            float(self.h)
        )

        self.display_scale = min(
            1.0,
            scale_w,
            scale_h,
        )

        print(
            f"GUI scale           : "
            f"{self.display_scale:.4f}"
        )

        # --------------------------------------------------------------
        # 清空当前 prompt
        # --------------------------------------------------------------

        self.points = []
        self.labels = []

        self.mask = None
        self.score = None
        self.mask_logits = None

        # --------------------------------------------------------------
        # SAM2 image encoder
        #
        # 关键：
        #
        # 送进去的是原始 image_rgb，
        # 不是 GUI 缩小后的图像。
        #
        # SAM2 内部会自行执行模型需要的预处理。
        # --------------------------------------------------------------

        print("Encoding image...")

        with torch.inference_mode():

            self.predictor.set_image(
                self.image_rgb
            )

        print("Image encoded.")

    # ==================================================================
    # 保证 mask 与原始图像尺寸完全一致
    # ==================================================================

    def ensure_mask_original_size(self, mask):
        """
        保证最终 mask 为：

            shape = (self.h, self.w)
            dtype = bool

        如果 SAM2 已经输出原始尺寸，
        则不会执行 resize。

        如果尺寸不同，
        使用 INTER_NEAREST 恢复到原始尺寸，
        避免产生灰度过渡。
        """

        if mask is None:
            return None

        mask = np.asarray(mask)

        # --------------------------------------------------------------
        # 去掉多余的 singleton dimension
        #
        # 例如：
        #
        # 1 x H x W
        #
        # ->
        #
        # H x W
        # --------------------------------------------------------------

        mask = np.squeeze(mask)

        if mask.ndim != 2:

            raise RuntimeError(
                f"Unexpected mask shape: "
                f"{mask.shape}"
            )

        mask_h, mask_w = mask.shape

        # --------------------------------------------------------------
        # 已经是原始分辨率
        # --------------------------------------------------------------

        if (
            mask_h == self.h
            and
            mask_w == self.w
        ):

            return mask.astype(bool)

        # --------------------------------------------------------------
        # 尺寸不同则恢复
        # --------------------------------------------------------------

        print(
            f"Resize mask: "
            f"{mask_w} x {mask_h}"
            f" -> "
            f"{self.w} x {self.h}"
        )

        mask_u8 = (
            mask.astype(np.uint8)
            *
            255
        )

        mask_u8 = cv2.resize(
            mask_u8,
            (self.w, self.h),
            interpolation=cv2.INTER_NEAREST,
        )

        mask_bool = (
            mask_u8 > 127
        )

        return mask_bool

    # ==================================================================
    # OpenCV 显示坐标 -> 原始图像坐标
    # ==================================================================

    def display_to_original(self, x, y):

        ox = int(
            round(
                x
                /
                self.display_scale
            )
        )

        oy = int(
            round(
                y
                /
                self.display_scale
            )
        )

        ox = np.clip(
            ox,
            0,
            self.w - 1,
        )

        oy = np.clip(
            oy,
            0,
            self.h - 1,
        )

        return int(ox), int(oy)

    # ==================================================================
    # SAM2 推理
    # ==================================================================

    def predict(self, use_previous_mask=True):

        if len(self.points) == 0:

            self.mask = None
            self.score = None
            self.mask_logits = None

            return

        point_coords = np.asarray(
            self.points,
            dtype=np.float32,
        )

        point_labels = np.asarray(
            self.labels,
            dtype=np.int32,
        )

        # --------------------------------------------------------------
        # 第一轮：
        #
        # 如果只有 1 个点，
        # 让 SAM2 输出多个候选 mask。
        #
        # 后续 refinement：
        #
        # 输出单个 mask。
        # --------------------------------------------------------------

        if self.mask_logits is None:

            multimask = (
                len(self.points) == 1
            )

            mask_input = None

        else:

            multimask = False

            if use_previous_mask:

                # ------------------------------------------------------
                # 注意：
                #
                # SAM2 logits 必须保持其内部尺寸，
                # 不要转换到 5280x3956。
                # ------------------------------------------------------

                mask_input = (
                    self.mask_logits[
                        None,
                        :,
                        :
                    ]
                )

            else:

                mask_input = None

        # --------------------------------------------------------------
        # SAM2 prediction
        # --------------------------------------------------------------

        with torch.inference_mode():

            masks, scores, logits = (
                self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    mask_input=mask_input,
                    multimask_output=multimask,
                )
            )

        # --------------------------------------------------------------
        # 选择最高 score
        # --------------------------------------------------------------

        best_idx = int(
            np.argmax(scores)
        )

        selected_mask = (
            masks[best_idx] > 0
        )

        # --------------------------------------------------------------
        # 强制 mask 与原图分辨率一致
        # --------------------------------------------------------------

        self.mask = (
            self.ensure_mask_original_size(
                selected_mask
            )
        )

        self.score = float(
            scores[best_idx]
        )

        # --------------------------------------------------------------
        # 保留 SAM2 原始低分辨率 logits
        #
        # 用于下一次交互 refinement。
        #
        # 不 resize。
        # --------------------------------------------------------------

        self.mask_logits = (
            logits[best_idx]
        )

        print()
        print("Prompts:")

        for i, (point, label) in enumerate(
            zip(
                self.points,
                self.labels
            )
        ):

            prompt_type = (
                "FG"
                if label == 1
                else "BG"
            )

            print(
                f"  {i + 1:02d}: "
                f"{prompt_type} "
                f"({point[0]}, {point[1]})"
            )

        print(
            f"Score      : "
            f"{self.score:.6f}"
        )

        print(
            f"Mask pixels: "
            f"{int(self.mask.sum())}"
        )

        print(
            f"Mask ratio : "
            f"{self.mask.mean() * 100:.2f}%"
        )

        print(
            f"Mask size  : "
            f"{self.mask.shape[1]} x "
            f"{self.mask.shape[0]}"
        )

    # ==================================================================
    # 从头重新预测
    #
    # Undo 后不能继续使用旧 logits，
    # 因为其中包含了被删除 prompt 的信息。
    # ==================================================================

    def predict_from_scratch(self):

        self.mask = None
        self.score = None
        self.mask_logits = None

        if not self.points:
            return

        point_coords = np.asarray(
            self.points,
            dtype=np.float32,
        )

        point_labels = np.asarray(
            self.labels,
            dtype=np.int32,
        )

        multimask = (
            len(self.points) == 1
        )

        with torch.inference_mode():

            masks, scores, logits = (
                self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=multimask,
                )
            )

        best_idx = int(
            np.argmax(scores)
        )

        selected_mask = (
            masks[best_idx] > 0
        )

        # --------------------------------------------------------------
        # 强制恢复到原图尺寸
        # --------------------------------------------------------------

        self.mask = (
            self.ensure_mask_original_size(
                selected_mask
            )
        )

        self.score = float(
            scores[best_idx]
        )

        # --------------------------------------------------------------
        # logits 保持 SAM2 原始尺寸
        # --------------------------------------------------------------

        self.mask_logits = (
            logits[best_idx]
        )

    # ==================================================================
    # “没有天空”按钮
    # ==================================================================

    def _find_chinese_font(self):
        """查找可用于 OpenCV GUI 中文按钮的系统字体。"""

        for font_path in CHINESE_FONT_CANDIDATES:
            p = Path(font_path)

            if p.exists():
                return p

        return None

    def _create_no_sky_button_image(self):
        """
        创建按钮图像并缓存。

        优先使用 Pillow + 系统中文字体显示：
            没有天空

        如果没有 Pillow 或找不到中文字体，则回退为：
            NO SKY
        """

        button = np.full(
            (
                NO_SKY_BUTTON_HEIGHT,
                NO_SKY_BUTTON_WIDTH,
                3,
            ),
            245,
            dtype=np.uint8,
        )

        # 浅黄色按钮背景
        button[:, :] = (
            80,
            210,
            255,
        )

        # 黑色边框
        cv2.rectangle(
            button,
            (0, 0),
            (
                NO_SKY_BUTTON_WIDTH - 1,
                NO_SKY_BUTTON_HEIGHT - 1,
            ),
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        font_path = self._find_chinese_font()

        if (
            PILImage is not None
            and ImageDraw is not None
            and ImageFont is not None
            and font_path is not None
        ):

            # OpenCV BGR -> Pillow RGB
            rgb = cv2.cvtColor(
                button,
                cv2.COLOR_BGR2RGB,
            )

            pil_img = PILImage.fromarray(rgb)
            draw = ImageDraw.Draw(pil_img)

            font = ImageFont.truetype(
                str(font_path),
                32,
            )

            bbox = draw.textbbox(
                (0, 0),
                NO_SKY_BUTTON_TEXT,
                font=font,
            )

            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            tx = (
                NO_SKY_BUTTON_WIDTH
                - text_w
            ) // 2

            ty = (
                NO_SKY_BUTTON_HEIGHT
                - text_h
            ) // 2 - bbox[1]

            draw.text(
                (tx, ty),
                NO_SKY_BUTTON_TEXT,
                font=font,
                fill=(0, 0, 0),
            )

            button = cv2.cvtColor(
                np.asarray(pil_img),
                cv2.COLOR_RGB2BGR,
            )

        else:

            # 无中文字体时使用 ASCII 回退文字
            text = NO_SKY_BUTTON_FALLBACK_TEXT

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.9
            thickness = 2

            (text_w, text_h), _ = cv2.getTextSize(
                text,
                font,
                font_scale,
                thickness,
            )

            tx = (
                NO_SKY_BUTTON_WIDTH
                - text_w
            ) // 2

            ty = (
                NO_SKY_BUTTON_HEIGHT
                + text_h
            ) // 2

            cv2.putText(
                button,
                text,
                (tx, ty),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )

        return button

    def _draw_no_sky_button(self, vis):
        """
        在已经缩放到 GUI 尺寸的图像右上角绘制按钮。

        同时记录按钮区域，用于鼠标点击检测。
        """

        display_h, display_w = vis.shape[:2]

        button_w = min(
            NO_SKY_BUTTON_WIDTH,
            max(1, display_w - 2 * NO_SKY_BUTTON_MARGIN),
        )

        button_h = min(
            NO_SKY_BUTTON_HEIGHT,
            max(1, display_h - 2 * NO_SKY_BUTTON_MARGIN),
        )

        x2 = display_w - NO_SKY_BUTTON_MARGIN
        x1 = x2 - button_w

        y1 = NO_SKY_BUTTON_MARGIN
        y2 = y1 + button_h

        # 极小窗口时进行保护
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(display_w, x2)
        y2 = min(display_h, y2)

        if x2 <= x1 or y2 <= y1:
            self.no_sky_button_rect = None
            return vis

        button = self.no_sky_button_image

        if (
            button.shape[1] != (x2 - x1)
            or button.shape[0] != (y2 - y1)
        ):
            button = cv2.resize(
                button,
                (x2 - x1, y2 - y1),
                interpolation=cv2.INTER_AREA,
            )

        vis[y1:y2, x1:x2] = button

        self.no_sky_button_rect = (
            x1,
            y1,
            x2,
            y2,
        )

        return vis

    def _is_no_sky_button_clicked(self, x, y):
        """判断 GUI 显示坐标是否位于“没有天空”按钮内。"""

        if self.no_sky_button_rect is None:
            return False

        x1, y1, x2, y2 = self.no_sky_button_rect

        return (
            x1 <= x < x2
            and
            y1 <= y < y2
        )

    def save_no_sky_mask(self):
        """
        当前图像没有天空：

        直接保存全白 PNG mask。

            255 = 全图保留

        输出尺寸严格等于原始输入图像：

            (self.h, self.w)
        """

        stem = self.current_path.stem

        mask_path = (
            MASK_DIR
            /
            f"{stem}.png"
        )

        # 全白，严格 uint8 255
        mask_u8 = np.full(
            (
                self.h,
                self.w,
            ),
            255,
            dtype=np.uint8,
        )

        success = cv2.imwrite(
            str(mask_path),
            mask_u8,
            [
                cv2.IMWRITE_PNG_COMPRESSION,
                3,
            ],
        )

        if not success:
            raise RuntimeError(
                f"Failed to save no-sky mask:\n"
                f"{mask_path}"
            )

        # 保存后再次读取，确认尺寸与像素值
        check_mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if check_mask is None:
            raise RuntimeError(
                f"Failed to verify no-sky mask:\n"
                f"{mask_path}"
            )

        if check_mask.shape != (
            self.h,
            self.w,
        ):
            raise RuntimeError(
                "Saved no-sky mask size mismatch:\n"
                f"Saved : "
                f"{check_mask.shape[1]} x "
                f"{check_mask.shape[0]}\n"
                f"Image : "
                f"{self.w} x {self.h}"
            )

        unique_values = np.unique(
            check_mask
        )

        if not (
            unique_values.size == 1
            and
            unique_values[0] == 255
        ):
            raise RuntimeError(
                "Saved no-sky mask is not all white:\n"
                f"{unique_values}"
            )

        print()
        print("=" * 80)
        print("No sky -> saved all-white PNG mask")
        print("=" * 80)
        print(
            "Image size :",
            f"{self.w} x {self.h}"
        )
        print(
            "Mask size  :",
            f"{check_mask.shape[1]} x "
            f"{check_mask.shape[0]}"
        )
        print(
            "Mask values:",
            unique_values
        )
        print(
            "Mask       :",
            mask_path
        )

    # ==================================================================
    # 鼠标
    # ==================================================================

    def mouse_callback(
        self,
        event,
        x,
        y,
        flags,
        param,
    ):

        # --------------------------------------------------------------
        # “没有天空”按钮
        #
        # 必须在坐标转换之前检测，因为按钮位置属于 GUI 显示坐标系。
        # --------------------------------------------------------------

        if (
            event == cv2.EVENT_LBUTTONDOWN
            and
            self._is_no_sky_button_clicked(x, y)
        ):

            self.save_no_sky_mask()
            return

        # --------------------------------------------------------------
        # GUI 坐标转换回原图坐标
        # --------------------------------------------------------------

        ox, oy = (
            self.display_to_original(
                x,
                y,
            )
        )

        # --------------------------------------------------------------
        # 左键 = 前景
        # --------------------------------------------------------------

        if event == cv2.EVENT_LBUTTONDOWN:

            self.points.append(
                [ox, oy]
            )

            self.labels.append(1)

            print()

            print(
                f"[FG] ({ox}, {oy})"
            )

            self.predict(
                use_previous_mask=True
            )

        # --------------------------------------------------------------
        # 右键 = 背景
        # --------------------------------------------------------------

        elif event == cv2.EVENT_RBUTTONDOWN:

            self.points.append(
                [ox, oy]
            )

            self.labels.append(0)

            print()

            print(
                f"[BG] ({ox}, {oy})"
            )

            self.predict(
                use_previous_mask=True
            )

    # ==================================================================
    # 添加中心点
    # ==================================================================

    def add_center_point(self):

        x = self.w // 2
        y = self.h // 2

        self.points.append(
            [x, y]
        )

        self.labels.append(1)

        print()

        print(
            f"[CENTER FG] ({x}, {y})"
        )

        self.predict(
            use_previous_mask=True
        )

    # ==================================================================
    # Undo
    # ==================================================================

    def undo(self):

        if not self.points:

            print(
                "Nothing to undo."
            )

            return

        p = self.points.pop()

        label = self.labels.pop()

        print()

        print(
            "Undo:",
            "FG"
            if label == 1
            else "BG",
            p,
        )

        # --------------------------------------------------------------
        # 旧 logits 已经失效
        # --------------------------------------------------------------

        self.predict_from_scratch()

    # ==================================================================
    # Reset
    # ==================================================================

    def reset(self):

        self.points = []
        self.labels = []

        self.mask = None
        self.score = None
        self.mask_logits = None

        print()
        print("Prompts cleared.")

    # ==================================================================
    # 可视化
    # ==================================================================

    def render(self):

        # --------------------------------------------------------------
        # 始终从原始图像创建可视化
        # --------------------------------------------------------------

        vis = self.image_bgr.copy()

        # --------------------------------------------------------------
        # Mask overlay
        # --------------------------------------------------------------

        if self.mask is not None:

            # 安全检查
            if self.mask.shape != (
                self.h,
                self.w,
            ):

                self.mask = (
                    self.ensure_mask_original_size(
                        self.mask
                    )
                )

            overlay = vis.copy()

            # BGR 红色
            overlay[self.mask] = (
                0,
                0,
                255,
            )

            vis = cv2.addWeighted(
                overlay,
                MASK_ALPHA,
                vis,
                1.0 - MASK_ALPHA,
                0,
            )

        # --------------------------------------------------------------
        # Prompt 点
        # --------------------------------------------------------------

        for i, (point, label) in enumerate(
            zip(
                self.points,
                self.labels
            )
        ):

            x, y = point

            if label == 1:

                # 前景绿色
                color = (
                    0,
                    255,
                    0,
                )

            else:

                # 背景蓝色
                color = (
                    255,
                    0,
                    0,
                )

            cv2.circle(
                vis,
                (x, y),
                20,
                (255, 255, 255),
                5,
                cv2.LINE_AA,
            )

            cv2.circle(
                vis,
                (x, y),
                14,
                color,
                -1,
                cv2.LINE_AA,
            )

            cv2.putText(
                vis,
                str(i + 1),
                (
                    x + 25,
                    y - 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                3,
                cv2.LINE_AA,
            )

        # --------------------------------------------------------------
        # 状态文字
        # --------------------------------------------------------------

        line1 = (
            f"{self.image_index + 1}/"
            f"{len(self.image_paths)} "
            f"{self.current_path.name}"
        )

        if self.score is not None:

            line2 = (
                f"score={self.score:.4f}  "
                f"FG={self.labels.count(1)}  "
                f"BG={self.labels.count(0)}"
            )

        else:

            line2 = (
                "No mask - "
                "Left click to add foreground"
            )

        # --------------------------------------------------------------
        # 黑色描边 + 白色文字
        # --------------------------------------------------------------

        for text, pos in [
            (
                line1,
                (30, 55)
            ),
            (
                line2,
                (30, 105)
            ),
        ]:

            cv2.putText(
                vis,
                text,
                pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 0),
                5,
                cv2.LINE_AA,
            )

            cv2.putText(
                vis,
                text,
                pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # --------------------------------------------------------------
        # 最后才缩放 GUI 显示
        #
        # 这一步不会影响：
        #
        # self.image_bgr
        # self.image_rgb
        # self.mask
        #
        # 也不会影响保存结果。
        # --------------------------------------------------------------

        if self.display_scale < 1.0:

            dw = int(
                round(
                    self.w
                    *
                    self.display_scale
                )
            )

            dh = int(
                round(
                    self.h
                    *
                    self.display_scale
                )
            )

            vis = cv2.resize(
                vis,
                (dw, dh),
                interpolation=cv2.INTER_AREA,
            )

        # --------------------------------------------------------------
        # GUI 按钮必须在最终显示尺寸上绘制。
        # --------------------------------------------------------------

        vis = self._draw_no_sky_button(
            vis
        )

        return vis

    # ==================================================================
    # 保存
    # ==================================================================

    def save(self):

        if self.mask is None:

            print()

            print(
                "No mask to save."
            )

            return

        stem = self.current_path.stem

        # --------------------------------------------------------------
        # 最终 Mask：
        #
        # PNG 无损格式
        # --------------------------------------------------------------

        mask_path = (
            MASK_DIR
            /
            f"{stem}.png"
        )

        # --------------------------------------------------------------
        # 可视化同样保存 PNG
        #
        # 避免 JPEG 压缩。
        # --------------------------------------------------------------

        vis_path = (
            VIS_DIR
            /
            f"{stem}_vis.png"
        )

        prompt_path = (
            PROMPT_DIR
            /
            f"{stem}.json"
        )

        # --------------------------------------------------------------
        # 保存前最终保证：
        #
        # mask.shape ==
        # original_image.shape[:2]
        # --------------------------------------------------------------

        self.mask = (
            self.ensure_mask_original_size(
                self.mask
            )
        )

        if self.mask.shape != (
            self.h,
            self.w,
        ):

            raise RuntimeError(
                "Mask size mismatch:\n"
                f"Mask  : "
                f"{self.mask.shape[1]} x "
                f"{self.mask.shape[0]}\n"
                f"Image : "
                f"{self.w} x {self.h}"
            )

        # --------------------------------------------------------------
        # bool ->
        # strictly 0 / 255
        # --------------------------------------------------------------

        mask_u8 = (
            self.mask.astype(
                np.uint8
            )
            *
            255
        )

        # --------------------------------------------------------------
        # 最后再严格二值一次
        #
        # 理论上这一步不是必须的，
        # 但可以保证内部一定只有：
        #
        # 0
        # 255
        # --------------------------------------------------------------

        mask_u8 = np.where(
            mask_u8 > 127,
            255,
            0,
        ).astype(
            np.uint8
        )

        unique_values = np.unique(
            mask_u8
        )

        print()
        print(
            "Mask unique values:",
            unique_values
        )

        # --------------------------------------------------------------
        # PNG 无损保存
        # --------------------------------------------------------------

        success = cv2.imwrite(
            str(mask_path),
            mask_u8,
            [
                cv2.IMWRITE_PNG_COMPRESSION,
                3,
            ],
        )

        if not success:

            raise RuntimeError(
                f"Failed to save mask:\n"
                f"{mask_path}"
            )

        # --------------------------------------------------------------
        # PNG 保存后再次读取检查
        #
        # 确认：
        #
        # 1. 尺寸没变
        # 2. 仍然只有 0 / 255
        # --------------------------------------------------------------

        check_mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if check_mask is None:

            raise RuntimeError(
                f"Failed to verify mask:\n"
                f"{mask_path}"
            )

        if check_mask.shape != (
            self.h,
            self.w,
        ):

            raise RuntimeError(
                "Saved mask size mismatch:\n"
                f"Saved : "
                f"{check_mask.shape[1]} x "
                f"{check_mask.shape[0]}\n"
                f"Image : "
                f"{self.w} x {self.h}"
            )

        saved_unique = np.unique(
            check_mask
        )

        if not np.all(
            np.isin(
                saved_unique,
                [0, 255]
            )
        ):

            raise RuntimeError(
                "Saved PNG mask is not binary:\n"
                f"{saved_unique}"
            )

        # --------------------------------------------------------------
        # 保存原分辨率 visualization
        # --------------------------------------------------------------

        vis = self.image_bgr.copy()

        overlay = vis.copy()

        overlay[self.mask] = (
            0,
            0,
            255,
        )

        vis = cv2.addWeighted(
            overlay,
            MASK_ALPHA,
            vis,
            1.0 - MASK_ALPHA,
            0,
        )

        # --------------------------------------------------------------
        # 在原始分辨率图像上绘制 prompt
        # --------------------------------------------------------------

        for point, label in zip(
            self.points,
            self.labels,
        ):

            x, y = point

            color = (
                (0, 255, 0)
                if label == 1
                else
                (255, 0, 0)
            )

            cv2.circle(
                vis,
                (x, y),
                15,
                color,
                -1,
                cv2.LINE_AA,
            )

        # --------------------------------------------------------------
        # visualization 也保存为 PNG
        # --------------------------------------------------------------

        cv2.imwrite(
            str(vis_path),
            vis,
            [
                cv2.IMWRITE_PNG_COMPRESSION,
                3,
            ],
        )

        # --------------------------------------------------------------
        # 保存 prompt 坐标
        # --------------------------------------------------------------

        prompt_data = {

            "image": (
                self.current_path.name
            ),

            "width": int(
                self.w
            ),

            "height": int(
                self.h
            ),

            "mask_width": int(
                mask_u8.shape[1]
            ),

            "mask_height": int(
                mask_u8.shape[0]
            ),

            "score": (
                self.score
            ),

            "points": [

                {

                    "x": int(
                        p[0]
                    ),

                    "y": int(
                        p[1]
                    ),

                    "label": int(
                        label
                    ),

                    "type": (
                        "foreground"
                        if label == 1
                        else
                        "background"
                    ),

                }

                for p, label
                in zip(
                    self.points,
                    self.labels,
                )
            ],
        }

        with open(
            prompt_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                prompt_data,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print()
        print("=" * 80)
        print("Saved")
        print("=" * 80)

        print(
            "Image size :",
            f"{self.w} x {self.h}"
        )

        print(
            "Mask size  :",
            f"{mask_u8.shape[1]} x "
            f"{mask_u8.shape[0]}"
        )

        print(
            "Mask values:",
            saved_unique
        )

        print(
            "Mask       :",
            mask_path
        )

        print(
            "Vis        :",
            vis_path
        )

        print(
            "Prompts    :",
            prompt_path
        )

    # ==================================================================
    # 下一张
    # ==================================================================

    def next_image(self):

        if self.image_index >= (
            len(self.image_paths) - 1
        ):

            print(
                "Already at last image."
            )

            return

        self.load_image(
            self.image_index + 1
        )

    # ==================================================================
    # 上一张
    # ==================================================================

    def previous_image(self):

        if self.image_index <= 0:

            print(
                "Already at first image."
            )

            return

        self.load_image(
            self.image_index - 1
        )

    # ==================================================================
    # 主循环
    # ==================================================================

    def run(self):

        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_NORMAL,
        )

        cv2.setMouseCallback(
            WINDOW_NAME,
            self.mouse_callback,
        )

        print()
        print("=" * 80)
        print("Controls")
        print("=" * 80)

        print(
            "Left click   : foreground point"
        )

        print(
            "Right click  : background point"
        )

        print(
            "C            : add center foreground point"
        )

        print(
            "U            : undo last point"
        )

        print(
            "R            : reset prompts"
        )

        print(
            "S            : save PNG mask"
        )

        print(
            "H            : no sky -> save all-white PNG mask"
        )

        print(
            "GUI button   : 没有天空 -> save all-white PNG mask"
        )

        print(
            "N            : next image"
        )

        print(
            "P            : previous image"
        )

        print(
            "Q / ESC      : quit"
        )

        print("=" * 80)
        print()

        while True:

            vis = self.render()

            cv2.imshow(
                WINDOW_NAME,
                vis,
            )

            key = (
                cv2.waitKey(20)
                &
                0xFF
            )

            if key == 255:
                continue

            # q / ESC
            if key in [
                ord("q"),
                ord("Q"),
                27,
            ]:

                break

            # center
            elif key in [
                ord("c"),
                ord("C"),
            ]:

                self.add_center_point()

            # undo
            elif key in [
                ord("u"),
                ord("U"),
            ]:

                self.undo()

            # reset
            elif key in [
                ord("r"),
                ord("R"),
            ]:

                self.reset()

            # save
            elif key in [
                ord("s"),
                ord("S"),
            ]:

                self.save()

            # no sky -> save all-white mask
            elif key in [
                ord("h"),
                ord("H"),
            ]:

                self.save_no_sky_mask()

            # next
            elif key in [
                ord("n"),
                ord("N"),
            ]:

                self.next_image()

            # previous
            elif key in [
                ord("p"),
                ord("P"),
            ]:

                self.previous_image()

        cv2.destroyAllWindows()


# ======================================================================
# main
# ======================================================================

def main():

    # ------------------------------------------------------------------
    # 路径检查
    # ------------------------------------------------------------------

    if not IMAGE_DIR.exists():

        raise FileNotFoundError(
            f"Image directory does not exist:\n"
            f"{IMAGE_DIR}"
        )

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            f"Checkpoint does not exist:\n"
            f"{CHECKPOINT}"
        )

    # ------------------------------------------------------------------
    # 输出目录
    # ------------------------------------------------------------------

    MASK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    VIS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROMPT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # 搜索图像
    # ------------------------------------------------------------------

    image_paths = sorted(
        [
            p

            for p in IMAGE_DIR.iterdir()

            if (
                p.is_file()
                and
                p.suffix.lower()
                in IMAGE_SUFFIXES
            )
        ]
    )

    if not image_paths:

        raise RuntimeError(
            f"No images found:\n"
            f"{IMAGE_DIR}"
        )

    print("=" * 80)

    print(
        "SAM2 OpenCV Interactive Segmenter"
    )

    print("=" * 80)

    print(
        "Images :",
        IMAGE_DIR
    )

    print(
        "Count  :",
        len(image_paths)
    )

    print(
        "Output :",
        OUTPUT_DIR
    )

    print()

    # ------------------------------------------------------------------
    # SAM2
    # ------------------------------------------------------------------

    predictor = build_predictor()

    app = SAM2Interactive(
        predictor,
        image_paths,
    )

    app.run()


if __name__ == "__main__":
    main()