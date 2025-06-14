import io
import logging
import torch
import numpy as np
import cv2
from doctr.models import ocr_predictor
from doctr.io import Document
import base64
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("OCRManagerDocTR")
logging.basicConfig(level=logging.INFO)

class OCRManager:
    def __init__(self, device_idx: int = 0, warmup_on_init: bool = True, use_fp16: bool = True):
        self.torch_device = None
        self._setup_device(device_idx)
        self.fp16_enabled = use_fp16 and self.torch_device.type == 'cuda'

        try:
            self.predictor = ocr_predictor(
                det_arch='linknet_resnet18',
                reco_arch='crnn_mobilenet_v3_small',
                pretrained=True,
                assume_straight_pages=True,
                detect_orientation=False,
                detect_language=False,
            )

            if self.fp16_enabled:
                self.predictor = self.predictor.half()

            self.predictor = self.predictor.to(self.torch_device)
            logger.info(f"OCRManagerDocTR: Predictor loaded on {self.torch_device}" + (" (FP16 enabled)" if self.fp16_enabled else ""))
        except Exception as e:
            logger.exception(f"OCRManagerDocTR: Failed to load predictor: {e}")
            self.predictor = None
            raise

        if warmup_on_init:
            self.warm_up_model()
            logger.info("OCRManagerDocTR: Warm-up complete")

    def _setup_device(self, device_idx_req: int):
        if device_idx_req >= 0 and torch.cuda.is_available():
            self.torch_device = torch.device(f"cuda:{device_idx_req}")
            name = torch.cuda.get_device_name(self.torch_device)
            mem = torch.cuda.get_device_properties(self.torch_device).total_memory / (1024**3)
            logger.info(f"OCRManagerDocTR: Using GPU {device_idx_req} ({name}, {mem:.1f} GB)")
            torch.cuda.empty_cache()
        else:
            self.torch_device = torch.device("cpu")
            logger.info("OCRManagerDocTR: Using CPU")

    def warm_up_model(self):
        dummy_img = np.zeros((128, 128, 3), dtype=np.uint8)
        if self.predictor:
            self.predictor([dummy_img])

    def _bytes_to_cv2(self, image_bytes: bytes) -> np.ndarray:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else np.zeros((128, 128, 3), dtype=np.uint8)

    def _extract_text_from_page(self, page_obj) -> str:
        if not hasattr(page_obj, "blocks"):
            logger.warning(f"OCRManagerDocTR: Page object missing 'blocks' attribute, got {type(page_obj)}")
            return ""
        lines = []
        for block in page_obj.blocks:
            for line in block.lines:
                line_text = " ".join(word.value for word in line.words)
                lines.append(line_text)
        return "\n".join(lines).strip()

    def ocr(self, image_bytes: bytes) -> str:
        if not self.predictor:
            logger.error("OCRManagerDocTR: Predictor not initialized")
            return ""

        try:
            img_np = self._bytes_to_cv2(image_bytes)
            result_doc = self.predictor([img_np])

            if isinstance(result_doc, Document) and result_doc.pages:
                return self._extract_text_from_page(result_doc.pages[0])
            else:
                logger.warning(f"OCRManagerDocTR: Predictor did not return Document with pages, got {type(result_doc)}")
                return ""
        except Exception as e:
            logger.error(f"OCRManagerDocTR: OCR failed: {e}", exc_info=True)
            return ""

    def batch_ocr(self, images_b64: list[str]) -> list[str]:
        if not self.predictor:
            logger.error("OCRManagerDocTR: Predictor not initialized")
            return []

        def process_image(b64_str: str) -> np.ndarray:
            try:
                np_arr = np.frombuffer(base64.b64decode(b64_str), np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else np.zeros((128, 128, 3), dtype=np.uint8)
            except Exception as e:
                logger.warning(f"Image processing failed: {e}")
                return np.zeros((128, 128, 3), dtype=np.uint8)

        with ThreadPoolExecutor() as executor:
            images_np = list(executor.map(process_image, images_b64))

        try:
            doc = self.predictor(images_np)
            return [self._extract_text_from_page(page) for page in doc.pages]
        except Exception as e:
            logger.error(f"Batch OCR failed: {e}", exc_info=True)
            return []
