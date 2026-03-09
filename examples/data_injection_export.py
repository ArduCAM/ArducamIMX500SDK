import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from ArducamIMX500SDK import IMX500Uvc
from postprocess.only_input_tensor import only_input_tensor


SCRIPT_DIR_PATH = os.path.dirname(__file__)

PRETRAIN_MODEL_CARD = {
    "mobilenetv2": {
        "weights": "../model/arducam_imx500_model_zoo/mobilenetv2/network.fpk",
    },
    "mobilenetssd": {
        "weights": "../model/arducam_imx500_model_zoo/mobilenetssd/network.fpk",
    },
    "yolov8n_det": {
        "weights": "../model/arducam_imx500_model_zoo/yolov8n_det/network.fpk",
    },
    "yolov8n_pos": {
        "weights": "../model/arducam_imx500_model_zoo/yolov8n_pos/network.fpk",
    },
    "yolov8n_pos_hand": {
        "weights": "../model/arducam_imx500_model_zoo/yolov8n_pos_hand/320_320/network.fpk",
    },
    "deeplabv3plus": {
        "weights": "../model/arducam_imx500_model_zoo/deeplabv3plus/network.fpk",
    },
}

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm", ".pbm"
}

def parse_cmdline():
    parser = argparse.ArgumentParser(
        description="Run one-shot data injection and export image/input-tensor/network-json files."
    )
    parser.add_argument("-lf", "--loader-firmware", type=str, required=False, help="Loader firmware path.")
    parser.add_argument("-mf", "--main-firmware", type=str, required=False, help="Main firmware path.")
    parser.add_argument("-m", "--model", type=str, required=False, help="Model path.")
    parser.add_argument("-pm", "--pretrain-model", type=str, required=False, help="Pretrain model name.")
    parser.add_argument("-d", "--device-id", type=int, default=0, required=False, help="Device index. (default: 0)")
    parser.add_argument("--network-info", type=str, required=False, help="network_info.txt path")
    parser.add_argument(
        "-i",
        "--input-image",
        type=str,
        required=True,
        help="Input image path or directory path for data injection.",
    )
    parser.add_argument("-o", "--output-dir", type=str, required=True, help="Output root directory.")
    parser.add_argument("--fps", type=int, default=20, help="NN run fps passed to download_firmware. (default: 20)")
    parser.add_argument("--read-timeout-sec", type=float, default=5.0, help="Read timeout in seconds. (default: 5)")
    parser.add_argument("--max-read-attempts", type=int, default=200, help="Max read attempts. (default: 200)")
    return parser.parse_args()


def resolve_model_path(model_path, pretrain_model_name):
    if model_path:
        return model_path
    if pretrain_model_name:
        if pretrain_model_name not in PRETRAIN_MODEL_CARD:
            raise ValueError(
                f"Unsupported pretrain model: {pretrain_model_name}. "
                f"Available: {', '.join(sorted(PRETRAIN_MODEL_CARD.keys()))}"
            )
        return PRETRAIN_MODEL_CARD[pretrain_model_name]["weights"]
    return PRETRAIN_MODEL_CARD["mobilenetssd"]["weights"]


def networks_to_serializable(networks):
    if networks is None:
        return {"error": "networks is None"}

    if hasattr(networks, "to_dict"):
        return {"parsed_metadata": networks.to_dict()}
    if hasattr(networks, "_to_dict"):
        return {"parsed_metadata": networks._to_dict()}

    if isinstance(networks, (list, tuple)):
        result = []
        for item in networks:
            if hasattr(item, "to_dict"):
                result.append(item.to_dict())
            elif hasattr(item, "_to_dict"):
                result.append(item._to_dict())
            else:
                result.append(repr(item))
        return {"parsed_metadata": result}

    return {"parsed_metadata": {"repr": repr(networks)}}


def read_once(app, timeout_sec, max_attempts):
    start = time.time()
    attempts = 0
    while attempts < max_attempts and (time.time() - start) < timeout_sec:
        attempts += 1
        ret, img, networks, img_postprocess, model_input_img_postprocess = app.read(postprocess_cb=only_input_tensor)
        if ret:
            return img, networks, img_postprocess, model_input_img_postprocess
    raise RuntimeError(
        f"Failed to read a valid frame after {attempts} attempts within {timeout_sec:.2f} seconds."
    )


def warmup_read(app, timeout_sec=2.0, max_attempts=80):
    start = time.time()
    attempts = 0
    while attempts < max_attempts and (time.time() - start) < timeout_sec:
        attempts += 1
        ret, _, _, _, _ = app.read(postprocess_cb=only_input_tensor)
        if ret:
            return True
    return False


def convert_json_value(obj):
    if isinstance(obj, bytes):
        return obj.decode()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Type not serializable: {type(obj)}")


def collect_input_images(input_path):
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {input_path.suffix}")
        return [input_path]

    if input_path.is_dir():
        images = [
            p for p in sorted(input_path.iterdir())
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        if not images:
            raise FileNotFoundError(f"No supported images found in directory: {input_path}")
        return images

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def process_one_image(app, image_path, output_root, read_timeout_sec, max_read_attempts):
    test_img = cv2.imread(str(image_path))
    if test_img is None:
        raise RuntimeError(f"cv2.imread failed for input image: {image_path}")

    output_dir = output_root / image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    app.data_injection(str(image_path))
    img, networks, img_postprocess, model_input_img_postprocess = read_once(
        app, read_timeout_sec, max_read_attempts
    )

    image_save = img_postprocess if img_postprocess is not None else img
    if image_save is None:
        raise RuntimeError(f"[{image_path.name}] image output is None.")
    if model_input_img_postprocess is None:
        raise RuntimeError(f"[{image_path.name}] input tensor image output is None.")

    image_file = output_dir / "image.png"
    input_tensor_path = output_dir / "input_tensor.png"
    network_json_path = output_dir / "parsed_metadata.json"

    cv2.imwrite(str(image_file), image_save)
    cv2.imwrite(str(input_tensor_path), model_input_img_postprocess)

    with open(network_json_path, "w", encoding="utf-8") as f:
        json.dump(networks_to_serializable(networks), f, ensure_ascii=False, indent=2, default=convert_json_value)

    print(f"[{image_path.name}] saved -> {output_dir}")


def main():
    args = parse_cmdline()
    input_path = Path(args.input_image).resolve()
    output_root = Path(args.output_dir).resolve()
    model_path = resolve_model_path(args.model, args.pretrain_model)

    images = collect_input_images(input_path)
    print(
        f"Input={input_path} ({'dir' if input_path.is_dir() else 'file'}), "
        f"image_count={len(images)}, output_root={output_root}"
    )
    print(f"Model={model_path}, fps={args.fps}, device_id={args.device_id}")

    app = IMX500Uvc(args.device_id)
    app.open(dump_yuv=False, dump_yuv_raw=False)

    app.download_firmware(
        loader_firmware_path=args.loader_firmware,
        main_firmware_path=args.main_firmware,
        model_path=model_path,
        is_flash_write_required=False,
        network_info_file_path=args.network_info,
        fps=args.fps,
    )

    for idx, image in enumerate(images, start=1):
        print(f"Processing [{idx}/{len(images)}]: {image.name}")
        process_one_image(
            app=app,
            image_path=image,
            output_root=output_root,
            read_timeout_sec=args.read_timeout_sec,
            max_read_attempts=args.max_read_attempts,
        )

    print(f"Completed: total={len(images)}")


if __name__ == "__main__":
    main()
