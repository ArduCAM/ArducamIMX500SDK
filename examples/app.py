import os
import cv2
import numpy as np
import argparse
import traceback
from ArducamIMX500SDK import IMX500Uvc
from postprocess.parse_mobilenetv2 import ParserMobilenetv2
from postprocess.parse_mobilenetssd import ParserMobilenetSsd
from postprocess.parse_yolov8n_det import ParserYolov8nDet
from postprocess.parse_yolov8n_pos import ParserYolov8nPos
from postprocess.parse_yolov8n_pos_hand import ParserYolov8nPosHand
from postprocess.parse_deeplabv3plus import ParserDeeplabv3plus
from postprocess.parse_yolov8n_det_geofencing import ParserYolov8nDetGeofencing
from postprocess.parse_yolov8n_det_label_package import ParserYolov8nDetLabelPackage
from postprocess.only_input_tensor import only_input_tensor


script_dir_path = os.path.dirname(__file__)
root_dir_path = os.path.join(script_dir_path, '..')
labels_dir_path = os.path.join(root_dir_path, 'labels')

pretrain_model_card = {
    "mobilenetv2": {
        "weights": "../model/arducam_imx500_model_zoo/mobilenetv2/network.fpk",
        "parser": ParserMobilenetv2(os.path.join(labels_dir_path, 'imagenet_labels.txt')),
    },
    "mobilenetssd": {
        "weights": "../model/arducam_imx500_model_zoo/mobilenetssd/network.fpk",
        "parser": ParserMobilenetSsd(os.path.join(labels_dir_path, 'coco_ssd.txt')),
    },
    "yolov8n_det": {
        "weights": "../model/arducam_imx500_model_zoo/yolov8n_det/network.fpk",
        "parser": ParserYolov8nDet(os.path.join(labels_dir_path, 'coco.txt')),
    },
    "yolov8n_pos": {
        "weights": "../model/arducam_imx500_model_zoo/yolov8n_pos/network.fpk",
        "parser": ParserYolov8nPos(),
    },
    "yolov8n_pos_hand": {
        "weights": "../model/arducam_imx500_model_zoo/yolov8n_pos_hand/320_320/network.fpk",
        "parser": ParserYolov8nPosHand(),
    },
    "deeplabv3plus": {
        "weights": "../model/arducam_imx500_model_zoo/deeplabv3plus/network.fpk",
        "parser": ParserDeeplabv3plus(),
    },
}

demo_project_card = {
    "geofencing": {
        "weights": "../model/arducam_imx500_model_zoo/yolov8n_det/network.fpk",
        "parser": ParserYolov8nDetGeofencing(os.path.join(labels_dir_path, 'coco.txt')),
    },
    "package": {
        "weights": "../model/arducam_imx500_model_zoo/research/yolov8n_det_label_package/network.fpk",
        "parser": ParserYolov8nDetLabelPackage(os.path.join(labels_dir_path, 'label_package.txt')),
    },
}

def parse_cmdline():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-wf', '--write-flash', action='store_true', required=False, default=False, help='Flag of flash write.')
    parser.add_argument('-lf', '--loader-firmware', type=str, required=False, help='Loader firmware path.')
    parser.add_argument('-mf', '--main-firmware', type=str, required=False, help='Main firmware path.')
    parser.add_argument('-m', '--model', type=str, required=False, help='Model path.')
    parser.add_argument('-dp', '--demo-project', type=str, required=False, help='Demo project name.')
    parser.add_argument('-pm', '--pretrain-model', type=str, required=False, help='Pretrain model name.')
    parser.add_argument('-d', '--device-id', type=int, default=0, required=False, help='Device Index. (default: 0)')
    parser.add_argument('-dy', '--dump-yuv', action='store_true', required=False, help='Dump YUV.')
    parser.add_argument('-dyr', '--dump-yuv-raw', action='store_true', required=False, help='Dump raw YUV.')
    parser.add_argument('-di', '--data-injection', type=str, required=False, help='Data injection.')
    parser.add_argument('--network-info', type=str, required=False, help='network_info.txt path')
    parser.add_argument('--rect-crop', type=int, nargs=4, metavar=('XMIN', 'YMIN', 'XMAX', 'YMAX'),
                        help='Rect crop area in absolute xyxy format. '
                        'X range: 0-4056, Y range: 0-3040.')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_cmdline()
    is_flash_write_required = args.write_flash
    loader_firmware_path = args.loader_firmware
    main_firmware_path = args.main_firmware
    model_path = args.model
    pretrain_model_name = args.pretrain_model
    network_info = args.network_info
    demo_project = args.demo_project
    if demo_project is not None:
        model_path = demo_project_card[demo_project]["weights"]
        postprocess_cb = demo_project_card[demo_project]["parser"]
    else:
        if model_path is None:
            if pretrain_model_name is not None:
                model_path = pretrain_model_card[pretrain_model_name]["weights"]
                postprocess_cb = pretrain_model_card[pretrain_model_name]["parser"]
            else:
                model_path = pretrain_model_card["mobilenetssd"]["weights"]
                postprocess_cb = pretrain_model_card["mobilenetssd"]["parser"]
        else:
            postprocess_cb = only_input_tensor
    device_index = args.device_id
    dump_yuv_raw = args.dump_yuv_raw
    dump_yuv = args.dump_yuv
    data_injection_path = args.data_injection

    app = IMX500Uvc(device_index)

    app.open(
        dump_yuv=dump_yuv, 
        dump_yuv_raw=dump_yuv_raw,
    )

    app.download_firmware(
        loader_firmware_path=loader_firmware_path,
        main_firmware_path=main_firmware_path,
        model_path=model_path,
        is_flash_write_required=is_flash_write_required,
        network_info_file_path=network_info
    )
    
    print(app.get_fw_version())
    print("Sensor Device ID: ", app.get_sensor_device_id())

    if data_injection_path is not None:
        app.data_injection(data_injection_path)
    else:
        if args.rect_crop:
            xmin, ymin, xmax, ymax = args.rect_crop
            app.rect_nn_input_map_xyxy_absolute(xmin, ymin, xmax, ymax)

    while True:
        try:
            ret, img, networks, img_postprocess, model_input_img_postprocess = app.read(
                postprocess_cb=postprocess_cb
            )
            if img_postprocess is not None:
                cv2.imshow("YUV_DNN", img_postprocess)
            if model_input_img_postprocess is not None:
                cv2.imshow("DNN", model_input_img_postprocess)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        except Exception as e:
            traceback.print_exc()
            break

    cv2.destroyAllWindows()
