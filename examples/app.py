import cv2
import numpy as np
import argparse
import traceback
from ArducamIMX500SDK import IMX500Uvc
from postprocess.mobilenetv2 import parse_mobilenetv2
from postprocess.mobilenetssd import ParseMobilenetSSD
from postprocess.yolov8n_det import ParserYolov8Det
from postprocess.yolov8n_pos import parse_yolov8n_pos
from postprocess.yolov8n_pos_hand import parse_yolov8n_hand_pos
from postprocess.deeplabv3plus import parse_deeplabv3plus


pretrain_model_card = {
    "mobilenetv2": {
        "weights": "../model/arducam_imx500_model_zoo/mobilenetv2/network.fpk",
        "parser": parse_mobilenetv2,
    },
    "mobilenetssd": {
        "weights": "../model/arducam_imx500_model_zoo/mobilenetssd/network.fpk",
        "parser": ParseMobilenetSSD(),
    },
    "yolov8n_det": {
        "weights": "../model/arducam_imx500_model_zoo/yolov8n_det/network.fpk",
        "parser": ParserYolov8Det(),
    },
    "yolov8n_pos": {
        "weights": "../model/arducam_imx500_model_zoo/yolov8n_pos/network.fpk",
        "parser": parse_yolov8n_pos,
    },
    "yolov8n_pos_hand": {
        "weights": "../model/arducam_imx500_model_zoo/yolov8n_pos_hand/network.fpk",
        "parser": parse_yolov8n_hand_pos,
    },
    "deeplabv3plus": {
        "weights": "../model/arducam_imx500_model_zoo/deeplabv3plus/network.fpk",
        "parser": parse_deeplabv3plus,
    },
}

def parse_cmdline():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-wf', '--write-flash', action='store_true', required=False, default=False, help='Flag of flash write.')
    parser.add_argument('-lf', '--loader-firmware', type=str, required=False, help='Loader firmware path.')
    parser.add_argument('-mf', '--main-firmware', type=str, required=False, help='Main firmware path.')
    parser.add_argument('-m', '--model', type=str, required=False, help='Model path.')
    parser.add_argument('-pm', '--pretrain-model', type=str, required=False, help='Pretrain model name.')
    parser.add_argument('-d', '--device-id', type=int, default=0, required=False, help='Device Index. (default: 0)')
    parser.add_argument('-dy', '--dump-yuv', action='store_true', required=False, help='Dump YUV.')
    parser.add_argument('-dyr', '--dump-yuv-raw', action='store_true', required=False, help='Dump raw YUV.')
    parser.add_argument('-di', '--data-injection', type=str, required=False, help='Data injection.')
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
        is_flash_write_required=is_flash_write_required
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
            cv2.imshow("YUV_DNN", img_postprocess)
            cv2.imshow("DNN", model_input_img_postprocess)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        except Exception as e:
            traceback.print_exc()
            break

    cv2.destroyAllWindows()
