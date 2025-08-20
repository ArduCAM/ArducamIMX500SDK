# ArducamIMX500SDK

<img src="https://www.arducam.com/wp-content/uploads/2023/07/logo-4.png" width="100%" />

## Install

Please visit the [Releases](https://github.com/ArduCAM/ArducamIMX500SDK/releases/latest) page and download the corresponding `.whl` installation package based on your operating system and Python version.

After downloading, install it using the following command:

```bash
pip install your_package_name‑<version>‑<python_version>‑<platform>.whl
```

### Example

Suppose you are using Windows and Python 3.8, and the downloaded file is:

```
arducamimx500sdk‑0.1.1‑cp38‑cp38‑win_amd64.whl
```

Then run the following command in the terminal:

```bash
pip install arducamimx500sdk‑0.1.1‑cp38‑cp38‑win_amd64.whl
```


## Simple and Powerful API

ArducamIMX500SDK is designed to be both **easy to use** and **highly extensible**. Whether you're looking for a plug-and-play AI camera experience or planning to build complex computer vision pipelines, this SDK offers a clean and flexible Python interface.

### 🚀 Get Started

Clone this repository and run the following command in the project directory:

```bash
git submodule update --init --recursive
# mobilenetv2
python app.py -pm mobilenetv2
# mobilenetssd
python app.py -pm mobilenetssd
# yolov8n_det
python app.py -pm yolov8n_det
# yolov8n_pos
python app.py -pm yolov8n_pos
# yolov8n_pos_hand
python app.py -pm yolov8n_pos_hand
# deeplabv3plus
python app.py -pm deeplabv3plus
```

### Command-line Options

```bash
usage: app.py [-h] [-wf] [-lf LOADER_FIRMWARE] [-mf MAIN_FIRMWARE] [-m MODEL] [-pm PRETRAIN_MODEL] [-d DEVICE_ID] [-dy] [-dyr] [-di DATA_INJECTION] [--rect-crop XMIN YMIN XMAX YMAX]

options:
  -h, --help            show this help message and exit
  -wf, --write-flash    Flag of flash write.
  -lf LOADER_FIRMWARE, --loader-firmware LOADER_FIRMWARE
                        Loader firmware path.
  -mf MAIN_FIRMWARE, --main-firmware MAIN_FIRMWARE
                        Main firmware path.
  -m MODEL, --model MODEL
                        Model path.
  -pm PRETRAIN_MODEL, --pretrain-model PRETRAIN_MODEL
                        Pretrain model name.
  -d DEVICE_ID, --device-id DEVICE_ID
                        Device Index. (default: 0)
  -dy, --dump-yuv       Dump YUV.
  -dyr, --dump-yuv-raw  Dump raw YUV.
  -di DATA_INJECTION, --data-injection DATA_INJECTION
                        Data injection.
  --rect-crop XMIN YMIN XMAX YMAX
                        Rect crop area in absolute xyxy format. X range: 0-4056, Y range: 0-3040.
```


## Ready for Advanced Development

If you plan to build your own applications, pipelines, or integrate with other systems, these APIs give you the control and flexibility you need.

📚 **For full API reference and data structures**, please refer to the official documentation:
[**ArducamIMX500SDK.pdf**](https://github.com/ArduCAM/ArducamIMX500SDK/releases/latest/download/ArducamIMX500SDK.pdf)

This document includes detailed explanations of class methods, parameter definitions, and usage examples — essential for anyone doing secondary development on the Arducam IMX500 platform.

