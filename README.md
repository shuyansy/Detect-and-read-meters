# Detect and Read meters in the wild
This is areleased system towards detection and recognition of complex meters in wild. The system can be divided into three moduels. Fisrtly, a yolo-based detector is applied to get pure meter region. Secondly, a spatial transformer module is eatablished to rectify the position of meter. Lastly, an end-to-end network is to read meter values, which is implemented by pointer/dail predcition and key number learning.    

## Visulization results
![](1.png)
![](2.png)

Left row is the original image, middle row is the process of meter rectification, right row is the result of meter value reading.



## ToDo List

- [x] Release testing code
- [x] Release training code and dataset
- [x] existing three-stage models
- [ ] A new branch for digital-meter recognition
- [x] Document for testing
- [x] Document for training
- [x] Demo script for single image


## Installation

### Requirements:
- Python3 (Python3.7 is recommended)
- PyTorch >= 1.0 
- torchvision from master
- numpy
- skimage
- OpenCV==3.0.x
- CUDA >= 9.0 (10.0 is recommended)

## Meter Detection 
We use official YOLO-V5 to detect meters.
We release a dataset for training model, which can be downloaded from https://drive.google.com/file/d/1RKcqJ0RWaBPpBbMtWwcgQ4S66Iwf97RS/view?usp=drive_link The data is COCO-format and label 0 and 1 represent pointer meters and digital meters.
We also provide trained weight in https://drive.google.com/file/d/1bHYpJro3ERmNTRO2JEo1inyU0_juqw5z/view?usp=drive_link You dan put it in the yolov5 folder for inference.

## Models
Download Trained [model](https://drive.google.com/drive/folders/1juFFjBz9BlJEuLc_IxFj5RUz0Z_UfO0M?usp=sharing)

Please put distro_net.pt into meter_distro/weight.  
put textgraph_vgg_450.pth into model/meter_data.

## Demo 
You can run a demo script for a single image inference by two steps.

```python get_meter_area.py```. and the detected meter will be stored in scene_image_data/deteced_meter

```python predict.py``` to get distored meter and final result.




