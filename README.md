# A three-stage detection and recognition pipeline of complex meters in wild
This is the first released system towards detection and recognition of complex meters in wild. The system can be divided into three moduels. Fisrtly, a yolo-based detector is applied to get pure meter region. Secondly, a spatial transformer module is eatablished to rectify the position of meter. Lastly, an end-to-end network is to read meter values, which is implemented by pointer/dail predcition and key number learning.      


## ToDo List

- [x] Release testing code
- [] Release training code and dataset(after paper release)
- [x] existing three-stage models
- [ ] A new end-to-end model for image distoration and meter value reading)
- [x] Document for testing
- [] Document for training(after paper release)
- [x] Demo script for single image
- [ ] Demo script for video
- [] Evaluation

