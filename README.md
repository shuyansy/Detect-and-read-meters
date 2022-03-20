# A detection and recognition pipeline of complex meters in wild
This is the first released system towards detection and recognition of complex meters in wild. The system can be divided into three moduels. Fisrtly, a yolo-based detector is applied to get pure meter region. Secondly, a spatial transformer module is eatablished to rectify the position of meter. Lastly, an end-to-end network is to read meter values, which is implemented by pointer/dail predcition and key number prediction.      


## ToDo List

- [x] Release code
- [x] Document for Installation
- [x] Trained models
- [x] Document for testing
- [x] Document for training
- [x] Demo script
- [x] Evaluation
- [ ] Release the standalone recognition model
