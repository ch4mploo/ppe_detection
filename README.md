# PPE Detection Using Ultralytics YOLO 26 Nano

## Approach
1. Trained a custom YOLO26 model (nano) to detect PPE classes. This is done with Ultralytics YOLO library.
2. Dataset to train the model is obtained from Ultralytics as well. You may refer to the source [here](https://docs.ultralytics.com/datasets/detect/construction-ppe/)
3. Trained with mostly default hyperparameters configuration (100 epochs, image size 640)
4. Results:

![training_result](/assets/img/results.png)

## Future Plan
1. Implement on Raspberry Pi to have a portable setup.
2. Implement on NVIDIA Jetson Orin Nano for high frame rate and real time detection.

## License
This project is under AGPL-3.0 License, as both Ultralytics YOLO and Construction-PPE dataset are under the same license as well.