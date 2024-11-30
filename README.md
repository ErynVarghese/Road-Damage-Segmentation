This project uses YOLOv8 for detecting and segmenting Road damage (cracks and potholes). 

The model is trained on a dataset provided by Roboflow.
* Number of images in training set: 650
* Number of images in validation set: 200
* Number of images in test set: 150

Due to size limitations, I have included one image each for training and validation's images and labels.

The training and validation sets mostly have long roads with small cracks and potholes. This means the model works best for similar types of road damage.

The test images in test-images directory are a mix of images from the dataset's test folder and some external images. The results are saved in the test_result folder.
The detected road damage is represented with color-coded masks:
* Cracks are shown in light red.
* Potholes are shown in light blue.
