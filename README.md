# Rotational data augmentation for Yolo

Image data augmentation is the technique that generates new transformed versions of training images from the given image dataset. This process increases dataset diversity and improves the generalization ability of the model.

The polar coordinates of a point P in a plane specify the distance of the point from the origin O at a specified angle θ from a reference direction. We find the best-fit rotation bounding boxes using the polar coordinate system on the LabelMe annotation tools output image. 

![Screenshot from 2023-01-23 17-04-15](https://user-images.githubusercontent.com/19527298/214054632-5d5c17c8-21e5-4eb1-9fd0-95ebfe25cb90.png)

One common data augmentation technique is the random rotation of an image which is an affine transformation. Affine transformation is a linear mapping method that preserves the parallelism of lines. The output of rotationAugmentationFromJSONToYoloFormat.py (https://github.com/marjanjalali/Yolo-rotation-augmentation/blob/main/rotationAugmentationFromJSONToYoloFormat.py) shown below:

<p align="center">
  <img width = 1000 src="https://user-images.githubusercontent.com/19527298/214285859-89c712db-7f30-46b1-8393-98d7d7466670.png">
</p>

























