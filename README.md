# Rotational data augmentation for Yolo

Image data augmentation is the technique that generates new transformed versions of training images from the given image dataset. This process increases dataset diversity and improves the generalization ability of the model.

One common data augmentation technique is the random rotation of an image which is an affine transformation. Affine transformation is a linear mapping method that preserves the parallelism of lines. 

![Screenshot from 2023-01-23 18-31-24](https://user-images.githubusercontent.com/19527298/214074724-0337f645-7cad-41eb-9ed2-439506ed61a9.png)

The polar coordinates of a point P in a plane specify the distance of the point from the origin O at a specified angle θ from a reference direction. We find the best-fit rotation bounding boxes using the polar coordinate system and the LabelMe annotation tools. 

![Screenshot from 2023-01-23 17-04-15](https://user-images.githubusercontent.com/19527298/214054632-5d5c17c8-21e5-4eb1-9fd0-95ebfe25cb90.png)





























