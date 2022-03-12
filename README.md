# S5CL: Unifying fully-supervised, self-supervised, and semi-supervised learning through hierarchical contrastive learning

S5CL combines fully-supervised, self-supervised, and semi-supervised learning into one single framework. The approch combines three contrastive losses on labeled, unlabeled and augmented images to learn a hierarchy of feature representations. In particular, given an input, similar images and augmented views are embedded the closest, followed by different looking images, while images from different classes have the greatest distance.

## Description

First, a labeled image is augmented weakly and strongly. Weak augmentations consists of simple color and geometric augmentations like rotation or flipping. Strong augmenatations, on the other hand, applies cropping, deformations, and other color augmentations techniques. Both weakly and strongly augmented images are then inserted into a supervsied contrastive loss that pushes feature representations of images and transformations of the same class together and from other lasses away. 

We 
