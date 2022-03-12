# S5CL: Unifying fully-supervised, self-supervised, and semi-supervised learning through hierarchical contrastive learning

S5CL combines fully supervised, self-supervised, and semi-supervised learning into one single framework. The approach combines three contrastive losses on labeled, unlabeled and augmented images to learn a hierarchy of feature representations. In particular, given an input, similar images and augmented views are embedded the closest, followed by different looking images, while images from different classes have the greatest distance.

## Description

Apply weak augmentations (e.g., rotation) and strong augmentations (e.g., cropping) on labeled images. Then use a supervised contrastive loss to push feature representations of augmented images from the same class together and those from other classes away. 

If unlabeled images are available, augment them weakly and strongly as well. Insert them into a self-supervised contrastive loss and treat each image as its own class.

After a few epochs, predict pseudo-labeled for the unlabeled images and replace the loss with a semi-supervised contrastive loss. 

