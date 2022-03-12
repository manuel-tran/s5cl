# S5CL: Unifying fully-supervised, self-supervised, and semi-supervised learning through hierarchical contrastive learning

S5CL combines fully supervised, self-supervised, and semi-supervised learning into one single framework. The approach combines three contrastive losses on labeled, unlabeled and augmented images to learn a hierarchy of feature representations. In particular, given an input, similar images and augmented views are embedded the closest, followed by different looking images, while images from different classes have the greatest distance.

## Description

Here, we give a quick overview of S5CL: Apply weak augmentations (e.g., rotation) and strong augmentations (e.g., cropping) on labeled images. Then use a supervised contrastive loss to push feature representations of augmented images from the same class together and those from other classes away. 

If unlabeled images are available, augment them weakly and strongly as well. Insert them into a self-supervised contrastive loss and treat each image as its own class. Since we are in the same embedding space, the unlabeld images indirectly will be moved to their corresponding labeled clusters. 

After a few epochs, use the classifier to predict pseudo-labeles for the unlabeled images and replace the loss with a semi-supervised contrastive loss. The first image below serves as an intuition of how this method works, while the other two figures above illustrate the actual implementation. 


## Losses

To avoid that the contrastive losses on labeled and unlabeld images to be in conflict with each other, the hyperparameter which controls the cluster density should be different.

In our case we, use the SupConLoss with a single hyperparameters called the temperature. Its value should be higher for unlabeled images than for labeled images. Instead of this loss, it is also pososible to use this implementation with other loss functions.

