# S5CL: Supervised, Self-Supervised, and Semi-Supervised Contrastive Learning

![results](img/results.png)

S5CL unifies fully-supervised, self-supervised, and semi-supervised learning into one single framework. In particular, S5CL uses a hierarchy of contrastive losses to extract feature representations from both labeled and unlabeled data at the same time. This leads to richer, more structured, and more compact feature embeddings that can be used for various downstream tasks such as image classification or image retrieval.

Evaluations on two public datasets show strong improvements over other fully-supervised and semi-supervised methods in case of sparse labels: On the colon cancer dataset NCT-CRC-HE-100K, the accuracy increases by up to 9%; while on the highly unbalanced leukemia single-cell dataset Munich AML Morphology, the F1-score increases by up to 6%. Notably, on these two datasets, S5CL also outperforms the SOTA semi-supervised method Meta Pseudo Labels (MPL).

## Overview

S5CL employs the following steps: 

* Apply weak augmentations (e.g., rotation) and strong augmentations (e.g., cropping) to labeled images. 

* Use a supervised contrastive loss to push feature representations of images from the same class towards each other and feature representations from other classes away. 

* If unlabeled images are available, augment them weakly and strongly as well. 

* Insert them into a self-supervised contrastive loss and treat each image as its own class. 

* Since we use the same embedding space for both labeled and unlabeled data, the unlabeled images will indirectly be moved to their corresponding labeled clusters. 

* After a few epochs, use the classifier to predict pseudo-labeles for the unlabeled images and replace the self-supervised contrastive loss with a semi-supervised contrastive loss. 

![illustration](img/illustration.png)


## Implementation

We use the state-of-the-art SupConLoss. It outperforms other contrastive losses such as SimCLR, does not require hard-negative mining, and only has one hyperparameter called the temperature that controls the cluster density. To avoid conflicts between the supervised, self-supervised , and semi-supervised losses, the temperature for the unlabeled images should always be larger than the temperature for the labeled images.

It is worth noting, that our framework also works with other contrastive or metric losses. To use other loss functions, the training script might need to be adapted to accommodate mining or additional hyperparameters.

