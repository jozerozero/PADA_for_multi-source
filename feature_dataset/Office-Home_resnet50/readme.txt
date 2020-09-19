The name of the file is X_Y.csv, which means:

X is the labeled source domain, Y is the unlabeled target domain.

First, we split X as training and validation set (8:2). 

Then, we run ImageNet pretrained ResNet-50 on training and validation set to acquire the best model
(save it, call it model Z).

Finally, we use model Z to extract features on Y (just one feedforward, no bp). Then we can get feature file X_Y.csv.

The shape of the data in the file is [n_sample, feature_dim + 1], n_sample is the total number of sample in that domain,
 feature_dim is feature dimensionality (in ResNet-50, it's 2048). The last column denotes the label.

How to use:

If we want to perform transfer from X to Y, we can use the features X_X.csv and X_Y.csv as source and target domain,
respectively.

You are free to finetune other networks (VGG, Inception...) in this way.