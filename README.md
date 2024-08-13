# DeepShape

![DeepShape Model](DeepShape.png)

DeepShape is a deep convolutional neural network designed to predict molecular phenotypes from DNA sequences. Unlike traditional models that rely solely on one-hot encoded DNA sequences, DeepShape integrates DNA structural attributes indicative of local shape: minor groove width (MGW), helical twist (HelT), propeller twist (ProT), roll, and electrostatic potential (EP). This combination enhances the interpretability of the model and helps identify regulatory patterns that are not apparent from sequence information alone.

DeepShape is built upon DeeperDeepSEA as implemented in [Selene](https://www.nature.com/articles/s41592-019-0360-8), a PyTorch-based deep learning model originally designed to predict chromatin features from DNA sequence alone.

