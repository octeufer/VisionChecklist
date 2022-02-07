## Checklist for image model test.

This project provide tools to evaluate the performance of image models including Vision Transformer-based model (ViT), and CNN-based model (Resnet, VGG).

There are 3 level of evaluation tools:

### Image level
In this level, the changes in images are associated with domain shifts. E.g. the class domain, the shifts of color and background. Evaluation results are organized as per groups to report the performance of the models on each domain combination.

### Patch level
In this level, the changes in images are associated with each patch manipulation. This is specific to Transformer operations. In Vision Transformer-based model, the images are firstly discretize into patches, so that the patches can formulate a 'sentence' as an input for regular transformer.
The patch level manipulations include: rotation, blur, occlude, shuffle. All operations are based on a probabilistic sampler to randomly select patches for manipulation.

### Pixel level
In this level, the change in images are associate with adversarial attack.

Paper link*[Vision Checklist: Towards Testable Error Analysis of Image Models to Help System Designers Interrogate Model Capabilities](https://arxiv.org/abs/2201.11674)*