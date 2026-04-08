# Multi-Task Visual Perception Pipeline

**Course:** DA6401 - Assignment 2  
**Name:** Aaradhya Dashore
**Roll No:** ME22B089


This repository contains a unified, multi-task deep learning architecture built from scratch in PyTorch. The model simultaneously performs breed classification, object localization (bounding box regression), and semantic segmentation (pixel-wise trimap prediction) using a single forward pass on the Oxford-IIIT Pet Dataset.

---

#### Weights & Biases Final Report link:
**https://github.com/AaradhyaDashore/da6401_assignment_2/**

#### Weights & Biases Final Report link:
**https://wandb.ai/aaradhyadashore784-iit-madras/da6401_assignment_2/reports/DA6401-Assignment-2-Report--VmlldzoxNjQ1NTY1NQ?accessToken=paw0fg9t2t9xvvif7hqgvbxjqq02x61f48m57mr2oseglg1v4jiqbci98umtb4sj**


## Architecture Overview

The pipeline leverages a shared convolutional backbone to extract foundational features, which then branch into three specialized task heads:

1. **Shared Backbone:** A VGG11 feature extractor. We experimented with frozen, partially un-frozen, and fully fine-tuned variations to optimize transfer learning.
2. **Classification Head:** A fully connected network outputting probabilities across 37 distinct pet breeds.
3. **Localization Head:** A continuous coordinate regression decoder outputting bounding box coordinates `[xc, yc, w, h]` optimized via a custom IoU loss function.
4. **Segmentation Head:** A symmetric, expansive U-Net style decoder that reconstructs spatial dimensions to predict 3-class trimap masks (Foreground, Background, Boundary) optimized via Dice Loss.
