Multimodal Image Fusion via Coupled Feature Learning based on Convolutional Sparse Coding
 
%
Run the script files:

1 - script_IV_fusion.m : for infrared-visible image fusion
 
2 - script_medical_grey_image_fusion.m : for greyscale medical image fusion (MR-CT)

3 - script_medical_color_image_fusion.m : for functional anatomical image fusion (MR-PET, MR-SPECT) (this is not included in the paper) 

%
The codes include:

The covolutional coupled feature learning (CCFL) algorithm: ConvCFL.m
The multimodal image fusion algorithm: fuse_grey.m
The algorithm for orthoganl projections (least squared error minimization) on the spase support: sparse_orth_proj.m 
code for generating Gaussian random multiscale dictionaries: initdict.m
code for visualizing multiscale filters: dict2image.m
pre-learned dictionaries (.mat files in dicts folder)

%
The code for lowpass filtering (lowpass.m) is taken from SPORCO toolbox.

%
Medical images are taken from the Whole Brain Atlas database (Harvard medical school)
Infrared-Visible image are taken from https://github.com/hli1221/imagefusion_resnet50/tree/master/IV_images

%
Reference : FG Veshki, SA Vorobyov, Coupled Feature Learning via Structured Convolutional Sparse Coding for Multimodal Image Fusion, ICASSP, May 2022.



Summary:
ADMM-based algorithm for learning correlated features in multimodal images based on convolutional sparse coding with applications to image fusion.
The correlated components are captured using a set of common sparse feature maps (Gamma) and coupled convolutional dictionaries D_i, i= 1,...,n.  
The shared and independent components are represented using a common dictionary (C) and separate sparse feature maps X_i, i= 1,...,n.
The coupled filters (learned correlated features) are fused based on a maximum-variance rule. 
A maximum-absolute-value rule is used to fuse the redundant sparse codes (independent features).