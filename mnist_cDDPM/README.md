# MNIST Conditional DDPM (Row-Shift Artifact Removal)

This directory contains a conditional Denoising Diffusion Probabilistic Model (cDDPM) 
designed to remove synthetic row-shift artifacts from MNIST images.

The model takes a corrupted image (with row-wise shift artifacts) as a condition and 
generates a restored image that preserves the digit structure while reducing artifacts.
