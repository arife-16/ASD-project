# ASD-project
Link to colab for adjustments:
https://colab.research.google.com/drive/1CJA_RGKEQ0R5dQKd5EUrnAUF4y3WK_67?usp=sharing

Project's Objectives:
This project investigates alterations in functional connectivity, particularly within the Default Mode Network, in individuals with ASD compared to neurotypical controls using preprocessed fMRI data and deep learning techniques. The project will leverage transfer learning with a ResNet architecture and culminate in the development of a basic analysis tool to illustrate the potential for automated ASD classification.

Exploring Functional Connectivity in Autism Spectrum Disorder Using Convolutional Neural Networks: A Transfer Learning Approach

Abstract

Autism Spectrum Disorder (ASD) is a complex neurodevelopmental condition characterized by deficits in social communication and restricted, repetitive behaviors. Current diagnostic methods rely on subjective behavioral assessments, which are time-consuming, resource-intensive, and prone to variability. This project explores the application of transfer learning with convolutional neural networks (CNNs) to preprocessed functional Magnetic Resonance Imaging (fMRI) data from the Autism Brain Imaging Data Exchange I (ABIDE I) dataset, aiming to identify objective biomarkers for ASD. By leveraging a ResNet architecture, we investigate alterations in functional connectivity, particularly within the Default Mode Network (DMN), to classify individuals with ASD and neurotypical controls. Additionally, we develop a basic analysis tool to demonstrate the potential for translating fMRI-based findings into a practical, clinically relevant application. Our findings highlight the promise of deep learning techniques in improving ASD diagnosis and understanding its neurobiological underpinnings.

ROIs and Atlases: ROIs are specific brain regions that are defined based on anatomical or functional criteria.  Brain atlases (like AAL, Schaefer, Harvard-Oxford, etc.) provide standardized definitions and locations of these ROIs in a standard space.  When calculating FCs, you use an atlas to determine which voxels (3D pixels) in the fMRI data belong to which ROI.  Then, you can calculate the average fMRI signal within each ROI and compute the correlation between the average signals of different ROIs. This correlation value becomes the entry (i, j) in the FC matrix.

Functional Connectivity Matrices (FCs): A very common approach is to compute Functional Connectivity Matrices (FCs) from the raw fMRI data.  These matrices represent the statistical relationships (usually correlations) between the activity of different brain regions.  Each element (i, j) of the FC matrix represents the connectivity strength between brain region i and brain region j.
