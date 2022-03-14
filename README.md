# StageD_HF

This is an implementation of a simple feedforward neural network using PyTorch lightning, designed to identify whether an individual has Stage C or Stage D HF, based on established risk factors collectively found in three risk scores (Seattle Heart Failure Model; MAGGIC, Meta-Analysis Global Group in Chronic Heart Failure Risk Calculator; MARKER-HF, Machine learning Assessment of RisK and EaRly mortality in Heart Failure) along with expert concensus. 

This model was trained on private health information such as diagnoses, treatment information, medical test results, and prescription information that is protected under HIPAA, but no patient data is shared in this public repo. This repo is only meant to share the code underlying the models. 
