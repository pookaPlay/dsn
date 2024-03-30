# dsn

Deep Segmentation Networks

Some of the python dependencies:
- pytorch
- networkx
- matplotlib
- scikit-image

With pytorch and networkx you should be able to:

# Run small synthetic data example

cd src/rbp

python randTrainUnet_V1.py

# Run small BSDS500 data example

Benchmark data: http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz

Extract the BSDS500 benchmark to data/

python randTrainUnet_V2.py

# Experiment scipts

randTrainUnet_V1 :  Purity weighted Connected Components (baseline) using synthetic test (two outputs)

randTrainUnet_V1_Graph :  Purity weighted UNET with one output 

randTrainUnet_V2 :  Purity weighted Connected Components (baseline) using BSDS test    

randTrainUnet_V3 :  (legacy) Purity weighted Watershed Cuts using edge indicators

# Heirarchical Watershed / DSN scripts

trainDSN_V1.py :  Compares rand error estimate from DSN to exhaustive

trainDSN_V2.py : Legacy 

trainDSN_V3.py :  Purity UNET with single output 

trainDSN_V4.py :  Purity UNET with 2 outputs 

# Google doc with random stuff

https://docs.google.com/document/d/1-PvwhlLDm0mQl_IFsn6kUzz98d_JwomlAYCu4wckWUM/edit?usp=sharing
