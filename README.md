# MixVPR: Feature Mixing for Visual Place Recognition

### Summary
This repo contains the code for using MixVPR for Sequence-based VPR. It uses late aggregation of single-image MixVPR descriptors on Mapillary Street-Level Sequences Dataset. 

``` bash 
python seq2seq.py \
    --config_path config.ini
    --dataset_path <MSLS val or test set path>
    --trainset_path <MSLS train set path>
    --pca_dim 4096
```
`dataset_path` should have the city-wise database and query sequences from the val/test set split on which you want to evaluate.  
`trainset_path` should have the database and query sequences of the entire train set. This is needed if you want to apply PCA on the sequence descriptors obtained after late aggregation (concatenation).  
For constructing sequences from the MSLS dataset, please refer to - https://github.com/vandal-vpr/vg-transformers/tree/main/main_scripts/msls  

This approach gives R@1 of 90.48 and R@5 of 93.08 on the official MSLS validation set.  

### References
We thank the authors of the following repositories for their open source code:
 * [[amaralibey/MixVPR](https://github.com/amaralibey/MixVPR)]
 * [[vandal-vpr/vg-transformers](https://github.com/vandal-vpr/vg-transformers)]