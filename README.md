# Deconfounded Causal Collaborative Filtering

## Introduction
This repository includes the implementation for Deconfounded Causal Collaborative Filtering

> Paper: Deconfounded Causal Collaborative Filtering <br>
> Paper Link: [https://dl.acm.org/doi/full/10.1145/3606035](https://dl.acm.org/doi/full/10.1145/3606035)

## Environment

Environment requirements can be found in `./requirement.txt`

## Datasets
  
- **Electronics** and **CDs and Vinyl**: The origin dataset can be found [here](https://nijianmo.github.io/amazon/index.html). 

- **Yelp**: The origin dataset can be found [here](https://www.yelp.com/dataset).

- The data processing code can be found in `./src/data_preprocessing/`

## Generate feature_embedding

The feature_embedding is generated by a pre-trained sentence embedding models. We applied the pre-trained paraphrase-distilroberta-base-v1 sentence embedding model in a public transformer implementation: https://github.com/UKPLab/sentence-transformers

Specifically, we take the average of embedding of the 'title', 'description' and 'feature' as feature_embedding. We used pre-trained model to encode sentences separately and manually compute the average.

## Generate Exposure Probability

To generate the exposure probability in ips_expo_prob.npy, we train an IPSBiasedMF model and save the full predicted user-item matrix as the exposure probability. 

## Example to run the codes

After generating feature_embedding and exposure probability, we run the code to train DCCF.

For example:

```
# DCCF on Electronics dataset
> cd ./src/
> python ./main.py --rank 1 --model_name DCCF --optimizer Adam --lr 0.001 --dataset Electronics --metric ndcg@5,recall@5,precision@5 --gpu 0 --epoch 100 --test_neg_n 1000
```

## Citation

```
@article{xu2023deconfounded,
  title={Deconfounded causal collaborative filtering},
  author={Xu, Shuyuan and Tan, Juntao and Heinecke, Shelby and Li, Vena Jia and Zhang, Yongfeng},
  journal={ACM Transactions on Recommender Systems},
  volume={1},
  number={4},
  pages={1--25},
  year={2023},
  publisher={ACM New York, NY}
}
```
