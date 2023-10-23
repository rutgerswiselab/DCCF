# DCCF

## Introduction
This repository includes the implementation for Deconfounded Causal Collaborative Filtering

> Paper: Deconfounded Causal Collaborative Filtering <br>
> Paper Link: [https://dl.acm.org/doi/full/10.1145/3606035](https://dl.acm.org/doi/full/10.1145/3606035)

## Environment

Environment requirements can be found in `./requirement.txt`

## Datasets
  
- **Electronics** and **CDs and Vinyl**: The origin dataset can be found [here](https://nijianmo.github.io/amazon/index.html.). 

- **Yelp**: The origin dataset can be found [here](https://www.yelp.com/dataset).

- The data processing code can be found in `./src/data_preprocessing/'

## Example to run the codes

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
