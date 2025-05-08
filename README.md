# DINS
This repository is for the paper “Domain-Informed Negative Sampling Strategies for Dynamic Graph Embedding in Meme Stock-Related Social Networks”.
## Introduction
Social network platforms like Reddit are increasingly impacting real-world economics. Meme stocks are a recent phenomena where price movements are driven by retail investors organizing themselves via social networks. To study the impact of social networks on meme stocks, the first step is to analyze these networks. Going forward, predicting meme stocks' returns would require to predict dynamic interactions first. This is different from conventional link prediction, frequently applied in e.g. recommendation systems. For this task, it is essential to predict more complex interaction dynamics, such as the exact timing. These are crucial for linking the network to meme stock price movements. Dynamic graph embedding (DGE) has recently emerged as a promising approach for modeling dynamic graph-structured data. However, current negative sampling strategies, an important component of DGE, are designed for conventional dynamic link prediction and do not capture the specific patterns present in meme stock-related social networks. This limits the training and evaluation of DGE models in such social networks. To overcome this drawback, we propose novel negative sampling strategies based on the analysis of real meme stock-related social networks and financial knowledge. Our experiments show that the proposed negative sampling strategies can better evaluate and train DGE models targeted at meme stock-related social networks compared to existing baselines.
## Running the Experiments
### Environments
The code is tested under the following environment:
- python==3.8.8
- numpy==1.24.3
- pandas==1.1.0
- pytorch==1.13.0
- scikit-learn==0.23.1
- scipy==1.10.1
- tqdm==4.66.5 
### Data processing
To pre-process the data for experiments, please put the original data in folder ProcessData/data (in csv format) and make sure that the csv file meets the following requirements: 
- It should have 5 columns and each row represents a temporal edge
- The 1st column of each row represents the course node id of the corresponding temporal edge
- The 2nd column of each row represents the destination node id of the corresponding temporal edge
- The 3rd column represents the timestamp (in Unix format) of the temporal edge
- The 4th column represents the label of the temporal edge (if the edges do not have label in your dataset, set all to 1)
- The 5th column represents the feature of the temporal edge (if the edges do not have feature in your dataset, set all to 1)
Process the original data by running ProcessData/preprocess_data.py. Three new files will generate.

### Model training
#### TGNs [1]
Train TGNs with our proposed negative sampling strategy, you first need to move the generated files by ProcessData/preprocess_data.py to TGNs/data. Then you can train the model with
```{bash}
python TGNs/DINS.py -d DATASET—NAME --use_memory --valid_index INDEX—START-EDGE—VALIDATION-SET --test_index INDEX—START-EDGE—TEST-SET
```

#### DyGFormer [2]
Train DyGFormer with our proposed negative sampling strategy, you first need to move the generated files by ProcessData/preprocess_data.py to DyGFormerAnd GraphMixer/processed_data/DATASET-NAME. Then you can train the model with
```{bash}
python DyGFormer/DINS.py -d DATASET—NAME --model_name DyGFormer --valid_index INDEX—START-EDGE—VALIDATION-SET --test_index INDEX—START-EDGE—TEST-SET
```

#### GraphMixer [3]
Train GraphMixer with our proposed negative sampling strategy, you first need to move the generated files by ProcessData/preprocess_data.py to DyGFormerAnd GraphMixer/processed_data/DATASET-NAME. Then you can train the model with
```{bash}
python DyGFormer/DINS.py -d DATASET—NAME --model_name GraphMixer --valid_index INDEX—START-EDGE—VALIDATION-SET --test_index INDEX—START-EDGE—TEST-SET
```

## Acknowledgments
We acknowledge the authors of TGNs[1] and DyGFormer[2]. The implementation of mode TGNs is based on the [code](https://github.com/twitter-research/tgn) released by the authors of TGNs and the implementation of mode DyGFormer and GraphMixer is based on the [code](https://github.com/yule-BUAA/DyGLib) released by the authors of DyGFormer.

## Cite us
```bibtex
@inproceedings{hui2025domain,
  title={Domain-Informed Negative Sampling Strategies for Dynamic Graph Embedding in Meme Stock-Related Social Networks},
  author={Hui, Yunming and Zwetsloot, Inez Maria and Trimborn, Simon and Rudinac, Stevan},
  booktitle={Proceedings of the ACM on Web Conference 2025},
  pages={518--529},
  year={2025}
}
```

## Reference
[1] Rossi, Emanuele, et al. "Temporal graph networks for deep learning on dynamic graphs." arXiv preprint arXiv:2006.10637 (2020).

[2] Yu, Le, et al. "Towards better dynamic graph learning: New architecture and unified library." Advances in Neural Information Processing Systems 36 (2023): 67686-67700.

[3] Cong, Weilin, et al. "Do we really need complicated model architectures for temporal networks?." arXiv preprint arXiv:2302.11636 (2023).
