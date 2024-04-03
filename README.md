# TTHF
The code of [Text-Driven Traffic Anomaly Detection with Temporal High-Frequency Modeling in Driving Videos]
Rongqin Liang, Yuanman Li, Jiantao Zhou, and Xia Li

Our STGlow network architecture:

<img src="TTHF.png" width="1000">

## Installation
### Dependencies
 - Python 3.8
 - pytorch 1.11.0
 - cuda 11.3
 - Ubuntu 20.04
 - RTX 3090
 - Please refer to the "requirements.txt" file for more details.

## Training
Users can train the STGlow models on ETH/UCY or SDD dataset easily by runing the following command:

For ETH/UCY:
```
python tools/train_for_eth_ucy.py 
```

For SDD:
```
python tools/train_for_sdd.py 
```

## Inference 
Users can test the STGlow models on ETH/UCY or SDD dataset easily by runing the following command:

For ETH/UCY:
```
python tools/test_for_eth_ucy.py 
```

For SDD:
```
python tools/test_for_sdd.py 
```

Note that our project is developed based on the [code](https://github.com/umautobots/bidireaction-trajectory-prediction) of [BiTraP: Bi-directional Pedestrian Trajectory Prediction with Multi-modal Goal Estimation](https://arxiv.org/abs/2007.14558).

## Citation

If you found the repo is useful, please feel free to cite our papers:
```
@article{liang2022stglow,
      title={STGlow: A Flow-based Generative Framework with Dual Graphormer for Pedestrian Trajectory Prediction}, 
      author={Rongqin Liang and Yuanman Li and Jiantao Zhou and Xia Li},
      journal={arXiv preprint arXiv:2211.11220}
      year={2022}
}

```
