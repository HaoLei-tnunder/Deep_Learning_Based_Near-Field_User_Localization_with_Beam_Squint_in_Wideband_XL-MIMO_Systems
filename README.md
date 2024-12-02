# Code for paper "Deep Learning Based Near-Field User Localization with Beam Squint in Wideband XL-MIMO Systems"

This is a code package related to the following scientific article:

H. Lei, J. Zhang, H. Xiao, D. W. K. Ng, and B. Ai, “Deep Learning Based Near-Field User Localization with Beam Squint in Wideband XL-MIMO Systems,” IEEE Trans. Wireless Commun., to appear, 2024.

The package contains the core part of our algorithm. We encourage you to also perform reproducible research! And feel free to email me for more information!

## Abstract of the article

Extremely large-scale multiple-input multipleoutput (XL-MIMO) is gaining attention as a prominent technology for enabling the sixth-generation (6G) wireless networks. However, the vast antenna array and the huge bandwidth introduce a non-negligible beam squint effect, causing beams of different frequencies to focus at different locations. One approach to cope with this is to employ true-time-delay lines (TTDs)-based beamforming to control the range and trajectory of near-field beam squint, known as the near-field controllable beam squint (CBS) effect. In this paper, we investigate the user localization in near-field wideband XL-MIMO systems under the beam squint effect and spatial non-stationary properties. Firstly, we derive the expressions for Cramer-Rao Bounds (CRBs) for characterizing ´ the performance of estimating both angle and distance. This analysis aims to assess the potential of leveraging CBS for precise user localization. Secondly, a user localization scheme combining CBS and beam training is proposed. Specifically, we organize multiple subcarriers into groups, directing beams from different groups to distinct angles or distances through the CBS to obtain the estimates of users’ angles and distances. Furthermore, we design a user localization scheme based on a convolutional neural network model, namely ConvNeXt. This scheme utilizes the inputs and outputs of the CBS-based scheme to generate high-precision estimates of angle and distance. The numerical results derived from CRBs reveal that the inherent spatial non-stationary characteristics notably increase the CRB for angle, but have an insignificant impact on the CRB for distance estimation. In addition, the CRBs for both angle and distance decrease with increasing bandwidth and number of subcarriers. More importantly, our proposed ConvNeXt-based user localization scheme achieves centimeter-level accuracy in localization estimates..


## License and Referencing

This code package is licensed under the MIT license. If you in any way use this code for research that results in publications, please cite our original article.

[1] H. Lei, J. Zhang, H. Xiao, D. W. K. Ng, and B. Ai, “Deep Learning Based Near-Field User Localization with Beam Squint in Wideband XL-MIMO Systems,” IEEE Trans. Wireless Commun., to appear, 2024.

```
@ARTICLE{[1],
  author={Lei, Hao and Zhang, Jiayi and Xiao, Huahua and Ng, Derrick Wing Kwan and Ai, Bo},
  journal={IEEE Trans. Wireless Commun.},
  title={Deep Learning Based Near-Field User Localization with Beam Squint in Wideband XL-MIMO Systems}, 
  year={to appear, 2024},
  volume={},
  number={},
  pages={1-16},
  doi={10.1109/TWC.2024.3510303}}
```
