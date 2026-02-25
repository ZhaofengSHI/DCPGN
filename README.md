# DCPGN
The code of our paper: *Test-time Ego-Exo-centric Adaptation for Action Anticipation via Multi-Label Prototype Growing and Dual-Clue Consistency*, accepted by **CVPR 2026**

## Requirements
Following the environment of EgoExoLearn (action_anticipation_planning_benchmark): https://github.com/OpenGVLab/EgoExoLearn/tree/main/action_anticipation_planning_benchmark

## Data Preparation
We use CLIP (ViT-L-14) to extract video feature for *EgoMe-anti* and *EgoExoLearn* benchmarks at 5FPS. 

You can access the processed video features of the *EgoMe-anti* and *EgoExoLearn* benchmarks by contacting the author by e-mail: zfshi@std.uestc.edu.cn

## Usage
Take the **noun** anticipation under **Ego2Exo** setting on the *EgoExoLearn* benchmark as an example:

Step1: Train the source-view (Ego) model:

```
bash scripts_eel/noun_ego_train_test.sh
```

Step2: Test in the source view (Ego):

```
bash scripts_eel/noun_ego_test_in.sh
```

Step3: Test in the target view (Ego2Exo without adaptation):

```
bash scripts_eel/noun_ego_test_out.sh
```

Step4: Run our DCPGN TTA method (Ego2Exo with DCPGN):

```
bash scripts_eel/noun_ego_test_out_tta.sh
```

## References

**DCPGN method:**

```
(To be published at CVPR 2026) Test-time Ego-Exo-centric Adaptation for Action Anticipation via Multi-Label Prototype Growing and Dual-Clue Consistency 
```

**EgoMe dataset:**

```
@article{qiu2025egome,
  title={EgoMe: A new dataset and challenge for following me via egocentric view in real world},
  author={Qiu, Heqian and Shi, Zhaofeng and Wang, Lanxiao and Xiong, Huiyu and Li, Xiang and Li, Hongliang},
  journal={arXiv preprint arXiv:2501.19061},
  year={2025}
}
```
