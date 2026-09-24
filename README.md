# DKNet: Dual-Stream Kinematic Network with Anchored Pooling and Orthogonal Fusion for 3D Human Pose Estimation

> **[DKNet: A Dual-Stream Kinematic Network with Anchored Pooling and Orthogonal Fusion for 3D Human Pose Estimation](https://github.com/zdrmxc/DKNet)**
> Aihua Zhao, Bo Li, Zhixuan Li, Kefan Chen, Longjie Huang, Jiajun Zhang

This repository is the official PyTorch implementation of **DKNet**. DKNet is a lightweight dual-stream kinematic network for monocular 2D-to-3D human pose lifting, achieving state-of-the-art accuracy with only **2.12M parameters** and **37.84 MFLOPs**.

---

## Highlights

* **Kinematic-Anchored Pooling (KAP)**: Constructs a bidirectional kinematic information flow (bottom-up trunk hub aggregation + top-down bone difference vector injection) to actively suppress distal joint distortion and spatial drift under self-occlusion.
* **Kinematic-Hub Orthogonal Fusion (KHOF)**: Employs Gram-Schmidt orthogonalization relative to parent anatomical hubs to strip away parallel translation redundancy, achieving collinearity-free complementary fusion.

---

## Results on Human3.6M

Quantitative evaluation under Protocol #1 (MPJPE) and Protocol #2 (P-MPJPE) on single-frame ($F=1$) input:

| Method | 2D Detector | MPJPE (P1) $\downarrow$ | P-MPJPE (P2) $\downarrow$ | Params | FLOPs |
| --- | --- | --- | --- | --- | --- |
| **DKNet** | 2D Ground Truth (GT) | **30.8 mm** | **24.6 mm** | **2.12 M** | **37.84 M** |
| **DKNet** | CPN (Cascaded Pyramid Network) | **48.2 mm** | **38.1 mm** | **2.12 M** | **37.84 M** |

---

## Dependencies

* Python >= 3.8
* PyTorch >= 1.10.0
* torchvision >= 0.11.0

Install the required packages:

```bash
pip install -r requirements.txt

```

---

## Dataset Setup

Please set up the preprocessed Human3.6M dataset following [VideoPose3D](https://github.com/facebookresearch/VideoPose3D). Place the `.npz` files in the `./dataset` directory as follows:

```bash
${POSE_ROOT}/
|-- dataset
|   |-- data_3d_h36m.npz
|   |-- data_2d_h36m_gt.npz
|   `-- data_2d_h36m_cpn_ft_h36m_dbb.npz

```

---

## Download Pretrained Models

Download the pretrained weights and place them into `./ckpt/pretrained/`:

| Model | 2D Pose Input | MPJPE |
| --- | --- | --- |
| `DKNet_cpn.pth` | CPN | 48.2 mm |
| `DKNet_gt.pth` | Ground Truth | 30.8 mm |

---

## Evaluation

To evaluate the pretrained model on Human3.6M (CPN input, single frame):

```bash
python main.py --reload --previous_dir "./ckpt/pretrained/DKNet_cpn.pth"

```

To evaluate on Ground Truth 2D keypoints:

```bash
python main.py --reload -k gt --previous_dir "./ckpt/pretrained/DKNet_gt.pth"

```

---

## Training

To train **DKNet** on Human3.6M from scratch:

```bash
# Training with CPN 2D detections
python main.py --train --model DKNet --batch_size 512 --lr 0.0005 --nepoch 30 -n cpn

# Training with Ground Truth 2D poses
python main.py --train --model DKNet -k gt --batch_size 512 --lr 0.0005 --nepoch 30 -n gt

```

---
