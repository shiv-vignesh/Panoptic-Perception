## Understanding the model architecture

In YOLO-style models (including YOLOv5-inspired ones), the network is usually split into:

```mathematica
Backbone → Neck → Head
```

## 1. Backbone (feature extractor)

**Purpose:** extract increasingly semantic features at multiple scales

**Ends at:** the deepest, lowest-resolution feature map before feature fusion begins

**Layers 0 → 9**

| Layer_id | Module        | Output Shape          | Operation                     | Role                             |
| -------- | ------------- | --------------------- | ----------------------------- | -------------------------------- |
| 0        | Focus         | `[B, 32, 320, 320]` | Spatial slicing + Conv        | Early spatial & edge features    |
| 1        | ConvBlock     | `[B, 64, 160, 160]` | 3×3 Conv, stride 2           | Low-level feature extraction     |
| 2        | BottleneckCSP | `[B, 64, 160, 160]` | CSP residual bottlenecks      | Enhance low-level features       |
| 3        | ConvBlock     | `[B, 128, 80, 80]`  | 3×3 Conv, stride 2           | Mid-level feature extraction     |
| 4        | BottleneckCSP | `[B, 128, 80, 80]`  | CSP residual bottlenecks      | Strengthen mid-level semantics   |
| 5        | ConvBlock     | `[B, 256, 40, 40]`  | 3×3 Conv, stride 2           | High-level feature extraction    |
| 6        | BottleneckCSP | `[B, 256, 40, 40]`  | CSP residual bottlenecks      | Refine high-level features       |
| 7        | ConvBlock     | `[B, 512, 20, 20]`  | 3×3 Conv, stride 2           | Deep semantic feature extraction |
| 8        | SPP           | `[B, 512, 20, 20]`  | Multi-scale pooling           | Enlarge receptive field          |
| 9        | BottleneckCSP | `[B, 512, 20, 20]`  | CSP bottlenecks (no residual) | Deepest semantic features        |

**Layer 9 is the backbone output** (P5-level feature, 20×20).


## 2. Neck (feature aggregation & fusion)

**Purpose:**

* Combine multi-scale features
* Mix semantic (deep) + spatial (shallow) info
* Prepare feature maps for detection at different scales

In YOLOv5 terms: **FPN + PAN**

**Layer 10 is the start of the neck** 

```mathematica

[ConvBlock]
layer_idx=10
in_channels=512
out_channels=256
kernel_size=1

```


### **FPN – Top-Down Path (Semantic → Spatial)**

**Layers: 10 → 17**

| Layer idx | Module           | Output shape         | Operation            | Role                               |
| --------- | ---------------- | -------------------- | -------------------- | ---------------------------------- |
| 10        | ConvBlock (1×1) | `[B, 256, 20, 20]` | Channel reduction    | Prepare deep features for fusion   |
| 11        | Upsample         | `[B, 256, 40, 40]` | ×2 spatial upsample | Align with mid-level features      |
| 12        | Concat           | `[B, 512, 40, 40]` | With layer 6         | Fuse deep + mid features           |
| 13        | BottleneckCSP    | `[B, 256, 40, 40]` | CSP fusion           | Refine fused features              |
| 14        | ConvBlock (1×1) | `[B, 128, 40, 40]` | Channel reduction    | Prep for next scale                |
| 15        | Upsample         | `[B, 128, 80, 80]` | ×2 spatial upsample | Align with shallow features        |
| 16        | Concat           | `[B, 256, 80, 80]` | With layer 4         | Fuse shallow + semantic            |
| 17        | BottleneckCSP    | `[B, 128, 80, 80]` | CSP fusion           | **Small-object feature map** |

**PAN – Bottom-Up Path (Spatial → Semantic)**

**Layers: 18 → 23**


| Layer idx | Module                | Output shape         | Operation     | Role                                |
| --------- | --------------------- | -------------------- | ------------- | ----------------------------------- |
| 18        | ConvBlock (3×3, s=2) | `[B, 128, 40, 40]` | Downsample    | Re-aggregate spatial info           |
| 19        | Concat                | `[B, 256, 40, 40]` | With layer 14 | Merge FPN + PAN features            |
| 20        | BottleneckCSP         | `[B, 256, 40, 40]` | CSP fusion    | **Medium-object feature map** |
| 21        | ConvBlock (3×3, s=2) | `[B, 256, 20, 20]` | Downsample    | Increase semantic strength          |
| 22        | Concat                | `[B, 512, 20, 20]` | With layer 10 | Merge deep + PAN features           |
| 23        | BottleneckCSP         | `[B, 512, 20, 20]` | CSP fusion    | **Large-object feature map**  |


## 3. Head (task-specific prediction)

**Purpose:** turn fused features into bounding boxes & class scores

**Layer 24 only**

```mathematica
[Detect]
layer_idx=24
route=17,20,23

```


| Feature        | Resolution | Source layer |
| -------------- | ---------- | ------------ |
| Small objects  | 80×80     | 17           |
| Medium objects | 40×40     | 20           |
| Large objects  | 20×20     | 23           |
