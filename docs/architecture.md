# System Architecture & Technical Specification

This document provides an engineering-level deep dive into the **Multi-Scale Attention Feature Fusion Network (MSAF-Net / CompactMSAF)** for $4\times$ Single-Image Super-Resolution (SISR).

---

## 1. End-to-End Pipeline & Tensor Flow

```mermaid
graph TD
    A[Low-Resolution RGB Image: B x 3 x H x W] --> B[Stem Convolution 3x3, 3 -> 48]
    A --> G[Bicubic Upsampling x4: B x 3 x 4H x 4W]
    
    subgraph Trunk ["Deep Feature Extraction Trunk (6 MultiScaleBlocks)"]
        B --> C1[Block 1]
        C1 --> C2[Block 2]
        C2 --> C3[Block 3]
        C3 --> C4[Block 4]
        C4 --> C5[Block 5]
        C5 --> C6[Block 6]
    end
    
    C6 --> D[Body Convolution 3x3, 48 -> 48]
    B -.->|Long Residual Skip Connection| D_Sum[Element-wise Sum]
    D --> D_Sum
    
    subgraph Reconstruction ["Sub-Pixel Upsampling & Reconstruction"]
        D_Sum --> E1[Conv 3x3, 48 -> 192]
        E1 --> E2[PixelShuffle x2 + LeakyReLU: B x 48 x 2H x 2W]
        E2 --> E3[Conv 3x3, 48 -> 192]
        E3 --> E4[PixelShuffle x2 + LeakyReLU: B x 48 x 4H x 4W]
        E4 --> F[RGB Projection Head Conv 3x3, 48 -> 3]
    end
    
    F --> H[Global Residual Sum]
    G --> H
    H --> I[Super-Resolved Output: B x 3 x 4H x 4W]

    style Trunk fill:#f0f7ff,stroke:#0366d6,stroke-width:2px
    style Reconstruction fill:#f6ffed,stroke:#52c41a,stroke-width:2px
```

---

## 2. MultiScaleBlock Architecture

Each block within the trunk extracts multi-scale representations through parallel dilated convolutions, fuses them using dynamic soft spatial gating, recalibrates features via channel attention, and applies residual scaling:

```mermaid
graph LR
    In[Input Feature F: B x 48 x H x W] --> B1[Conv 3x3, Dilation=1]
    In --> B2[Conv 3x3, Dilation=2]
    In --> B3[Conv 3x3, Dilation=3]
    In --> Gate[Gate: 1x1 Conv -> 3 Logits]
    
    Gate --> Softmax[Spatial Softmax over Branches]
    
    B1 --> WeightedSum[Weighted Spatial Feature Sum M]
    B2 --> WeightedSum
    B3 --> WeightedSum
    Softmax --> WeightedSum
    
    subgraph CA ["Squeeze-and-Excitation Channel Attention"]
        WeightedSum --> GAP[Global Avg Pool 1x1]
        GAP --> CA_Conv1[Conv 1x1, 48 -> 12 + ReLU]
        CA_Conv1 --> CA_Conv2[Conv 1x1, 12 -> 48 + Sigmoid]
    end
    
    WeightedSum --> Mult[Channel-wise Multiply]
    CA_Conv2 --> Mult
    
    Mult --> Proj[1x1 Projection Conv]
    Proj --> Scale[Residual Scale factor 0.2]
    In -.->|Identity Residual| Add[Element-wise Add]
    Scale --> Add
    Add --> Out[Block Output: B x 48 x H x W]

    style CA fill:#fffbe6,stroke:#faad14,stroke-width:1px
```

### Mathematical Formulation

1. **Multi-Scale Convolutional Sampling:**
   $$\mathbf{B}_k = \text{LeakyReLU}\left(\text{Conv}_{3\times3, d_k}(\mathbf{F})\right), \quad d_k \in \{1, 2, 3\}$$
   - $d_1 = 1$: Standard dense $3\times3$ spatial receptive field.
   - $d_2 = 2$: Dilated $5\times5$ effective receptive field ($3\times3$ sparse grid with rate 2).
   - $d_3 = 3$: Dilated $7\times7$ effective receptive field ($3\times3$ sparse grid with rate 3).

2. **Position-Wise Soft Attention Gating:**
   $$\mathbf{g}(\mathbf{F}) = \text{Conv}_{1\times1}(\mathbf{F}) \in \mathbb{R}^{3 \times H \times W}$$
   $$w_k(p) = \frac{\exp(g_k(p))}{\sum_{j=1}^3 \exp(g_j(p))}, \quad \forall p \in \Omega$$
   $$\mathbf{M}(p) = \sum_{k=1}^3 w_k(p) \cdot \mathbf{B}_k(p)$$

3. **Channel Attention Recalibration:**
   $$\mathbf{s} = \sigma\left(\mathbf{W}_2 \cdot \text{ReLU}\left(\mathbf{W}_1 \cdot \text{GAP}(\mathbf{M})\right)\right)$$
   $$\mathbf{M}_{\text{att}} = \mathbf{s} \odot \mathbf{M}$$

4. **Residual Scaling:**
   $$\mathbf{F}_{\text{out}} = \mathbf{F} + 0.2 \cdot \mathbf{W}_{\text{proj}} \mathbf{M}_{\text{att}}$$

---

## 3. Parameter and Computational Complexity

| Component | Operation | Kernel / Configuration | Parameters |
|---|---|---|---|
| Shallow Stem | Conv2d | $3 \to 48$, kernel $3\times3$ | 1,344 |
| MultiScaleBlock $\times 6$ | 3 Dilation Branches | $3 \times (48 \to 48)$, kernel $3\times3$ | $6 \times 62,352 = 374,112$ |
| | Soft Attention Gate | $48 \to 3$, kernel $1\times1$ | $6 \times 147 = 882$ |
| | Channel Attention | $48 \to 12 \to 48$, $1\times1$ | $6 \times 1,200 = 7,200$ |
| | Projection Conv | $48 \to 48$, kernel $1\times1$ | $6 \times 2,352 = 14,112$ |
| Body Conv | Conv2d | $48 \to 48$, kernel $3\times3$ | 20,784 |
| Upsampler 1 | Conv2d + PixelShuffle(2) | $48 \to 192$, kernel $3\times3$ | 83,136 |
| Upsampler 2 | Conv2d + PixelShuffle(2) | $48 \to 192$, kernel $3\times3$ | 83,136 |
| Head Conv | Conv2d | $48 \to 3$, kernel $3\times3$ | 1,302 |
| **Total Parameter Count** | — | — | **586,077** |

---

## 4. Controlled Research Extension: Error-Supervised Branch Routing

In addition to standard inference, `research.py` implements a training-only error supervision mechanism:
- Branch candidates $k \in \{1, 2, 3\}$ are evaluated through the downstream trunk and decoder.
- Local reconstruction error $e_k$ is computed against the ground-truth high-resolution patch:
  $$e_k = \text{AvgPool}_{3\times3}\left(\text{Downsample}_{4\times}\left(|I_k^{\text{SR}} - I^{\text{HR}}|\right)\right)$$
- Softmax error preference targets are formed using detached gradients:
  $$q_k = \text{Softmax}\left(-\frac{\text{detach}(e_k)}{\tau \cdot \text{detach}(\bar{e}) + \epsilon}\right)$$
- The spatial gate is supervised by minimizing cross-entropy with target $q_k$, encouraging spatial specialization without changing the final inference architecture or parameter count.
