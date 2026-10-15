"""Compact multi-scale SR model and training-only branch alternatives.

Ordinary attention and multi-scale fusion are established ideas.  The
``forward_research`` path supports a controlled error-supervision experiment;
normal inference continues to use ``forward`` and has the original graph.
This is original experimental code, not an MHARN reproduction.
"""
import torch
from torch import nn
from torch.nn import functional as F


class MultiScaleBlock(nn.Module):
    def __init__(self, channels, fusion="learned"):
        super().__init__()
        if fusion not in ("learned", "uniform"):
            raise ValueError("fusion must be learned or uniform")
        self.fusion = fusion
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=d, dilation=d),
                nn.LeakyReLU(0.1),
            ) for d in (1, 2, 3)
        ])
        self.gate = (
            nn.Conv2d(channels, 3, 1) if fusion == "learned" else None
        )
        hidden = max(4, channels // 4)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.project = nn.Conv2d(channels, channels, 1)

    def route(self, x):
        """Return branch features, FP32 gate logits, and normalized weights."""
        branches = tuple(branch(x) for branch in self.branches)
        if self.gate is None:
            logits = None
            weights = x.new_full((x.shape[0], len(branches), x.shape[2], x.shape[3]),
                                 1 / len(branches))
        else:
            # Float32 softmax, even when the convolutions use mixed precision.
            logits = self.gate(x).float()
            weights = logits.softmax(dim=1).to(x.dtype)
        return branches, logits, weights

    def combine(self, x, branches, weights=None, forced_branch=None):
        """Fuse precomputed branches normally or force one training-time route."""
        if forced_branch is not None:
            if not 0 <= forced_branch < len(branches):
                raise ValueError("forced_branch is outside the branch range")
            fused = branches[forced_branch]
        elif weights is None:
            raise ValueError("weights are required for ordinary fusion")
        else:
            fused = sum(weights[:, k:k + 1] * b for k, b in enumerate(branches))
        fused = fused * self.attention(fused)
        return x + 0.2 * self.project(fused)

    def forward(self, x):
        branches, _, weights = self.route(x)
        return self.combine(x, branches, weights)

    def forward_forced(self, x, branch_index):
        """Run exactly one branch for a training/diagnostic alternative."""
        if not 0 <= branch_index < len(self.branches):
            raise ValueError("branch_index is outside the branch range")
        fused = self.branches[branch_index](x)
        fused = fused * self.attention(fused)
        return x + 0.2 * self.project(fused)


class CompactMSAF(nn.Module):
    def __init__(self, scale=4, features=48, blocks=6, fusion="learned"):
        super().__init__()
        if scale not in (2, 4):
            raise ValueError("Supported scales: 2 and 4")
        if features < 4 or blocks < 1:
            raise ValueError("features >= 4 and blocks >= 1 are required")
        self.scale = scale
        self.stem = nn.Conv2d(3, features, 3, padding=1)
        # ModuleList preserves the same state-dict names (blocks.0.*, ...)
        # as the earlier Sequential implementation.
        self.blocks = nn.ModuleList([
            MultiScaleBlock(features, fusion) for _ in range(blocks)
        ])
        self.body = nn.Conv2d(features, features, 3, padding=1)
        up = []
        for _ in range(1 if scale == 2 else 2):
            up.extend([
                nn.Conv2d(features, features * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.1),
            ])
        self.upsample = nn.Sequential(*up)
        self.output = nn.Conv2d(features, 3, 3, padding=1)

    def _reconstruct(self, trunk, shallow, low_resolution):
        deep = shallow + self.body(trunk)
        residual = self.output(self.upsample(deep))
        # A reconstruction skip, not a benchmark LR-generation routine.
        base = F.interpolate(
            low_resolution, scale_factor=self.scale, mode="bicubic",
            align_corners=False,
        )
        return base + residual

    def forward(self, low_resolution):
        shallow = self.stem(low_resolution)
        trunk = shallow
        for block in self.blocks:
            trunk = block(trunk)
        return self._reconstruct(trunk, shallow, low_resolution)

    def forward_research(self, low_resolution, depth=1):
        """Return main SR, forced-route SRs, and suffix gate logits.

        Candidate k forces branch k through every one of the last ``depth``
        blocks.  Gate logits have shape N x depth x 3 x H x W.  Forced
        alternatives reuse all deployed parameters and remain training-only;
        ordinary ``forward`` is unchanged.
        """
        if not isinstance(depth, int) or isinstance(depth, bool):
            raise ValueError("depth must be an integer")
        if not 1 <= depth <= len(self.blocks):
            raise ValueError(f"depth must be between 1 and {len(self.blocks)}")
        suffix = self.blocks[len(self.blocks) - depth:]
        if any(block.gate is None for block in suffix):
            raise ValueError("Research routing requires fusion='learned'.")
        shallow = self.stem(low_resolution)
        shared = shallow
        for block in self.blocks[:len(self.blocks) - depth]:
            shared = block(shared)
        trunk = shared
        gate_logits = []
        for block in suffix:
            branches, logits, weights = block.route(trunk)
            trunk = block.combine(trunk, branches, weights)
            gate_logits.append(logits)
        main = self._reconstruct(trunk, shallow, low_resolution)
        branch_count = len(suffix[0].branches)
        candidates = []
        for branch_index in range(branch_count):
            trunk = shared
            for block in suffix:
                trunk = block.forward_forced(trunk, branch_index)
            candidates.append(self._reconstruct(trunk, shallow, low_resolution))
        return main, tuple(candidates), torch.stack(gate_logits, dim=1)

    def forward_forced_paths(self, low_resolution, depth):
        """Return three diagnostic SRs forced through the last ``depth`` blocks.

        Candidate k selects dilation branch k in every forced block.  The
        ordinary prefix and reconstruction decoder remain shared.  This method
        adds no parameters and is never used by normal ``forward`` inference.
        """
        if not isinstance(depth, int) or isinstance(depth, bool):
            raise ValueError("depth must be an integer")
        if not 1 <= depth <= len(self.blocks):
            raise ValueError(f"depth must be between 1 and {len(self.blocks)}")
        branch_count = len(self.blocks[0].branches)
        if any(len(block.branches) != branch_count for block in self.blocks):
            raise ValueError("All blocks must have the same branch count")
        shallow = self.stem(low_resolution)
        shared = shallow
        split = len(self.blocks) - depth
        for block in self.blocks[:split]:
            shared = block(shared)
        candidates = []
        for branch_index in range(branch_count):
            trunk = shared
            for block in self.blocks[split:]:
                trunk = block.forward_forced(trunk, branch_index)
            candidates.append(self._reconstruct(trunk, shallow, low_resolution))
        return tuple(candidates)
