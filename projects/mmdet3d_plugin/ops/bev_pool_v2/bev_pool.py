# Copyright (c) Phigent Robotics. All rights reserved.

import numpy as np
import torch

from . import bev_pool_v2_ext

__all__ = ['bev_pool_v2', 'TRTBEVPoolv2', 'AXBEVPoolv2', 'ax_bev_pool_v2_maxn']


class QuickCumsumCuda(torch.autograd.Function):
    r"""BEVPoolv2 implementation for Lift-Splat-Shoot view transformation.

    Please refer to the `paper <https://arxiv.org/abs/2211.17111>`_
    """
    @staticmethod
    def forward(ctx, depth, feat, ranks_depth, ranks_feat, ranks_bev,
                bev_feat_shape, interval_starts, interval_lengths):
        ranks_bev = ranks_bev.int()     # (N_points, ),
        depth = depth.contiguous().float()  # (B, N, D, fH, fW)
        feat = feat.contiguous().float()    # (B, N, fH, fW, C)
        ranks_depth = ranks_depth.contiguous().int()    # (N_points, ),
        ranks_feat = ranks_feat.contiguous().int()      # (N_points, ),
        interval_lengths = interval_lengths.contiguous().int()  # (N_pillar, )
        interval_starts = interval_starts.contiguous().int()    # (N_pillar, )

        out = feat.new_zeros(bev_feat_shape)    # (B, D_Z, D_Y, D_X, C)

        bev_pool_v2_ext.bev_pool_v2_forward(
            depth,
            feat,
            out,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_lengths,
            interval_starts,
        )

        ctx.save_for_backward(ranks_bev, depth, feat, ranks_feat, ranks_depth)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        ranks_bev, depth, feat, ranks_feat, ranks_depth = ctx.saved_tensors

        order = ranks_feat.argsort()
        ranks_feat, ranks_depth, ranks_bev = \
            ranks_feat[order], ranks_depth[order], ranks_bev[order]
        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_feat[1:] != ranks_feat[:-1]
        interval_starts_bp = torch.where(kept)[0].int()
        interval_lengths_bp = torch.zeros_like(interval_starts_bp)
        interval_lengths_bp[:-1] = interval_starts_bp[
            1:] - interval_starts_bp[:-1]
        interval_lengths_bp[-1] = ranks_bev.shape[0] - interval_starts_bp[-1]

        depth = depth.contiguous()
        feat = feat.contiguous()
        ranks_depth = ranks_depth.contiguous()
        ranks_feat = ranks_feat.contiguous()
        ranks_bev = ranks_bev.contiguous()
        interval_lengths_bp = interval_lengths_bp.contiguous()
        interval_starts_bp = interval_starts_bp.contiguous()

        depth_grad = depth.new_zeros(depth.shape)
        feat_grad = feat.new_zeros(feat.shape)
        out_grad = out_grad.contiguous()
        bev_pool_v2_ext.bev_pool_v2_backward(
            out_grad,
            depth_grad,
            feat_grad,
            depth,
            feat,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_lengths_bp,
            interval_starts_bp,
        )
        return depth_grad, feat_grad, None, None, None, None, None, \
            None, None, None


def bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                bev_feat_shape, interval_starts, interval_lengths):
    """
    Args:
        depth: (B, N, D, fH, fW)
        feat:  (B, N, fH, fW, C)
        ranks_depth: (N_points, ),
        ranks_feat:  (N_points, ),
        ranks_bev:   (N_points, ),
        bev_feat_shape: (B, D_Z, D_Y, D_X, C)
        interval_starts: (N_pillar, )
        interval_lengths: (N_pillar, )

    Returns:
        x: bev feature in shape (B, C, Dz, Dy, Dx)
    """
    x = QuickCumsumCuda.apply(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                              bev_feat_shape, interval_starts,
                              interval_lengths)      # (B, Dz, Dy, Dx, C)
    x = x.permute(0, 4, 1, 2, 3).contiguous()        # (B, C, Dz, Dy, Dx)
    return x


class AXBEVPoolv2(torch.autograd.Function):

    @staticmethod
    def symbolic(g,
                 depth,
                 feat,
                 ranks_depth,
                 ranks_feat,
                 ranks_bev,
                 n_points, 
                 bev_feat_shape):
        """symbolic function for creating onnx op."""
        return g.op(
            'ax::BEVPoolV2',
            depth,
            feat,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            n_points,
            bev_feat_shape_i=bev_feat_shape)

    @staticmethod
    def forward(g,
                depth,  # B,N,D,H,W
                feat,  # B,N,H,W,C
                ranks_depth,
                ranks_feat,
                ranks_bev,
                n_points,
                bev_feat_shape):
        """run forward."""
        ranks_depth = ranks_depth[:n_points]
        ranks_feat = ranks_feat[:n_points]
        ranks_bev = ranks_bev[:n_points]

        B, N, _, iH, iW = depth.shape
        C = feat.shape[-1]
        _, oD, oW, oH, _ = bev_feat_shape

        # flatten inputs
        depth_1d = depth.flatten()
        feat_2d = feat.reshape(B * N * iH * iW, C)

        # gather depth and feat
        gathered_depth_1d = torch.gather(input=depth_1d, dim=0, index=ranks_depth.long())
        ranks_feat = ranks_feat.reshape(ranks_feat.shape[0], 1).repeat(1, C)
        gathered_feat = torch.gather(input=feat_2d, dim=0, index=ranks_feat.long())

        # subtract zp and mul
        gathered_depth_2d = gathered_depth_1d.reshape(gathered_depth_1d.shape[0], 1)
        r_mul = gathered_depth_2d * gathered_feat

        # init with zeros
        r_scatter = torch.full(fill_value=0, size=(B * oD * oW * oH, C), dtype=torch.float32, device=r_mul.device)

        # scatter_add
        ranks_bev = ranks_bev.reshape(ranks_bev.shape[0], 1).repeat(1, C)
        r_scatter = torch.scatter_add(input=r_scatter, dim=0, index=ranks_bev.long(), src=r_mul)

        # reshape
        r = r_scatter.reshape(B, oD, oW, oH, C)
        return r


def ax_bev_pool_v2_maxn(depth, feat, ranks_depth, ranks_feat, maxn, bev_feat_shape):
    """
    Args:
        depth: (B, N, D, fH, fW)
        feat:  (B, N, fH, fW, C)
        ranks_depth: (D_Z * D_Y * D_X * maxn),
        ranks_feat:  (D_Z * D_Y * D_X * maxn),
        bev_feat_shape: (B, D_Z, D_Y, D_X, C)

    Returns:
        r: bev feature in shape (B, C, Dz, Dy, Dx)
    """
    B, N, D, iH, iW = depth.shape
    C = feat.shape[-1]
    _, oD, oW, oH, _ = bev_feat_shape

    # flatten inputs
    depth_2d = depth.reshape(B * N * D * iH * iW, 1)
    feat_2d = feat.reshape(B * N * iH * iW, C)

    depth_2d = torch.cat((depth_2d, torch.zeros([1, 1], dtype=torch.float32, device=depth_2d.device)), 0)
    feat_2d = torch.cat((feat_2d, torch.zeros([1, 64], dtype=torch.float32, device=feat_2d.device)), 0)

    # gather depth and feat
    # gathered_depth = torch.gather(input=depth_2d, dim=0, index=ranks_depth.long())
    # gathered_feat = torch.gather(input=feat_2d, dim=0, index=ranks_feat.long())
    gathered_depth = depth_2d[ranks_depth.tolist()]
    gathered_feat = feat_2d[ranks_feat.tolist()]

    # subtract zp and mul
    r_mul = gathered_depth * gathered_feat

    # scatter_add
    r_mul = r_mul.reshape(oD, oW, oH, maxn, C)
    r_scatter = r_mul.sum(dim=3, keepdim=True)

    # permute
    r = r_scatter.permute(3, 0, 1, 2, 4)
    return r


class TRTBEVPoolv2(torch.autograd.Function):

    @staticmethod
    def symbolic(g,
                 depth,
                 feat,
                 ranks_depth,
                 ranks_feat,
                 ranks_bev,
                 interval_starts,
                 interval_lengths,
                 output_height=128,
                 output_width=128,
                 output_z=1):
        """symbolic function for creating onnx op."""
        return g.op(
            'mmdeploy::bev_pool_v2',
            depth,
            feat,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_starts,
            interval_lengths,
            output_height_i=output_height,
            output_width_i=output_width,
            output_z_i=output_z)

    @staticmethod
    def forward(g,
                depth,  # N,D,H,W
                feat,  # N,H,W,C
                ranks_depth,
                ranks_feat,
                ranks_bev,
                interval_starts,
                interval_lengths,
                output_height=128,
                output_width=128,
                output_z=1):
        """run forward."""
        feat = feat.unsqueeze(0)
        depth = depth.unsqueeze(0)
        bev_feat_shape = (depth.shape[0], output_z, output_height, output_width,
                          feat.shape[-1])  # (B, Z, Y, X, C)
        bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                               bev_feat_shape, interval_starts,
                               interval_lengths)
        if output_z == 1:
            bev_feat = bev_feat.squeeze(2)
            bev_feat = bev_feat.permute(0, 2, 3, 1)
        return bev_feat


def test_bev_pool_v2():
    depth = np.array([0.3, 0.4, 0.2, 0.1, 0.7, 0.6, 0.8, 0.9])
    depth = torch.from_numpy(depth).float().cuda()
    depth = depth.view(1, 1, 2, 2, 2).requires_grad_()
    feat = torch.ones(
        size=[1, 1, 2, 2, 2], dtype=torch.float,
        device='cuda').requires_grad_()
    ranks_depth = torch.from_numpy(np.array([0, 4, 1, 6])).int().cuda()
    ranks_feat = torch.from_numpy(np.array([0, 0, 1, 2])).int().cuda()
    ranks_bev = torch.from_numpy(np.array([0, 0, 1, 1])).int().cuda()

    kept = torch.ones(
        ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
    kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
    interval_starts = torch.where(kept)[0].int()
    if len(interval_starts) == 0:
        return None, None, None, None, None
    interval_lengths = torch.zeros_like(interval_starts)
    interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
    interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
    bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                           (1, 1, 2, 2, 2), interval_starts, interval_lengths)
    loss = torch.sum(bev_feat)
    loss.backward()
    assert loss == 4.4
    grad_depth = np.array([2., 2., 0., 0., 2., 0., 2., 0.])
    grad_depth = torch.from_numpy(grad_depth).float()
    grad_depth = grad_depth.cuda().view(1, 1, 2, 2, 2)
    assert depth.grad.allclose(grad_depth)
    grad_feat = np.array([1.0, 1.0, 0.4, 0.4, 0.8, 0.8, 0., 0.])
    grad_feat = torch.from_numpy(grad_feat).float().cuda().view(1, 1, 2, 2, 2)
    assert feat.grad.allclose(grad_feat)
