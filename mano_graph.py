"""
build_mano_edges.py — one-shot precomputation of the MANO mesh structures.

Produces:
    - mano_edge_index.pt    : (2, E) bidirectional mesh edges
    - mano_edge_features.pt : (E, 8) static per-edge features
    - mano_region_map.pt    : (778,) anatomical region per vertex (0-5)

Anatomical regions (from the MANO skinning weights):
    0 : palm / wrist
    1 : index
    2 : middle
    3 : ring
    4 : little
    5 : thumb

The MANO skinning weights officially define which vertices belong to which
joint — the assignment is fixed, deterministic and anatomically correct.
Each vertex is assigned to the region of the joint that influences it most.

Usage:
    python models/diffusion/build_mano_edges.py

    # With a visualization to check the regions:
    python models/diffusion/build_mano_edges.py --viz
"""

import os
import argparse
import torch
import torch.nn.functional as F
import smplx
import numpy as np

from models.diffusion.utils import extract_mano_edges_from_faces
from matplotlib.patches import Patch
NUM_VERTICES = 778
NUM_REGIONS  = 6
OUTPUT_DIR   = os.environ.get('ROOT_DIR', '.')

REGION_NAMES = ['Palm', 'Index', 'Middle', 'Ring', 'Little', 'Thumb']
REGION_COLORS = [
    [0.7, 0.7, 0.7],   # 0 palm    grey
    [1.0, 0.2, 0.2],   # 1 index   red
    [0.2, 0.8, 0.2],   # 2 middle  green
    [0.2, 0.2, 1.0],   # 3 ring    blue
    [1.0, 0.8, 0.0],   # 4 little  yellow
    [0.8, 0.2, 0.8],   # 5 thumb   purple
]


# Region map

def build_region_map(mano_layer):
    """
    Assign each MANO vertex to an anatomical region using the skinning
    weights (lbs_weights).

    mano_layer.lbs_weights : (778, J) influence weight of each joint.
    A vertex is assigned to the joint that influences it most (argmax).

    MANO with use_pca=False has 15 pose joints + 1 global = 16 lbs joints:
        0        : wrist (palm)
        1, 2, 3  : index  (MCP, PIP, DIP)
        4, 5, 6  : middle
        7, 8, 9  : ring
        10,11,12 : little
        13,14,15 : thumb

    Note: fingertips have no separate lbs joint in MANO — they are captured
    by the DIP (last joint of the finger).
    """
    lbs = mano_layer.lbs_weights   # (778, 16) tensor or numpy

    if not isinstance(lbs, torch.Tensor):
        lbs = torch.tensor(lbs, dtype=torch.float32)

    print(f'  lbs_weights shape : {lbs.shape}')

    # lbs joint -> anatomical region mapping
    # MANO 16 joints: 0=wrist, 1-3=index, 4-6=middle,
    #                 7-9=ring, 10-12=little, 13-15=thumb
    n_joints = lbs.shape[1]

    if n_joints == 16:
        joint_to_region = torch.tensor([
            0,            # 0  wrist -> palm
            1, 1, 1,      # 1-3   index
            2, 2, 2,      # 4-6   middle
            3, 3, 3,      # 7-9   ring
            4, 4, 4,      # 10-12 little
            5, 5, 5,      # 13-15 thumb
        ], dtype=torch.long)
    elif n_joints == 21:
        # Some MANO variants expose 21 joints
        joint_to_region = torch.tensor([
            0,                  # 0     wrist
            1, 1, 1, 1,         # 1-4   index
            2, 2, 2, 2,         # 5-8   middle
            3, 3, 3, 3,         # 9-12  ring
            4, 4, 4, 4,         # 13-16 little
            5, 5, 5, 5,         # 17-20 thumb
        ], dtype=torch.long)
    else:
        raise ValueError(f'lbs_weights has {n_joints} joints — unsupported. '
                         f'Expected 16 or 21.')

    # For each vertex: the joint with the largest influence weight
    dominant_joint = lbs.argmax(dim=-1)                # (778,)
    region_map     = joint_to_region[dominant_joint]   # (778,)

    return region_map


def validate_region_map(region_map, template_verts):
    """Print per-region statistics and check consistency."""
    print('\n  Anatomical regions (MANO skinning weights):')
    print(f'  {"Region":<15} {"Vertices":>12} {"% total":>8}')
    print('  ' + '-' * 38)

    for r in range(NUM_REGIONS):
        idx = (region_map == r).nonzero().squeeze()
        n   = idx.shape[0] if idx.dim() > 0 else 1
        pct = 100.0 * n / NUM_VERTICES
        # Mean position of the region vertices
        if idx.dim() > 0 and n > 0:
            center = template_verts[idx].mean(dim=0) * 100   # in cm
            center_str = f'({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) cm'
        else:
            center_str = 'N/A'
        print(f'  {REGION_NAMES[r]:<15} {n:>12} {pct:>7.1f}%   center={center_str}')

    # Check that every vertex is assigned
    assert region_map.shape == (NUM_VERTICES,), \
        f'region_map shape {region_map.shape} != ({NUM_VERTICES},)'
    assert region_map.min() >= 0 and region_map.max() < NUM_REGIONS, \
        f'region_map values out of range [0, {NUM_REGIONS-1}]'
    print(f'\n  OK — all {NUM_VERTICES} vertices assigned to a region')


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_regions(region_map, template_verts, faces, save_path):
    """Save a 3D matplotlib image of the anatomical regions."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        verts_np = template_verts.numpy() * 100   # in cm
        faces_np = faces.numpy()
        colors_per_face = []

        for face in faces_np:
            # Face color = color of the region of its first vertex
            r = region_map[face[0]].item()
            colors_per_face.append(REGION_COLORS[r] + [0.8])

        fig = plt.figure(figsize=(10, 8))
        ax  = fig.add_subplot(111, projection='3d')

        poly = Poly3DCollection(
            verts_np[faces_np],
            facecolors=colors_per_face,
            linewidths=0.1,
            edgecolors='none',
        )
        ax.add_collection3d(poly)

      
      
        legend_elements = [
            Patch(facecolor=REGION_COLORS[r], label=REGION_NAMES[r])
            for r in range(NUM_REGIONS)
        ]
        ax.legend(handles=legend_elements, loc='upper left')

        # Axes
        ax.set_xlim(verts_np[:, 0].min(), verts_np[:, 0].max())
        ax.set_ylim(verts_np[:, 1].min(), verts_np[:, 1].max())
        ax.set_zlim(verts_np[:, 2].min(), verts_np[:, 2].max())
        ax.set_title('MANO anatomical regions (skinning weights)')
        ax.set_xlabel('X (cm)'); ax.set_ylabel('Y (cm)'); ax.set_zlabel('Z (cm)')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f'  -> Visualization: {save_path}')

    except Exception as e:
        print(f'  Visualization skipped: {e}')



# Main


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--viz', action='store_true',
                   help='Save a 3D visualization of the anatomical regions')
    args = p.parse_args()

    mano_path = os.environ['MANO_PATH']
    print(f'Loading MANO from {mano_path}')

    mano_layer = smplx.MANO(mano_path, use_pca=False, flat_hand_mean=True)
    faces = torch.tensor(mano_layer.faces.astype('int64'), dtype=torch.long)

    with torch.no_grad():
        out = mano_layer(
            global_orient=torch.zeros(1, 3),
            hand_pose=torch.zeros(1, 45),
            betas=torch.zeros(1, 10),
            transl=torch.zeros(1, 3),
        )
        template_verts = out.vertices[0]   # (778, 3)

    print(f'Faces: {faces.shape} | Template verts: {template_verts.shape}')

    # === 1. Edge index ===
    print('\n[1/3] Edge index...')
    edge_index = extract_mano_edges_from_faces(faces)
    print(f'  edge_index : {edge_index.shape}')

    out_idx = os.path.join(OUTPUT_DIR, 'mano_edge_index.pt')
    torch.save(edge_index, out_idx)
    print(f'  Saved -> {out_idx}')

    # === 2. Region map (skinning weights) ===
    print('\n[2/3] Region map (skinning weights)...')
    region_map = build_region_map(mano_layer)
    validate_region_map(region_map, template_verts)

    out_region = os.path.join(OUTPUT_DIR, 'mano_region_map.pt')
    torch.save(region_map, out_region)
    print(f'  Saved -> {out_region}')

    if args.viz:
        visualize_regions(
            region_map, template_verts, faces,
            save_path=os.path.join(OUTPUT_DIR, 'mano_regions.png'),
        )

    # === 3. Edge features (with the actual region information) ===
    print('\n[3/3] Edge features...')
    src, dst = edge_index[0], edge_index[1]

    # Normalized euclidean distance in the template pose
    euc = (template_verts[src] - template_verts[dst]).norm(dim=-1, keepdim=True)
    euc = (euc - euc.mean()) / (euc.std() + 1e-6)   # (E, 1)

    # Per-vertex region (from skinning weights — no placeholder)
    region_src = region_map[src]   # (E,)
    region_dst = region_map[dst]   # (E,)

    # same_region: 1 if both endpoints lie in the same anatomical region
    same_region = (region_src == region_dst).float().unsqueeze(-1)   # (E, 1)

    # One-hot of the destination region
    region_oh = F.one_hot(region_dst, num_classes=NUM_REGIONS).float()   # (E, 6)

    edge_features = torch.cat([euc, same_region, region_oh], dim=-1)   # (E, 8)
    print(f'  edge_features : {edge_features.shape}')
    print(f'  same_region edges : {same_region.sum().int().item()} / {edge_index.shape[1]}')

    out_feat = os.path.join(OUTPUT_DIR, 'mano_edge_features.pt')
    torch.save(edge_features, out_feat)
    print(f'  Saved -> {out_feat}')

    print('\n' + '=' * 50)
    print('Done.')
    print(f'  mano_edge_index.pt   : {edge_index.shape}')
    print(f'  mano_region_map.pt   : {region_map.shape}')
    print(f'  mano_edge_features.pt: {edge_features.shape}')
    print('=' * 50)


if __name__ == '__main__':
    main()