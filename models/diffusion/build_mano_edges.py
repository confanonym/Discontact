

import os
import torch
import torch.nn.functional as F
import smplx

from models.diffusion.utils import extract_mano_edges_from_faces

NUM_VERTICES = 778
OUTPUT_DIR = os.environ.get('ROOT_DIR', '.')


def _basic_finger_map():
   
    return torch.zeros(NUM_VERTICES, dtype=torch.long)


def main():
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
        template_verts = out.vertices[0]  # (778, 3)
    
    print(f'Faces : {faces.shape} | Template verts : {template_verts.shape}')
    
    # === Edge index ===
    edge_index = extract_mano_edges_from_faces(faces)
    print(f'edge_index : {edge_index.shape}')
    
    out_idx = os.path.join(OUTPUT_DIR, 'mano_edge_index.pt')
    torch.save(edge_index, out_idx)
    print(f'Saved -> {out_idx}')
    
    
    src, dst = edge_index[0], edge_index[1]
    
    euc = (template_verts[src] - template_verts[dst]).norm(dim=-1, keepdim=True)  # (E, 1)
    euc = (euc - euc.mean()) / (euc.std() + 1e-6)
    
    finger_map = _basic_finger_map()
    finger_dst = finger_map[dst]
    finger_oh = F.one_hot(finger_dst, num_classes=6).float()  # (E, 6)
    same_finger = (finger_map[src] == finger_map[dst]).float().unsqueeze(-1)  # (E, 1)
    
    edge_features = torch.cat([euc, same_finger, finger_oh], dim=-1)  # (E, 8)
    print(f'edge_features : {edge_features.shape}')
    
    out_feat = os.path.join(OUTPUT_DIR, 'mano_edge_features.pt')
    torch.save(edge_features, out_feat)
    print(f'Saved -> {out_feat}')


if __name__ == '__main__':
    main()