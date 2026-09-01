
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    import common.rotation_conversions as rot_conv
except ImportError:
    import sys, os
    sys.path.insert(0, os.environ.get('ROOT_DIR', '.'))
    import common.rotation_conversions as rot_conv


NUM_VERTICES = 778




def aa_to_rot6d(motion: torch.Tensor) -> torch.Tensor:
    T, _ = motion.shape
    m = motion.reshape(B * T, 61)

    orient_mat = rot_conv.axis_angle_to_matrix(m[:, :3])
    orient_6d = orient_mat[:, :, :2].permute(0, 2, 1).reshape(B * T, 6)

    pose_aa = m[:, 3:48].reshape(B * T * 15, 3)
    pose_mat = rot_conv.axis_angle_to_matrix(pose_aa)
    pose_6d = pose_mat[:, :, :2].permute(0, 2, 1).reshape(B * T, 90)

    cam_t = m[:, 58:61]
    result = torch.cat([orient_6d, pose_6d, cam_t], dim=-1)
    return result.reshape(B, T, 99)


def rot6d_to_aa(motion_6d: torch.Tensor) -> torch.Tensor:
    B, T, _ = motion_6d.shape
    m = motion_6d.reshape(B * T, 99)

    def _rot6d_to_matrix(x6d):
        a1, a2 = x6d[:, :3], x6d[:, 3:6]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack([b1, b2, b3], dim=-1)

    orient_mat = _rot6d_to_matrix(m[:, :6])
    orient_aa = rot_conv.matrix_to_axis_angle(orient_mat)

    pose_6d = m[:, 6:96].reshape(B * T * 15, 6)
    pose_mat = _rot6d_to_matrix(pose_6d)
    pose_aa = rot_conv.matrix_to_axis_angle(pose_mat).reshape(B * T, 45)

    cam_t = m[:, 96:99]
    betas = torch.zeros(B * T, 10, device=motion_6d.device, dtype=motion_6d.dtype)
    return torch.cat([orient_aa, pose_aa, betas, cam_t], dim=-1).reshape(B, T, 61)



@torch.no_grad()
def compute_vertex_marginals(loader, device, num_vertices: int = NUM_VERTICES,
                              clamp_min: float = 1e-3, clamp_max: float = 1.0 - 1e-3):
    sum_contacts = torch.zeros(num_vertices, device=device)
    n_valid = torch.zeros(1, device=device)

    for batch in tqdm(loader, desc='Computing vertex marginals'):
        c = batch['contact_map'].float().to(device)
        mask = batch['valid_mask'].float().to(device)
        mask_3d = mask.unsqueeze(-1).expand_as(c)
        sum_contacts += (c * mask_3d).sum(dim=(0, 1))
        n_valid += mask.sum()

    p_v = sum_contacts / n_valid.clamp(min=1.0)
    return p_v.clamp(clamp_min, clamp_max).cpu()




def extract_mano_edges_from_faces(faces) -> torch.Tensor:
    
    if not isinstance(faces, torch.Tensor):
        faces = torch.tensor(faces, dtype=torch.long)
    
    edges_set = set()
    for face in faces.tolist():
        for i in range(3):
            a, b = face[i], face[(i + 1) % 3]
            if a != b:
                edges_set.add((min(a, b), max(a, b)))
    
    edges = torch.tensor(list(edges_set), dtype=torch.long).t()  # (2, E)
    return torch.cat([edges, edges.flip(0)], dim=1)              # (2, 2 * E)