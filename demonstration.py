

import os
import json
import pickle
from os.path import join as pjoin

import numpy as np
import torch
import imageio
from PIL import Image, ImageDraw

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import common.rotation_conversions as rot_conv
from common.renderer import Renderer, get_text_image
from models.mano.wrapper import MANO, LIGHT_BLUE

from models.diffusion.denoiser import ContactDenoiser
from models.diffusion.diffusion import HybridDDPM, fk_verts_and_joints
from models.diffusion.utils import rot6d_to_aa
from options.discontact_option import DemoDisContactOptions




_VERBS = {
    'open', 'close', 'pick', 'place', 'put', 'take', 'grab', 'hold', 'push',
    'pull', 'turn', 'rotate', 'press', 'lift', 'move', 'insert', 'remove',
    'attach', 'detach', 'screw', 'unscrew', 'pour', 'cut', 'wipe', 'clean',
    'adjust', 'align', 'connect', 'disconnect', 'tighten', 'loosen', 'up',
    'down', 'on', 'off', 'the', 'a', 'an', 'with', 'to', 'from', 'in', 'out',
}


def text_to_query(text):
    """'open cap' -> 'cap'; 'pick up the red mug' -> 'red mug'."""
    toks = [w for w in text.lower().replace(',', ' ').split() if w]
    nouns = [w for w in toks if w not in _VERBS]
    return ' '.join(nouns) if nouns else toks[-1]


def detect_contact_pixel(img, query, args, device):
    from transformers import AutoProcessor, Owlv2ForObjectDetection

    print(f'  detection query: "{query}"')
    processor = AutoProcessor.from_pretrained(args.det_model)
    model = Owlv2ForObjectDetection.from_pretrained(args.det_model).to(device).eval()

    prompts = [[f'a photo of a {query}', query]]
    inputs = processor(text=prompts, images=img, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    target = torch.tensor([[img.size[1], img.size[0]]], device=device)

    post_fn = getattr(processor, 'post_process_grounded_object_detection',
                      None) or processor.post_process_object_detection
    res = post_fn(outputs=outputs, target_sizes=target, threshold=args.det_thr)[0]

    del model
    torch.cuda.empty_cache()

    if len(res['scores']) == 0:
        u, v = img.size[0] // 2, img.size[1] // 2
        print(f'  NO detection (threshold {args.det_thr}) -> image center ({u}, {v})')
        return u, v, None, 0.0

    best = int(res['scores'].argmax())
    box = [float(x) for x in res['boxes'][best].tolist()]
    score = float(res['scores'][best])
    u = int(round((box[0] + box[2]) / 2))
    v = int(round((box[1] + box[3]) / 2))
    print(f'  box {[round(b) for b in box]} | score {score:.3f} -> center ({u}, {v})')
    return u, v, box, score


def refine_with_depth(u, v, box, depth, mode='center'):
    H, W = depth.shape
    valid = np.isfinite(depth) & (depth > 1e-3)

    if box is not None and mode == 'near':
        x0, y0, x1, y1 = [int(round(c)) for c in box]
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, W), min(y1, H)
        sub = depth[y0:y1, x0:x1].copy()
        sub_valid = valid[y0:y1, x0:x1]
        if sub_valid.any():
            sub[~sub_valid] = np.inf
            j, i = np.unravel_index(np.argmin(sub), sub.shape)
            return x0 + int(i), y0 + int(j)

    u = int(np.clip(u, 0, W - 1))
    v = int(np.clip(v, 0, H - 1))
    if valid[v, u]:
        return u, v

    # nearest valid pixel
    ys, xs = np.nonzero(valid)
    if len(ys) == 0:
        return u, v
    k = int(np.argmin((xs - u) ** 2 + (ys - v) ** 2))
    print(f'  invalid depth at ({u}, {v}) -> ({xs[k]}, {ys[k]})')
    return int(xs[k]), int(ys[k])




def unproject_depth_to_3d(depth_map, K):
    H, W = depth_map.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    X = (u - cx) * depth_map / fx
    Y = (v - cy) * depth_map / fy
    return np.dstack((X, Y, depth_map))


def run_unidepth(img, device):
    model = torch.hub.load('lpiccinelli-eth/UniDepth', 'UniDepth',
                           version='v2', backbone='vitl14',
                           pretrained=True, trust_repo=True).to(device).eval()
    rgb = torch.from_numpy(np.array(img)).permute(2, 0, 1)
    with torch.no_grad():
        pred = model.infer(rgb)

    depth = pred['depth'][0, 0].cpu().numpy()
    K = pred['intrinsics'][0].cpu().numpy()

    del model
    torch.cuda.empty_cache()
    return depth, K



# Conditioning features


def get_features(img_file, text, device):
    from timm import create_model
    from torchvision import transforms
    import clip

    vit = create_model('deit_base_patch16_224', pretrained=True)
    vit.head = torch.nn.Identity()
    vit = vit.to(device).eval()

    image = Image.open(img_file)
    max_dim = max(image.size)
    pad = (max_dim - image.size[0], max_dim - image.size[1])
    arr = np.pad(np.array(image),
                 ((pad[1] // 2, pad[1] - pad[1] // 2),
                  (pad[0] // 2, pad[0] - pad[0] // 2), (0, 0)), mode='constant')

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    with torch.no_grad():
        image_features = vit(tf(Image.fromarray(arr)).unsqueeze(0).to(device))

    clip_model, _ = clip.load('ViT-B/32', device=device)
    with torch.no_grad():
        text_features = clip_model.encode_text(clip.tokenize([text]).to(device))

    del vit, clip_model
    torch.cuda.empty_cache()
    return image_features.float(), text_features.float()



# Voxel grid


def gaussian_kernel(size=5, sigma=1.0):
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy, zz = np.meshgrid(ax, ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2. * sigma ** 2))
    return k / k.sum()


def compute_gaussian_heatmap(point, volume, sigma=1.0):
    G = volume.shape[0]
    ks = int(2 * np.ceil(3 * sigma) + 1)
    kernel = gaussian_kernel(ks, sigma)
    c = (np.clip(point, 0.0, 1.0) * (G - 1)).astype(int)

    s = [max(c[i] - ks // 2, 0) for i in range(3)]
    e = [min(c[i] + ks // 2 + 1, G) for i in range(3)]
    ks_s = [max(ks // 2 - c[i], 0) for i in range(3)]
    ks_e = [min(ks // 2 + (G - c[i]), ks) for i in range(3)]

    volume[s[0]:e[0], s[1]:e[1], s[2]:e[2]] += \
        kernel[ks_s[0]:ks_e[0], ks_s[1]:ks_e[1], ks_s[2]:ks_e[2]]
    return volume


def build_grid_and_volume(contact_point_3d, ranges_file, grid_size):
    with open(ranges_file, 'rb') as f:
        ranges = pickle.load(f)
    xr, yr, zr = ranges['x_range'], ranges['y_range'], ranges['z_range']

    xg, yg, zg = np.meshgrid(np.linspace(*xr, grid_size),
                             np.linspace(*yr, grid_size),
                             np.linspace(*zr, grid_size))
    grid = np.stack([xg, yg, zg], axis=-1)

    near = np.array([xr[0], yr[0], zr[0]])
    far = np.array([xr[1], yr[1], zr[1]])
    norm = (contact_point_3d - near) / (far - near)
    if np.any(norm < 0) or np.any(norm > 1):
        print(f'  WARNING: contact point outside the training ranges '
              f'(normalized = {np.round(norm, 3)}) -> clamped.')
    vol = compute_gaussian_heatmap(
        norm, np.zeros((grid_size, grid_size, grid_size)))
    return grid, vol



# Model


def build_model(args, device):
    edge_index = torch.load(pjoin(args.root, args.mano_edge_index_path)).to(device)

    ef_path = pjoin(args.root, args.mano_edge_features_path)
    edge_features = torch.load(ef_path).to(device) if os.path.exists(ef_path) else None
    edge_feat_dim = edge_features.shape[-1] if edge_features is not None else 0

    rm_path = pjoin(args.root, args.mano_region_map_path)
    region_map = torch.load(rm_path).to(device) if os.path.exists(rm_path) else None

    mano = MANO(os.environ['MANO_PATH'], is_rhand=True,
                flat_hand_mean=True).to(device).eval()
    for p in mano.parameters():
        p.requires_grad_(False)

    def fk_callable(motion_99d):
        _, verts = fk_verts_and_joints(rot6d_to_aa(motion_99d), mano, device)
        return verts

    denoiser = ContactDenoiser(
        latent_dim              = args.latent_dim,
        ff_size                 = args.ff_size,
        num_layers              = args.num_layers,
        num_heads               = args.num_heads,
        dropout                 = args.dropout,
        max_len                 = args.window_size + 10,
        contact_temporal_layers = args.contact_temporal_layers,
        contact_graph_layers    = args.contact_graph_layers,
        contact_graph_heads     = args.contact_graph_heads,
        contact_kernel_size     = args.contact_kernel_size,
        edge_feat_dim           = edge_feat_dim,
        region_map              = region_map,
        use_region_bottleneck   = region_map is not None,
        fk                      = fk_callable,
    ).to(device)

    p_v = torch.full((778,), 0.05)
    if args.vertex_marginals_path and os.path.exists(args.vertex_marginals_path):
        p_v = torch.load(args.vertex_marginals_path)
        print(f'[p_v] loaded from {args.vertex_marginals_path}')

    model = HybridDDPM(denoiser=denoiser, T=args.T_diffusion,
                       vertex_marginals=p_v, mano=mano).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()

    print(f'[ckpt] {args.ckpt} | epoch={ckpt.get("epoch", "?")} '
          f'| missing={len(missing)} unexpected={len(unexpected)}')
    if missing:
        print(f'  missing    : {missing[:5]}')
    if unexpected:
        print(f'  unexpected : {unexpected[:5]}')

    pv_mean = model.p_v.mean().item()
    print(f'  p_v mean : {pv_mean:.4f} | geom_tok : {denoiser.use_geometry}')
    if abs(pv_mean - 0.05) < 1e-6:
        print('  [WARN] p_v = uniform 0.05 fallback — marginals were not loaded. '
              'Pass --vertex_marginals_path, otherwise the generated contact '
              'density will be wrong.')

    return model, edge_index, edge_features


@torch.no_grad()
def run_inference(model, args, device, video_feats, text_feats,
                  contact_vol, contact_point_3d, grid, edge_index, edge_features):
    B, G = 1, args.contact_grid

    out = model.sample(
        video_feats   = video_feats.reshape(B, -1).to(device),
        text_feats    = text_feats.reshape(B, -1).to(device),
        contact_vol   = torch.from_numpy(contact_vol).float().reshape(B, G, G, G).to(device),
        contact_point = torch.from_numpy(contact_point_3d).float().reshape(B, 3).to(device),
        grid_vol      = torch.from_numpy(grid).float().reshape(B, G, G, G, 3).to(device),
        contact_mask  = torch.ones(B, 1, device=device),
        grid_mask     = torch.ones(B, device=device),
        T_seq         = args.n_frames,
        device        = device,
        verbose       = True,
        edge_index    = edge_index,
        edge_features = edge_features,
        T_diff_steps  = args.T_diff_steps,
        unconditional = False,
    )

    motion = out['motion'][0].detach().cpu()
    if args.contact_thr == 0.5:
        contact_maps = out['contact'][0].detach().cpu().long().numpy()
    else:
        probs = torch.sigmoid(out['contact_logits'][0].detach().cpu())
        contact_maps = (probs > args.contact_thr).long().numpy()

    return motion, contact_maps


# Rendering

def render_all(args, img, motion, contact_maps, contact_point_3d, K,
               point, box, mano_right, renderer):
    T = motion.shape[0]
    verb_noun = args.text.replace(' ', '+')
    curr_dir = pjoin(args.save_dir, verb_noun)

    dirs = {k: pjoin(curr_dir, k) for k in
            ['raw_image', 'pose_hand', 'other_hand', 'contact_map',
             'pose_checker', 'other_checker']}
    for d in [curr_dir] + list(dirs.values()):
        os.makedirs(d, exist_ok=True)

    with open(pjoin(curr_dir, f'{verb_noun}.json'), 'w') as f:
        json.dump({'text': args.text, 'img_name': args.img,
                   'point': list(point), 'box': box,
                   'contact_point_3d': contact_point_3d.tolist()}, f, indent=2)
    img.save(pjoin(dirs['raw_image'], f'{verb_noun}.jpg'))

    det = img.copy()
    dd = ImageDraw.Draw(det)
    if box is not None:
        dd.rectangle(box, outline='lime', width=4)
    dd.ellipse([point[0] - 12, point[1] - 12, point[0] + 12, point[1] + 12],
               fill='red', outline='white')
    det.save(pjoin(curr_dir, 'detection.jpg'))

    img_w, img_h = img.size
    focal = (K[0, 0] + K[1, 1]) / 2
    misc_args = dict(mesh_base_color=LIGHT_BLUE, scene_bg_color=(1., 1., 1.),
                     focal_length=focal, camera_z=0, render_res=(img_w, img_h))

    posed_verts, nview_verts, contact_verts = [], [], []
    other_mean = None

    for t in range(T):
        m = motion[t]
        go_ = rot_conv.axis_angle_to_matrix(m[:3][None]).view(1, -1, 3, 3)
        hp = rot_conv.axis_angle_to_matrix(m[3:48].view(-1, 3)).view(1, -1, 3, 3)
        betas = m[48:58][None]

        verts = mano_right(global_orient=go_, hand_pose=hp,
                           betas=betas, pose2rot=False).vertices[0].detach()

        cam_t = m[58:61][None]
        if args.shift_to_contact:
            cam_t = cam_t + torch.from_numpy(contact_point_3d).float()[None]
        ref_verts = (verts + cam_t).numpy()

        posed_verts.append(ref_verts.copy())
        contact_verts.append(contact_maps[t])

        other = ref_verts.copy()
        if t == 0:
            other_mean = other.mean(axis=0, keepdims=True)
        nview_verts.append(other - other_mean)

        rot_y = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        flat = mano_right(
            global_orient=torch.from_numpy(rot_y).float().view(1, 1, 3, 3),
            hand_pose=torch.eye(3).view(1, 1, 3, 3).repeat(1, 15, 1, 1),
            betas=betas, pose2rot=False).vertices[0].detach()
        flat = flat - flat.mean(dim=0, keepdim=True)

        fa = misc_args.copy()
        fa.update({'contact_vertices': contact_maps[t], 'rot_angle': 90,
                   'rot_axis': (1, 0, 0), 'camera_z': 0.25})
        Image.fromarray((renderer.render_rgba_contact(flat.numpy(), **fa)[..., :3] * 255)
                        .astype(np.uint8)).save(pjoin(dirs['contact_map'], f'{t:06d}.jpg'))

    all_v = np.stack(posed_verts).reshape(-1, 3)
    centroid = all_v.mean(axis=0, keepdims=True)
    for i in range(len(posed_verts)):
        posed_verts[i][:, 0] -= centroid[..., 0]
        posed_verts[i][:, 1] -= centroid[..., 1]

    fc = contact_point_3d[None].copy()
    fc[..., 0] -= centroid[..., 0]
    fc[..., 1] -= centroid[..., 1]
    verts_cam = np.dot(K, fc.T).T
    first_px = verts_cam[:, :2] / verts_cam[:, 2][:, None]

    args_multi = misc_args.copy()
    args_multi['contact_vertices'] = contact_verts

    pos_img = renderer.render_multiple_contacts(posed_verts, fading_factor=0.97, **args_multi)
    pos_img = Image.fromarray((pos_img[..., :3] * 255).astype(np.uint8))
    d = ImageDraw.Draw(pos_img)
    for pt in first_px:
        d.ellipse([pt[0] - 20, pt[1] - 20, pt[0] + 20, pt[1] + 20], fill='cyan')
    pos_img.save(pjoin(curr_dir, 'posed_motion.jpg'))

    bg_args = misc_args.copy()
    bg_args.update({'contact_vertices': contact_verts, 'bg_image': np.array(img).copy()})
    bg_img = renderer.render_multiple_contacts(
        [v.copy() for v in posed_verts], fading_factor=0.97, **bg_args)
    Image.fromarray((bg_img[..., :3] * 255).astype(np.uint8)) \
         .save(pjoin(curr_dir, 'bg_motion.jpg'))

    other_args = misc_args.copy()
    other_args.update({'contact_vertices': contact_verts, 'rot_angle': 90,
                       'rot_axis': (0, 1, 0), 'camera_z': 0.25})
    other_img = renderer.render_multiple_contacts(nview_verts, fading_factor=0.97, **other_args)
    Image.fromarray((other_img[..., :3] * 255).astype(np.uint8)) \
         .save(pjoin(curr_dir, 'other_motion.jpg'))

    for t in range(T):
        a = other_args.copy()
        a['contact_vertices'] = contact_verts[t]
        Image.fromarray((renderer.render_rgba_contact(nview_verts[t], **a)[..., :3] * 255)
                        .astype(np.uint8)).save(pjoin(dirs['other_hand'], f'{t:06d}.jpg'))

        a = misc_args.copy()
        a['contact_vertices'] = contact_verts[t]
        im = Image.fromarray((renderer.render_rgba_contact(posed_verts[t], **a)[..., :3] * 255)
                             .astype(np.uint8))
        dr = ImageDraw.Draw(im)
        for pt in first_px:
            dr.ellipse([pt[0] - 20, pt[1] - 20, pt[0] + 20, pt[1] + 20], fill='cyan')
        im.save(pjoin(dirs['pose_hand'], f'{t:06d}.jpg'))

    thumb = img.convert('RGBA').copy()
    dt = ImageDraw.Draw(thumb)
    dt.ellipse([point[0] - 20, point[1] - 20, point[0] + 20, point[1] + 20],
               fill='red', outline='red')
    thumb = thumb.resize((thumb.width // 4, thumb.height // 4), Image.Resampling.LANCZOS)

    for verts_list, out_dir, name, top_down in [
        (posed_verts, dirs['pose_checker'],  'posed_motion_checkerboard.mp4', False),
        (nview_verts, dirs['other_checker'], 'other_motion_checkerboard.mp4', True),
    ]:
        writer = imageio.get_writer(pjoin(out_dir, name), fps=args.fps, macro_block_size=1)
        for t in range(T):
            a = misc_args.copy()
            a.update({'contact_vertices': contact_verts[t], 'checkerboard': True})
            if top_down:
                a['camera_top_down'] = True
            im = Image.fromarray(
                (renderer.render_rgba_contact(verts_list[t], **a)[..., :3] * 255).astype(np.uint8))
            im.save(pjoin(out_dir, f'{t:06d}.jpg'))
            overlay = get_text_image(args.text, thumb.copy(), font_scale=0.5, thickness=1)
            im.paste(overlay, (0, 0), mask=overlay.convert('RGBA'))
            writer.append_data(np.array(im))
        writer.close()
        print(f'  -> {pjoin(out_dir, name)}')

    save_plotly(curr_dir, posed_verts, contact_verts, fc)
    return curr_dir


def save_plotly(curr_dir, posed_verts, contact_verts, first_contact):
    import plotly.graph_objects as go

    T = len(posed_verts)
    cverts = []
    for t in range(T):
        pts = posed_verts[t][contact_verts[t].astype(bool)]
        cverts.append(pts if pts.size else np.zeros((0, 3)))

    allp = np.concatenate(posed_verts + [first_contact], axis=0)
    lo, hi = allp.min(axis=0), allp.max(axis=0)

    def frame_data(t):
        return [
            go.Scatter3d(x=posed_verts[t][:, 0], y=posed_verts[t][:, 1], z=posed_verts[t][:, 2],
                         mode='markers', name='Hand', marker=dict(size=2.5, color='blue')),
            go.Scatter3d(x=cverts[t][:, 0], y=cverts[t][:, 1], z=cverts[t][:, 2],
                         mode='markers', name='Contact', marker=dict(size=2.5, color='red')),
            go.Scatter3d(x=first_contact[:, 0], y=first_contact[:, 1], z=first_contact[:, 2],
                         mode='markers', name='Anchor', marker=dict(size=5, color='cyan')),
        ]

    fig = go.Figure(
        data=frame_data(0),
        layout=go.Layout(
            scene=dict(xaxis=dict(range=[lo[0], hi[0]]),
                       yaxis=dict(range=[lo[1], hi[1]]),
                       zaxis=dict(range=[lo[2], hi[2]]), aspectmode='data'),
            updatemenus=[dict(type='buttons', showactive=False, y=1.05, x=0.8,
                              buttons=[dict(label='Play', method='animate',
                                            args=[None, dict(frame=dict(duration=200, redraw=True),
                                                             fromcurrent=True)])])],
            sliders=[dict(steps=[dict(method='animate', args=[[f't{t}'],
                          dict(mode='immediate', frame=dict(duration=200, redraw=True))])
                          for t in range(T)], currentvalue=dict(prefix='Time: '))],
        ),
        frames=[go.Frame(data=frame_data(t), name=f't{t}') for t in range(T)],
    )
    fig.write_html(pjoin(curr_dir, 'animation.html'))


# Main

def main():
    args = DemoDisContactOptions().parse()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    assert 'MANO_PATH' in os.environ, 'export MANO_PATH=...'
    print(f'Device : {device}')

    img = Image.open(args.img).convert('RGB')

    print('\n[1/6] Metric depth (UniDepth)...')
    depth, K = run_unidepth(img, device)

    print('[2/6] Object localization...')
    if args.point is not None:
        u, v, box, score = args.point[0], args.point[1], None, 1.0
        print(f'  manually provided point: ({u}, {v})')
    else:
        query = args.query or text_to_query(args.text)
        u, v, box, score = detect_contact_pixel(img, query, args, device)

    u, v = refine_with_depth(u, v, box, depth, mode=args.det_mode)
    points_3d = unproject_depth_to_3d(depth, K)
    contact_point_3d = np.array(points_3d[v, u])
    print(f'  pixel ({u}, {v}) -> 3D contact {np.round(contact_point_3d, 4)}')

    print('[3/6] DeiT + CLIP features...')
    video_feats, text_feats = get_features(args.img, args.text, device)

    print('[4/6] Grid + Gaussian volume...')
    grid, contact_vol = build_grid_and_volume(
        contact_point_3d, args.ranges, args.contact_grid)

    print('[5/6] DisContact model...')
    model, edge_index, edge_features = build_model(args, device)

    print(f'[6/6] Sampling ({args.n_frames} frames, '
          f'{args.T_diff_steps or args.T_diffusion} steps)...')
    motion, contact_maps = run_inference(
        model, args, device, video_feats, text_feats,
        contact_vol, contact_point_3d, grid, edge_index, edge_features)

    mano_right = MANO(model_path=os.environ['MANO_PATH'], gender='neutral',
                      num_hand_joints=15, create_body_pose=False)
    renderer = Renderer(mano_right.faces, img_res=img.size)

    print('Rendering...')
    out_dir = render_all(args, img, motion, contact_maps, contact_point_3d, K,
                         (u, v), box, mano_right, renderer)

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    for a, f, ttl in zip(ax, ['detection.jpg', 'posed_motion.jpg',
                              'bg_motion.jpg', 'other_motion.jpg'],
                         ['Detection', 'Posed', 'Overlay', 'Other view']):
        a.imshow(np.array(Image.open(pjoin(out_dir, f))))
        a.set_title(ttl); a.axis('off')
    plt.tight_layout()
    plt.savefig(pjoin(out_dir, 'summary.jpg'), dpi=150)
    print(f'\nDone -> {out_dir}')


if __name__ == '__main__':
    main()