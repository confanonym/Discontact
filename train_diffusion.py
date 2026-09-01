

import os
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from os.path import join as pjoin

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from options.discontact_option import TrainDisContactOptions

import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.t2m_dataset import HoloMotion, ArcticMotion
from models.diffusion.denoiser import ContactDenoiser
from models.diffusion.diffusion import HybridDDPM, fk_verts_and_joints
from models.diffusion.losses import compute_all_losses, region_density
from models.diffusion.utils import compute_vertex_marginals, rot6d_to_aa
from utils.metrics import (
    compute_mpjpe, compute_mpjpe_ra, compute_mpjpe_pa, binary_classification_metrics,
)
import common.rotation_conversions as rot

try:
    import imageio
    from PIL import Image, ImageDraw as ID
    from common.renderer import Renderer
    HAS_RENDERER = True
except ImportError:
    HAS_RENDERER = False




def make_dataset_opt(args):
   
    opt = argparse.Namespace()

    # From the CLI
    opt.dataset_name      = args.dataset_name
    opt.setting           = args.setting
    opt.window_size       = args.window_size
    opt.max_motion_length = args.window_size
    opt.checkpoints_dir   = args.checkpoints_dir
    opt.debug             = args.debug

    # Fixed for this pipeline
    opt.joints_num     = 21
    opt.motion_type    = 'mano'
    opt.video_feats    = 768
    opt.text_feats     = True
    opt.contact_grid   = 16
    opt.contact_dim    = 16
    opt.contact_map    = True
    opt.coord_sys      = 'contact'
    opt.only_first     = True
    opt.full_sequence  = True
    opt.interpolate    = False
    opt.use_inpaint    = False
    opt.return_indices = False
    opt.decoder_only   = False
    opt.load_indices   = None
    opt.transfer_from  = None
    opt.pred_cam       = False
    opt.viz_gt         = False
    opt.viz_pred       = None

    return opt


_MANO_CACHE = {}


def get_mano(device):
    """Charge le wrapper MANO une seule fois par device. Retourne None si indispo."""
    if 'MANO_PATH' not in os.environ:
        return None
    key = str(device)
    if key not in _MANO_CACHE:
        try:
            from models.mano.wrapper import MANO as _MANOWrapper
            m = _MANOWrapper(
                os.environ['MANO_PATH'], is_rhand=True, flat_hand_mean=True,
            ).to(device).eval()
            for p in m.parameters():
                p.requires_grad_(False)
            _MANO_CACHE[key] = m
        except Exception as e:
            print(f'[get_mano] MANO indisponible ({e}).')
            _MANO_CACHE[key] = None
    return _MANO_CACHE[key]



def run_val_loss(model, loader, device, loss_fn, edge_index, edge_features, debug=False):
    model.eval()
    val_losses = defaultdict(list)
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc='Val loss', leave=False)):
            for k in batch:
                if torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(device)
            losses = model(batch, loss_fn, edge_index=edge_index, edge_features=edge_features)
            for k, v in losses.items():
                val_losses[k].append(v.item() if torch.is_tensor(v) else float(v))
            if debug and i >= 2:
                break
    return {k: float(np.mean(v)) for k, v in val_losses.items()}


def run_val_inference(model, loader, device, max_samples, edge_index, edge_features,
                      T_diff_val=None, debug=False):
    model.eval()
    all_clogits, all_clabels, all_cmasks = [], [], []
    jp_list, jg_list, vm_list = [], [], []
    n_done = 0

    mano = get_mano(device)

    with torch.no_grad():
        for batch in tqdm(loader, desc='Val inference (cond)', leave=False):
            for k in batch:
                if torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(device)

            motion_gt  = batch['motion'].float()
            contact_gt = batch['contact_map'].float()
            valid_mask = batch['valid_mask'].float()
            T_seq      = motion_gt.shape[1]

            out = model.sample(
                video_feats   = batch['video_feats'][:, 0].float(),
                text_feats    = batch['text_feats'][:, 0].float(),
                contact_vol   = batch['contact'][:, 0].float(),
                contact_point = batch['contact_point'][:, 0].float(),
                grid_vol      = batch['grid'].float(),
                contact_mask  = batch['contact_mask'].float(),
                grid_mask     = batch['grid_mask'].float(),
                T_seq=T_seq, device=device, verbose=False,
                edge_index=edge_index, edge_features=edge_features,
                T_diff_steps=T_diff_val, unconditional=False,
            )

            motion_pred  = out['motion']
            contact_pred = out['contact_logits']

            if mano is not None:
                jp, _ = fk_verts_and_joints(motion_pred, mano, device)
                jg, _ = fk_verts_and_joints(motion_gt,   mano, device)
                J = jp.shape[-2]
                jp_list.append(jp.reshape(-1, J, 3).cpu())
                jg_list.append(jg.reshape(-1, J, 3).cpu())
                vm_list.append(valid_mask.reshape(-1).cpu())

            cmask = valid_mask.unsqueeze(-1)
            all_clogits.append(contact_pred.reshape(-1, 778))
            all_clabels.append(contact_gt.reshape(-1, 778))
            all_cmasks.append(cmask.reshape(-1, 1))

            n_done += motion_gt.shape[0]
            if n_done >= max_samples or debug:
                break

    logits_c = torch.cat(all_clogits, dim=0)
    labels_c = torch.cat(all_clabels, dim=0)
    mask_c   = torch.cat(all_cmasks,  dim=0)
    prec, rec, f1 = binary_classification_metrics(logits_c, labels_c, mask_c)

    if jp_list:
        jp_cat = torch.cat(jp_list, dim=0)
        jg_cat = torch.cat(jg_list, dim=0)
        vm_cat = torch.cat(vm_list, dim=0)
        mpjpe    = compute_mpjpe(jp_cat, jg_cat, valid=vm_cat).item()
        mpjpe_ra = compute_mpjpe_ra(jp_cat, jg_cat, valid=vm_cat).item()
        mpjpe_pa = float(compute_mpjpe_pa(jp_cat, jg_cat, valid=vm_cat))
    else:
        mpjpe = mpjpe_ra = mpjpe_pa = float('nan')

    return {
        'mpjpe':        mpjpe,
        'mpjpe_ra':     mpjpe_ra,
        'mpjpe_pa':     mpjpe_pa,
        'contact_prec': prec,
        'contact_rec':  rec,
        'contact_f1':   f1,
    }, n_done


def run_val_inference_unconditional(model, device, save_dir, epoch, T_seq=150,
                                    n_samples=4, edge_index=None, edge_features=None,
                                    T_diff_val=None, use_wandb=False):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    video_feats   = torch.zeros(n_samples, 768, device=device)
    text_feats    = torch.zeros(n_samples, 512, device=device)
    contact_vol   = torch.zeros(n_samples, 16, 16, 16, device=device)
    contact_point = torch.zeros(n_samples, 3, device=device)

    with torch.no_grad():
        out = model.sample(
            video_feats=video_feats, text_feats=text_feats,
            contact_vol=contact_vol, contact_point=contact_point,
            T_seq=T_seq, device=device, verbose=False,
            edge_index=edge_index, edge_features=edge_features,
            T_diff_steps=T_diff_val, unconditional=True,
        )

    contact_probs = torch.sigmoid(out['contact_logits']).cpu().numpy()
    motion        = out['motion'].cpu().numpy()

    mean_density     = contact_probs.mean(axis=(1, 2))
    var_inter_sample = contact_probs.var(axis=0).mean()
    var_intra_time   = contact_probs.var(axis=1).mean()
    motion_std       = motion.std(axis=1).mean(axis=-1)

    metrics = {
        'uncond/mean_density':     float(mean_density.mean()),
        'uncond/density_std':      float(mean_density.std()),
        'uncond/var_inter_sample': float(var_inter_sample),
        'uncond/var_intra_time':   float(var_intra_time),
        'uncond/motion_std':       float(motion_std.mean()),
    }

    print(f'[Uncond Ep {epoch}] density={metrics["uncond/mean_density"]:.4f} '
          f'(±{metrics["uncond/density_std"]:.4f}) | '
          f'var_inter={metrics["uncond/var_inter_sample"]:.4f} | '
          f'motion_std={metrics["uncond/motion_std"]:.4f}')

    fig, axes = plt.subplots(2, n_samples, figsize=(4 * n_samples, 8))
    fig.suptitle(f'Ep {epoch} — Unconditional sampling', fontsize=13)
    if n_samples == 1:
        axes = axes.reshape(2, 1)
    for k in range(n_samples):
        axes[0, k].imshow(contact_probs[k].T, aspect='auto', cmap='Reds', vmin=0, vmax=1)
        axes[0, k].set_title(f'Sample {k} | density={mean_density[k]:.3f}')
        axes[0, k].set_xlabel('Time'); axes[0, k].set_ylabel('Vertex')
        axes[1, k].plot(np.arange(T_seq), contact_probs[k].mean(-1))
        axes[1, k].set_title('Mean contact / frame')
        axes[1, k].set_ylim(0, max(0.3, mean_density[k] * 2))
        axes[1, k].grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(save_dir, f'uncond_ep{epoch:04d}.png')
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'  -> Uncond viz : {out_path}')
    if use_wandb:
        wandb.log({**metrics, 'demo/unconditional': wandb.Image(out_path), 'epoch': epoch})
    return metrics



REGION_NAMES  = ['Paume', 'Index', 'Majeur', 'Annulaire', 'Auriculaire', 'Pouce']
REGION_COLORS = [
    [0.7, 0.7, 0.7], [1.0, 0.2, 0.2], [0.2, 0.8, 0.2],
    [0.2, 0.2, 1.0], [1.0, 0.8, 0.0], [0.8, 0.2, 0.8],
]


def demo_sample(model, loader, device, save_dir, epoch, use_wandb,
                edge_index, edge_features, T_diff_val=None, region_map=None):
    model.eval()
    batch = next(iter(loader))
    for k in batch:
        if torch.is_tensor(batch[k]):
            batch[k] = batch[k].to(device)
    b1 = {k: v[:1] if torch.is_tensor(v) else v for k, v in batch.items()}

    contact_gt = b1['contact_map'].float()
    valid_mask = b1['valid_mask'].float()
    T_valid    = int(valid_mask[0].sum().item())
    T_seq      = contact_gt.shape[1]

    with torch.no_grad():
        out = model.sample(
            video_feats   = b1['video_feats'][:, 0].float(),
            text_feats    = b1['text_feats'][:, 0].float(),
            contact_vol   = b1['contact'][:, 0].float(),
            contact_point = b1['contact_point'][:, 0].float(),
            grid_vol      = b1['grid'].float(),
            contact_mask  = b1['contact_mask'].float(),
            grid_mask     = b1['grid_mask'].float(),
            T_seq=T_seq, device=device, verbose=False,
            edge_index=edge_index, edge_features=edge_features,
            T_diff_steps=T_diff_val, unconditional=False,
        )

    cp = torch.sigmoid(out['contact_logits'][0, :T_valid]).cpu().numpy()
    cg = contact_gt[0, :T_valid].cpu().numpy()
    t_axis = np.arange(T_valid)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f'Ep {epoch} — Contact GT vs Pred', fontsize=13)
    axes[0, 0].imshow(cg.T, aspect='auto', cmap='Reds', vmin=0, vmax=1)
    axes[0, 0].set_title('Contact GT (778 vertices)')
    axes[0, 0].set_xlabel('Time'); axes[0, 0].set_ylabel('MANO vertex')
    axes[0, 1].imshow(cp.T, aspect='auto', cmap='Reds', vmin=0, vmax=1)
    axes[0, 1].set_title(f'Contact Pred  mean={cp.mean():.3f}')
    axes[0, 1].set_xlabel('Time'); axes[0, 1].set_ylabel('MANO vertex')
    axes[1, 0].plot(t_axis, cg.mean(-1), label='GT')
    axes[1, 0].plot(t_axis, cp.mean(-1), '--', label='Pred')
    axes[1, 0].set_title('Mean contact / frame')
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].axis('off')
    axes[1, 1].text(0.05, 0.6,
        f'T valid      : {T_valid}\n'
        f'GT density   : {cg.mean():.4f}\n'
        f'Pred density : {cp.mean():.4f}',
        fontsize=12, family='monospace')
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f'demo_ep{epoch:04d}.png')
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'  -> Demo contact : {out_path}')
    if use_wandb:
        wandb.log({'demo/contact': wandb.Image(out_path), 'epoch': epoch})

    if region_map is not None:
        try:
            rm   = region_map.cpu()
            cg_t = torch.tensor(cg)
            cp_t = torch.tensor(cp)
            rg_gt   = region_density(cg_t.unsqueeze(0), rm).squeeze(0).numpy()
            rg_pred = region_density(cp_t.unsqueeze(0), rm).squeeze(0).numpy()

            fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
            fig2.suptitle(f'Ep {epoch} — Contact par région (GT vs Pred)', fontsize=13)
            for r, (name, color) in enumerate(zip(REGION_NAMES, REGION_COLORS)):
                ax = axes2[r // 3, r % 3]
                ax.plot(t_axis, rg_gt[:, r],   color=color, linewidth=2,   label='GT')
                ax.plot(t_axis, rg_pred[:, r], color=color, linewidth=1.5,
                        linestyle='--', alpha=0.8, label='Pred')
                ax.fill_between(t_axis, rg_gt[:, r],   alpha=0.15, color=color)
                ax.fill_between(t_axis, rg_pred[:, r], alpha=0.10, color=color,
                                linestyle='--')
                ax.set_title(f'{name} | GT={rg_gt[:, r].mean():.3f} '
                             f'Pred={rg_pred[:, r].mean():.3f}', fontsize=10)
                ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
                ax.set_xlabel('Frame'); ax.set_ylabel('Densité contact')
            plt.tight_layout()
            rpath = os.path.join(save_dir, f'demo_region_ep{epoch:04d}.png')
            plt.savefig(rpath, dpi=120); plt.close(fig2)
            print(f'  -> Demo région : {rpath}')
            if use_wandb:
                wandb.log({'demo/region': wandb.Image(rpath), 'epoch': epoch})
        except Exception as e:
            print(f'  [region viz error: {e}]')


def demo_video_mano(model, loader, device, save_dir, epoch,
                    fps=10, res=512, use_wandb=False,
                    edge_index=None, edge_features=None, T_diff_val=None):
    try:
        import imageio
        from PIL import Image as PILImage, ImageDraw as PILID
        from common.renderer import Renderer
        import smplx as smplx_lib
        LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
    except ImportError as e:
        print(f'  -> make_video ignoré ({e})'); return None
    if 'MANO_PATH' not in os.environ:
        print('  -> make_video ignoré : MANO_PATH non défini'); return None

    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    mano_layer = smplx_lib.MANO(
        os.environ['MANO_PATH'], use_pca=False, flat_hand_mean=True,
    ).to(device)
    renderer = Renderer(mano_layer.faces, img_res=(res, res))

    batch = next(iter(loader))
    for k in batch:
        if torch.is_tensor(batch[k]):
            batch[k] = batch[k].to(device)
    b1 = {k: v[:1] if torch.is_tensor(v) else v for k, v in batch.items()}

    valid_mask = b1['valid_mask'].float()
    T_valid    = int(valid_mask[0].sum().item())

    with torch.no_grad():
        out = model.sample(
            video_feats   = b1['video_feats'][:, 0].float(),
            text_feats    = b1['text_feats'][:, 0].float(),
            contact_vol   = b1['contact'][:, 0].float(),
            contact_point = b1['contact_point'][:, 0].float(),
            grid_vol      = b1['grid'].float(),
            contact_mask  = b1['contact_mask'].float(),
            grid_mask     = b1['grid_mask'].float(),
            T_seq=T_valid, device=device, verbose=False,
            edge_index=edge_index, edge_features=edge_features,
            T_diff_steps=T_diff_val, unconditional=False,
        )

    motion       = out['motion'][0].float().to(device)
    contact_maps = out['contact'][0].bool().cpu().numpy()
    frames = []

    for t in range(T_valid):
        curr    = motion[t]
        out_m   = mano_layer(
            global_orient=curr[:3].unsqueeze(0),
            hand_pose=curr[3:48].unsqueeze(0),
            betas=curr[48:58].unsqueeze(0),
            return_verts=True,
        )
        verts = (out_m.vertices[0] + curr[58:61]).detach().cpu().numpy()
        verts = verts - verts.mean(axis=0, keepdims=True)
        img   = renderer.render_rgba_contact(
            verts,
            mesh_base_color=LIGHT_BLUE, scene_bg_color=(1., 1., 1.),
            contact_vertices=contact_maps[t], checkerboard=True,
            camera_z=0.35, render_res=(res, res),
        )
        frame = (img[..., :3] * 255).astype(np.uint8)
        pil   = PILImage.fromarray(frame)
        PILID.Draw(pil).text((10, 10), f'Ep {epoch} | PRED | t={t}', fill=(0, 0, 0))
        frames.append(np.array(pil))

    out_path = os.path.join(save_dir, f'pred_ep{epoch:04d}.mp4')
    imageio.mimsave(out_path, frames, fps=fps, macro_block_size=1)
    print(f'  -> Pred video : {out_path}')
    if use_wandb:
        wandb.log({'demo/pred_video': wandb.Video(out_path, fps=fps, format='mp4'),
                   'epoch': epoch})
    return out_path




def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')

    save_root  = pjoin(args.checkpoints_dir, args.dataset_name, args.name)
    model_dir  = pjoin(save_root, 'model')
    log_dir    = pjoin(save_root, 'log')
    demo_dir   = pjoin(save_root, 'demo')
    video_dir  = pjoin(save_root, 'video')
    uncond_dir = pjoin(save_root, 'uncond')
    for d in [model_dir, log_dir, demo_dir, video_dir, uncond_dir]:
        os.makedirs(d, exist_ok=True)

    writer    = SummaryWriter(log_dir)
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   name=args.name, config=vars(args), dir=save_root)

    # === Mesh ===
    root = os.environ.get('ROOT_DIR', '.')

    edge_index_path    = pjoin(root, args.mano_edge_index_path)
    edge_features_path = pjoin(root, args.mano_edge_features_path)
    region_map_path    = pjoin(root, args.mano_region_map_path)

    if not os.path.exists(edge_index_path):
        raise FileNotFoundError(f'edge_index not found at {edge_index_path}.\n'
                                f'Run : python scripts/build_mano_edges.py')

    edge_index    = torch.load(edge_index_path).to(device)
    edge_features = torch.load(edge_features_path).to(device) \
                    if os.path.exists(edge_features_path) else None
    edge_feat_dim = edge_features.shape[-1] if edge_features is not None else 0
    print(f'edge_index : {edge_index.shape} | edge_feat_dim : {edge_feat_dim}')

    region_map = torch.load(region_map_path).to(device) \
                 if os.path.exists(region_map_path) else None
    if region_map is not None:
        print(f'region_map : {region_map.shape} | régions : {region_map.unique().tolist()}')
    else:
        print(f'region_map absent — losses région désactivées.')

    # === MANO ===
    mano = get_mano(device)
    if mano is None:
        print('MANO indisponible — l_motion_pos et geom_tok désactivés.')

    # === fk_callable pour le denoiser (geom_tok) ===
    # Accepte du 99D 6D (espace interne du denoiser) -> retourne (B,T,778,3)
    def fk_callable(motion_99d):
        aa = rot6d_to_aa(motion_99d)                          # (B, T, 61)
        _, verts = fk_verts_and_joints(aa, mano, device)
        return verts                                           # (B, T, 778, 3)

    fk_fn = fk_callable if mano is not None else None

    # === Datasets ===
    DatasetClass  = HoloMotion if args.dataset_name == 'holo' else ArcticMotion
    train_dataset = DatasetClass(make_dataset_opt(args), split='train')
    val_dataset   = DatasetClass(make_dataset_opt(args), split='val')
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               drop_last=True, num_workers=args.num_workers, pin_memory=True)
    val_loader    = DataLoader(val_dataset, batch_size=4, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    print(f'Train : {len(train_dataset)} | Val : {len(val_dataset)}')

    # === Marginales par-vertex ===
    if args.vertex_marginals_path and os.path.exists(args.vertex_marginals_path):
        p_v = torch.load(args.vertex_marginals_path)
        print(f'Loaded vertex marginals from {args.vertex_marginals_path}')
    else:
        print('Computing vertex marginals on train set...')
        marg_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers)
        p_v       = compute_vertex_marginals(marg_loader, device)
        save_path = pjoin(save_root, 'vertex_marginals.pt')
        torch.save(p_v, save_path)
        print(f'Saved -> {save_path} | mean={p_v.mean():.4f}')

    if use_wandb:
        wandb.config.update({'contact_density': p_v.mean().item()})

    # === Model ===
    use_bottleneck = region_map is not None

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
        use_region_bottleneck   = use_bottleneck,
        fk                      = fk_fn,
        grad_motion_to_contact  = args.grad_m2c,
        grad_contact_to_motion  = args.grad_c2m,
    ).to(device)

    model = HybridDDPM(
        denoiser         = denoiser,
        T                = args.T_diffusion,
        vertex_marginals = p_v,
        mano             = mano,
    ).to(device)

    # Sanity check
    print('\n' + '=' * 50)
    print('SANITY CHECK')
    print(f'  geom_tok actif      : {denoiser.use_geometry}')
    print(f'  grad_m2c            : {args.grad_m2c}')
    print(f'  grad_c2m            : {args.grad_c2m}')
    print(f'  bottleneck actif    : {denoiser.region_bottleneck is not None}')
    print(f'  l_motion_pos actif  : {mano is not None}')
    if denoiser.region_bottleneck is not None:
        rb = denoiser.region_bottleneck
        print(f'  region_counts       : {rb.region_counts.long().tolist()}')
        print(f'  gate init (tanh)    : {torch.tanh(rb.gate).item():.4f}')
    total_p = sum(p.numel() for p in model.parameters())
    print(f'  Parameters          : {total_p/1e6:.2f}M')
    print('=' * 50 + '\n')

    # === Loss ===
    def loss_fn(**kwargs):
        return compute_all_losses(
            region_map       = region_map,
            w_motion         = args.w_motion,
            w_motion_pos     = args.w_motion_pos,
            w_contact_vb     = args.w_contact_vb,
            w_contact_aux    = args.w_contact_aux,
            w_region_density = args.w_region_density,
            **kwargs,
        )

    # === Optim ===
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.99), weight_decay=args.weight_decay)
    scheduler = None
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.milestones, gamma=args.gamma)
        print(f'Scheduler : milestones={args.milestones}, gamma={args.gamma}')
    else:
        print(f'LR fixe : {args.lr}')

    # === Checkpoint state ===
    epoch_start, it = 0, 0
    best_train_loss = float('inf')
    best_epoch      = -1

    def lr_warmup(it, warm_up, lr):
        for g in optimizer.param_groups:
            g['lr'] = lr * (it + 1) / (warm_up + 1)

    def save_ckpt(path, epoch, it, loss=None, best_loss=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'epoch': epoch, 'it': it}
        if scheduler is not None:
            ckpt['scheduler'] = scheduler.state_dict()
        if loss is not None:
            ckpt['loss'] = float(loss)
        if best_loss is not None:
            ckpt['best_train_loss'] = float(best_loss)
        torch.save(ckpt, path)

    if args.is_continue:
        ckpt_path = pjoin(model_dir, 'latest.tar')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
            print(f'[resume] missing    ({len(missing)}): {missing[:15]}')
            print(f'[resume] unexpected ({len(unexpected)}): {unexpected[:15]}')
            model.load_state_dict(ckpt['model'], strict=False)
            optimizer.load_state_dict(ckpt['optimizer'])
            if scheduler is not None and 'scheduler' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler'])
            epoch_start = ckpt['epoch'] + 1
            it          = ckpt['it']

            if 'best_train_loss' in ckpt:
                best_train_loss = float(ckpt['best_train_loss'])
                best_epoch      = ckpt.get('epoch', -1)
                src = 'latest[best_train_loss]'
            else:
                finest_path = pjoin(model_dir, 'finest.tar')
                if os.path.exists(finest_path):
                    finest_ckpt     = torch.load(finest_path, map_location='cpu')
                    best_train_loss = float(finest_ckpt.get('loss', float('inf')))
                    best_epoch      = finest_ckpt.get('epoch', -1)
                    src = f'finest.tar (loss={best_train_loss:.4f}, ep={best_epoch})'
                else:
                    best_train_loss = float(ckpt.get('loss', float('inf')))
                    best_epoch      = ckpt.get('epoch', -1)
                    src = 'latest[loss] (finest absent)'

            print(f'Resumed ep {epoch_start}, it {it} | '
                  f'best_train_loss={best_train_loss:.4f} [{src}]')
        else:
            print(f'--is_continue mais {ckpt_path} introuvable — départ à zéro.')

    # === Training loop ===
    log_keys_motion  = ['l_motion', 'l_motion_pos']
    log_keys_contact = ['l_contact_vb', 'l_contact_aux', 'l_region_density']

    for epoch in tqdm(range(epoch_start, args.max_epoch), desc='Epochs'):
        model.train()
        log_buffer  = defaultdict(float)
        log_count   = 0
        ep_loss_sum = 0.0
        ep_loss_cnt = 0

        for i, batch in enumerate(tqdm(train_loader, desc=f'Ep {epoch}', leave=False)):
            for k in batch:
                if torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(device)

            if args.use_scheduler and it < args.warm_up_iter:
                lr_warmup(it, args.warm_up_iter, args.lr)

            losses = model(batch, loss_fn, edge_index=edge_index, edge_features=edge_features)

            optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            it           += 1
            total_loss    = losses['total'].item()
            ep_loss_sum  += total_loss
            ep_loss_cnt  += 1

            for k, v in losses.items():
                log_buffer[k] += v.item() if torch.is_tensor(v) else float(v)
            log_buffer['lr'] += optimizer.param_groups[0]['lr']
            log_count += 1

            if it % args.log_every == 0:
                log_dict = {k: v / max(log_count, 1) for k, v in log_buffer.items()}
                for k, v in log_dict.items():
                    writer.add_scalar(f'Train/{k}', v, it)
                if use_wandb:
                    wandb.log({**{f'train/{k}': v for k, v in log_dict.items()}, 'iter': it})

                print(f'\nEp {epoch:04d} | it {it:06d} | total={log_dict.get("total", 0):.4f}')
                print('  Motion  : ' + '  '.join(
                    f'{k}={log_dict.get(k, 0):.4f}' for k in log_keys_motion))
                print('  Contact : ' + '  '.join(
                    f'{k}={log_dict.get(k, 0):.4f}' for k in log_keys_contact))
                print(f'  lr={log_dict.get("lr", 0):.6f}')

                # Debug densité par région toutes les 5*log_every iter
                if region_map is not None and it % (args.log_every * 5) == 0:
                    try:
                        _cm  = batch['contact_map'].float()
                        _msk = batch['valid_mask'].float()
                        _gt_r = region_density(_cm, region_map.cpu())
                        gt_r_mean = (_gt_r * _msk.unsqueeze(-1)).sum((0, 1)) \
                                    / _msk.sum().clamp(min=1)
                        print('  [GT density par région]')
                        for r, (name, d) in enumerate(zip(REGION_NAMES, gt_r_mean.tolist())):
                            print(f'    {name:15} : {d:.4f}  {"█" * int(d * 20)}')
                    except Exception as e:
                        print(f'  [region debug error: {e}]')

                log_buffer = defaultdict(float)
                log_count  = 0

            if it % args.save_latest == 0:
                save_ckpt(pjoin(model_dir, 'latest.tar'), epoch, it,
                          total_loss, best_train_loss)

            if args.debug and i >= 2:
                break

        cur_train_loss = ep_loss_sum / max(ep_loss_cnt, 1)
        save_ckpt(pjoin(model_dir, 'latest.tar'), epoch, it,
                  cur_train_loss, best_train_loss)
        writer.add_scalar('Train/epoch_loss', cur_train_loss, epoch)

        if cur_train_loss < best_train_loss:
            best_train_loss = cur_train_loss
            best_epoch      = epoch
            save_ckpt(pjoin(model_dir, 'finest.tar'), epoch, it,
                      cur_train_loss, best_train_loss)
            print(f'  -> finest.tar | loss={cur_train_loss:.4f} | ep={best_epoch}')

        print(f'[Train Ep {epoch}] loss={cur_train_loss:.4f} | best={best_train_loss:.4f}')

        if (epoch + 1) % args.val_every_e == 0:
            val_losses = run_val_loss(
                model, val_loader, device, loss_fn,
                edge_index=edge_index, edge_features=edge_features, debug=args.debug)
            metrics, n_done = run_val_inference(
                model, val_loader, device, args.val_max_samples,
                edge_index=edge_index, edge_features=edge_features,
                T_diff_val=args.T_diff_val, debug=args.debug)

            for k, v in val_losses.items():
                writer.add_scalar(f'ValLoss/{k}', v, epoch)
            for k, v in metrics.items():
                writer.add_scalar(f'ValMetric/{k}', v, epoch)
            if use_wandb:
                val_log = {'epoch': epoch}
                for k, v in val_losses.items():
                    val_log[f'val/loss_{k}'] = v
                for k, v in metrics.items():
                    val_log[f'val/{k}'] = v
                wandb.log(val_log)

            print(f'[Val Ep {epoch}] ({n_done} samples)\n'
                  f'  Val total loss : {val_losses["total"]:.4f}\n'
                  f'  MPJPE          : {metrics["mpjpe"]:.4f}  '
                  f'(RA={metrics["mpjpe_ra"]:.4f}  PA={metrics["mpjpe_pa"]:.4f})\n'
                  f'  Contact F1     : {metrics["contact_f1"]:.4f}  '
                  f'P={metrics["contact_prec"]:.4f}  R={metrics["contact_rec"]:.4f}')

            demo_sample(model, val_loader, device, demo_dir, epoch,
                        use_wandb=use_wandb, edge_index=edge_index,
                        edge_features=edge_features, T_diff_val=args.T_diff_val,
                        region_map=region_map)

            if args.make_video and (epoch + 1) % args.video_every_e == 0:
                demo_video_mano(model=model, loader=val_loader, device=device,
                                save_dir=video_dir, epoch=epoch,
                                fps=args.video_fps, res=args.video_res,
                                use_wandb=use_wandb, edge_index=edge_index,
                                edge_features=edge_features, T_diff_val=args.T_diff_val)
            model.train()

        if (epoch + 1) % args.uncond_every_e == 0:
            run_val_inference_unconditional(
                model, device, save_dir=uncond_dir, epoch=epoch,
                T_seq=args.window_size, n_samples=args.uncond_n_samples,
                edge_index=edge_index, edge_features=edge_features,
                T_diff_val=args.T_diff_val, use_wandb=use_wandb)
            model.train()

    writer.close()
    if use_wandb:
        wandb.finish()
    print('Training done.')


if __name__ == '__main__':
    args = TrainDisContactOptions().parse()
    train(args)