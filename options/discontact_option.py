import argparse
import os

import torch


class DisContactBaseOptions:
    """Arguments shared by training and inference."""

    is_train = False

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    # -- shared groups ------------------------------------------------------

    def initialize(self):
        p = self.parser

        g = p.add_argument_group('run')
        g.add_argument('--name', type=str, default='debug_diffusion',
                       help='Name of this trial; run dir is logs/<dataset>/<name>')
        g.add_argument('--checkpoints_dir', type=str, default='./logs',
                       help='Models are saved here')
        g.add_argument('--dataset_name', type=str, default='holo',
                       choices=['holo', 'arctic'], help='Dataset name')
        g.add_argument('--gpu_id', type=int, default=0, help='GPU id')
        g.add_argument('--seed', type=int, default=42, help='Random seed')
        g.add_argument('--debug', action='store_true',
                       help='Run a couple of iterations per epoch only')

        g = p.add_argument_group('model - denoiser architecture '
                                 '(must match between training and inference)')
        g.add_argument('--latent_dim',              type=int,   default=128,
                       help='Dimension of transformer latent')
        g.add_argument('--ff_size',                 type=int,   default=1024,
                       help='Feed-forward size')
        g.add_argument('--num_layers',              type=int,   default=6,
                       help='Number of attention layers')
        g.add_argument('--num_heads',               type=int,   default=4,
                       help='Number of attention heads')
        g.add_argument('--dropout',                 type=float, default=0.1,
                       help='Dropout ratio in transformer')
        g.add_argument('--window_size',             type=int,   default=75,
                       help='Sequence window length; also sets denoiser max_len')
        g.add_argument('--contact_temporal_layers', type=int,   default=2,
                       help='Number of per-vertex temporal conv blocks')
        g.add_argument('--contact_graph_layers',    type=int,   default=2,
                       help='Number of sparse mesh graph layers')
        g.add_argument('--contact_graph_heads',     type=int,   default=4,
                       help='Number of heads in mesh graph attention')
        g.add_argument('--contact_kernel_size',     type=int,   default=5,
                       help='Temporal conv kernel size')
        g.add_argument('--grad_m2c', type=float, default=0.0,
                       help='Gradient scale motion->contact (0=detach, 1=full)')
        g.add_argument('--grad_c2m', type=float, default=0.0,
                       help='Gradient scale contact->motion (0=detach, 1=full)')

        g = p.add_argument_group('mesh - precomputed MANO structures')
        g.add_argument('--mano_edge_index_path',    type=str, default='mano_edge_index.pt',
                       help='Built by scripts/build_mano_edges.py')
        g.add_argument('--mano_edge_features_path', type=str, default='mano_edge_features.pt',
                       help='Static edge features; if missing, FiLM edge bias is disabled')
        g.add_argument('--mano_region_map_path',    type=str, default='mano_region_map.pt',
                       help='If missing, region losses are disabled')
        g.add_argument('--vertex_marginals_path',   type=str, default=None,
                       help='If unset, marginals are computed on the train set')

        g = p.add_argument_group('diffusion')
        g.add_argument('--T_diffusion', type=int, default=1000,
                       help='Number of diffusion steps used at training time')
        g.add_argument('--T_diff_val',  type=int, default=1000,
                       help='Number of diffusion steps used at validation sampling')

        self.initialized = True

    # -- parse --------------------------------------------------------------

    def parse(self):
        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()
        self.opt.is_train = self.is_train

        if self.opt.gpu_id != -1 and torch.cuda.is_available():
            torch.cuda.set_device(self.opt.gpu_id)

        args = vars(self.opt)
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        if self.is_train:
            expr_dir = os.path.join(self.opt.checkpoints_dir,
                                    self.opt.dataset_name, self.opt.name)
            os.makedirs(expr_dir, exist_ok=True)
            with open(os.path.join(expr_dir, 'opt.txt'), 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')

        return self.opt


class TrainDisContactOptions(DisContactBaseOptions):

    is_train = True

    def initialize(self):
        DisContactBaseOptions.initialize(self)
        p = self.parser

        g = p.add_argument_group('data')
        g.add_argument('--setting', type=str, default=None,
                       choices=[None, 'categories', 'tasks', 'actions', 'instances'],
                       help='Evaluation split protocol; set it explicitly per run')
        g.add_argument('--batch_size',  type=int, default=8, help='Batch size')
        g.add_argument('--num_workers', type=int, default=4,
                       help='Dataloader workers')

        g = p.add_argument_group('loss - term weights')
        g.add_argument('--w_motion',         type=float, default=1.0,
                       help='L2 on MANO parameters (6D)')
        g.add_argument('--w_motion_pos',     type=float, default=1.0,
                       help='L2 on FK joint positions')
        g.add_argument('--w_contact_vb',     type=float, default=1.0,
                       help='D3PM variational lower bound')
        g.add_argument('--w_contact_aux',    type=float, default=1.0,
                       help='Vertex-wise BCE')
        g.add_argument('--w_region_density', type=float, default=0.1,
                       help='Per-region contact density MSE')

        g = p.add_argument_group('optim')
        g.add_argument('--lr',            type=float, default=1e-4, help='Learning rate')
        g.add_argument('--max_epoch',     type=int,   default=2000,
                       help='Maximum number of epochs for training')
        g.add_argument('--weight_decay',  type=float, default=0.0,
                       help='AdamW weight decay')
        g.add_argument('--use_scheduler', action='store_true',
                       help='Enable warmup + MultiStepLR; otherwise fixed lr')
        g.add_argument('--warm_up_iter',  type=int,   default=2000,
                       help='Number of warmup iterations')
        g.add_argument('--milestones',    type=int,   nargs='+', default=[150000, 250000],
                       help='Learning rate schedule (iterations)')
        g.add_argument('--gamma',         type=float, default=0.1,
                       help='Learning rate schedule factor')

        g = p.add_argument_group('logging')
        g.add_argument('--log_every',        type=int, default=50,
                       help='Frequency of printing training progress (iterations)')
        g.add_argument('--save_latest',      type=int, default=500,
                       help='Frequency of saving checkpoint (iterations)')
        g.add_argument('--val_every_e',      type=int, default=15,
                       help='Frequency of validation (epochs)')
        g.add_argument('--val_max_samples',  type=int, default=5,
                       help='Number of validation samples for conditional sampling')
        g.add_argument('--uncond_every_e',   type=int, default=1000,
                       help='Frequency of unconditional sanity sampling (epochs)')
        g.add_argument('--uncond_n_samples', type=int, default=4,
                       help='Number of unconditional samples')

        g = p.add_argument_group('logging/video - MANO rendering')
        g.add_argument('--make_video',    action='store_true',
                       help='Render a MANO video at validation time')
        g.add_argument('--video_every_e', type=int, default=30,
                       help='Frequency of video rendering (epochs)')
        g.add_argument('--video_fps',     type=int, default=10,
                       help='Frames per second of the rendered video')
        g.add_argument('--video_res',     type=int, default=512,
                       help='Render resolution in pixels')

        g = p.add_argument_group('logging/wandb')
        g.add_argument('--wandb_project', type=str, default='hoi-diffusion',
                       help='Weights & Biases project')
        g.add_argument('--wandb_entity',  type=str, default=None,
                       help='Weights & Biases entity')
        g.add_argument('--no_wandb',      action='store_true',
                       help='Disable Weights & Biases logging')

        g = p.add_argument_group('resume')
        g.add_argument('--is_continue', action='store_true',
                       help='Is this trial continuing previous state? (latest.tar)')


class DemoDisContactOptions(DisContactBaseOptions):

    is_train = False

    def initialize(self):
        DisContactBaseOptions.initialize(self)
        p = self.parser

        g = p.add_argument_group('input')
        g.add_argument('--img',    type=str, required=True, help='Input image path')
        g.add_argument('--text',   type=str, required=True, help='e.g. "open cap"')
        g.add_argument('--ckpt',   type=str, required=True, help='Trained checkpoint')
        g.add_argument('--root',   type=str, default=os.environ.get('ROOT_DIR', '.'),
                       help='Repository root holding the MANO mesh files')
        g.add_argument('--ranges', type=str, default='./demo/model/ranges.pkl',
                       help='Voxel grid ranges used at training time')
        g.add_argument('--contact_grid', type=int, default=16,
                       help='Contact voxel grid resolution')

        g = p.add_argument_group('detection - contact point localization')
        g.add_argument('--point', type=int, nargs=2, default=None,
                       help='Pixel (u v); if unset, detected automatically from --text')
        g.add_argument('--query',     type=str,   default=None,
                       help='Detection query; defaults to the noun parsed from --text')
        g.add_argument('--det_model', type=str,   default='google/owlv2-base-patch16-ensemble',
                       help='Open-vocabulary detector')
        g.add_argument('--det_thr',   type=float, default=0.05,
                       help='Detection score threshold')
        g.add_argument('--det_mode',  type=str,   default='center',
                       choices=['center', 'near'],
                       help='center = box center; near = closest surface pixel in box')

        g = p.add_argument_group('sampling')
        g.add_argument('--n_frames',     type=int,   default=30,
                       help='Number of generated frames')
        g.add_argument('--T_diff_steps', type=int,   default=None,
                       help='Reduced number of reverse steps; None = full T_diffusion')
        g.add_argument('--contact_thr',  type=float, default=0.5,
                       help='Threshold on contact probabilities')
        g.add_argument('--shift_to_contact', action='store_true',
                       help='Add the 3D contact point to cam_t before rendering')

        g = p.add_argument_group('render')
        g.add_argument('--save_dir', type=str, default='./demo/out_discontact',
                       help='Output directory')
        g.add_argument('--fps',      type=int, default=10,
                       help='Frames per second of the rendered videos')

        
        p.set_defaults(dropout=0.0)