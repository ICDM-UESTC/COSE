import time
from math import ceil
import warnings
import datetime
import torch
import pytorch_lightning as pl
import torch.distributed as dist
from torchaudio import load
from torch_ema import ExponentialMovingAverage
from librosa import resample
import numpy as np
from sgmse import sampling
from sgmse.sdes import SDERegistry
from sgmse.backbones import BackboneRegistry
from sgmse.util.inference import evaluate_model
from sgmse.util.other import pad_spec, si_sdr
from pesq import pesq
from pystoi import stoi
from torch_pesq import PesqLoss
import torch.nn.functional as F
from functools import partial

class ScoreModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=5e-5, help="The learning rate (1e-4 by default)")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum process time (0.03 by default)")
        parser.add_argument("--num_eval_files", type=int, default=20, help="Number of files for speech enhancement performance evaluation during training. Pass 0 to turn off (no checkpoints based on evaluation metrics will be generated).")
        parser.add_argument("--loss_type", type=str, default="mean_flow", help="The type of loss function to use.")
        parser.add_argument("--loss_weighting", type=str, default="sigma^2", help="The weighting of the loss function.")
        parser.add_argument("--network_scaling", type=str, default=None, help="The type of loss scaling to use.")
        parser.add_argument("--c_in", type=str, default="1", help="The input scaling for x.")
        parser.add_argument("--c_out", type=str, default="1", help="The output scaling.")
        parser.add_argument("--c_skip", type=str, default="0", help="The skip connection scaling.")
        parser.add_argument("--sigma_data", type=float, default=0.1, help="The data standard deviation.")
        parser.add_argument("--l1_weight", type=float, default=0.001, help="The balance between the time-frequency and time-domain losses.")
        parser.add_argument("--pesq_weight", type=float, default=0.0, help="The balance between the time-frequency and time-domain losses.")
        parser.add_argument("--sr", type=int, default=16000, help="The sample rate of the audio files.")
        return parser

    def __init__(
        self, backbone, sde, lr=1e-4, ema_decay=0.999, t_eps=0.03, num_eval_files=20, loss_type='rectified_flow', 
        loss_weighting='sigma^2', network_scaling=None, c_in='1', c_out='1', c_skip='0', sigma_data=0.1, 
        l1_weight=0.001, pesq_weight=0.0, sr=16000, data_module_cls=None, **kwargs
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: Backbone DNN that serves as a score-based model.
            sde: The SDE that defines the diffusion process.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            loss_type: The type of loss to use (wrt. noise z/std). Options are 'mse' (default), 'mae'
        """
        super().__init__()
        # Initialize Backbone DNN
        self.backbone = backbone
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        self.dnn = dnn_cls(**kwargs)
        # Initialize SDE
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)
        # Store hyperparams and save them
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.loss_type = loss_type
        self.loss_weighting = loss_weighting
        self.l1_weight = l1_weight
        self.pesq_weight = pesq_weight
        self.network_scaling = network_scaling
        self.c_in = c_in
        self.c_out = c_out
        self.c_skip = c_skip
        self.sigma_data = sigma_data
        self.num_eval_files = num_eval_files
        self.sr = sr
        self.p = 0.5
        self.c = 1e-3
        #self.jvp_fn = torch.autograd.functional.jvp
        self.jvp_fn = torch.func.jvp
        # Initialize PESQ loss if pesq_weight > 0.0
        if pesq_weight > 0.0:
            self.pesq_loss = PesqLoss(1.0, sample_rate=sr).eval()
            for param in self.pesq_loss.parameters():
                param.requires_grad = False
        self.save_hyperparameters(ignore=['no_wandb'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.dnn.parameters())

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.dnn.parameters())        # store current params in EMA
                self.ema.copy_to(self.dnn.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.dnn.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def _loss(self, forward_out, x_t):
        """
        Different loss functions can be used to train the score model, see the paper: 
        
        Julius Richter, Danilo de Oliveira, and Timo Gerkmann
        "Investigating Training Objectives for Generative Speech Enhancement"
        https://arxiv.org/abs/2409.10753

        """
        if self.loss_type == "rectified_flow":
            real_mse = torch.nn.functional.mse_loss(forward_out.real, x_t.real)
            imag_mse = torch.nn.functional.mse_loss(forward_out.imag, x_t.imag)

            total_loss = real_mse + imag_mse
        elif self.loss_type == 'mean_flow':
            err = forward_out - x_t
            err_mag_sq = err.abs() ** 2 
            delta_sq = torch.sum(err_mag_sq, dim=[1, 2])  
            w = 1.0 / (delta_sq + self.c) ** self.p
            w_detached = w.detach()
            loss = w_detached * delta_sq  # [B]
            total_loss = torch.mean(loss)  # scalar
    
        else:
            raise ValueError("Invalid loss type: {}".format(self.loss_type))

        return total_loss

    def _step(self, batch, batch_idx):
        if self.global_step % 2 == 0: 
            x, y = batch
            b = x.size(0)

            mu, sigma = -0.4, 1.0
            normal_samples = np.random.randn(x.size(0), 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))  # Apply sigmoid

            t_np = np.maximum(samples[:, 0], samples[:, 1])
            r_np = np.minimum(samples[:, 0], samples[:, 1])
            num_selected = 0
            indices = np.random.permutation(x.size(0))[:num_selected]
            r_np[indices] = t_np[indices]
            t = torch.tensor(t_np).to(x.device)
            r = torch.tensor(r_np).to(x.device)
            lambda_param = torch.rand(b, device=x.device)
            s = (1 - lambda_param) * t + lambda_param * r
            texp = t.view([b, *([1] * len(x.shape[1:]))])
            x_T = torch.randn_like(x)  
            x_t = (1 - texp) * x + texp * x_T
            v = x_T - x
            u2 = self.dnn(x_t, t, s, y=y)  
            x_s = x_t - (t - s).view([b, *([1] * len(x.shape[1:]))]) * u2
            u1 = self.dnn(x_s, s, r, y=y)
            u_tr = self.dnn(x_t, t, r, y=y)
            target_u = ((1 - lambda_param).view(-1, 1, 1, 1) * u1 + lambda_param.view(-1, 1, 1, 1) * u2).detach()
            loss_COSE = self._loss(forward_out = u_tr, x_t = target_u)

            return loss_COSE
        else:
            x, y = batch
            b = x.size(0)
            nt = torch.randn((b,)).to(x.device)
            t = torch.sigmoid(nt)
            texp = t.view([b, *([1] * len(x.shape[1:]))])

            x_T = torch.randn_like(x) 
            x_t = (1 - texp) * x + texp * x_T
            v = x_T - x
            forward_out = self.dnn(x_t, t, t, y)  
            loss = self._loss(forward_out = forward_out, x_t = v)  # 计算损失
            return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss
      
    def validation_step(self, batch, batch_idx):
        if self.num_eval_files != 0:
            clean_files = self.data_module.valid_set.clean_files[:self.num_eval_files]
            noisy_files = self.data_module.valid_set.noisy_files[:self.num_eval_files]  
            # Evaluate the performance of the model
            pesq_sum = 0; si_sdr_sum = 0; estoi_sum = 0; 
            for (clean_file, noisy_file) in zip(clean_files, noisy_files):
                # Load the clean and noisy speech
                x, sr_x = load(clean_file)
                x = x.squeeze().numpy()
                y, sr_y = load(noisy_file) 
                assert sr_x == sr_y, "Sample rates of clean and noisy files do not match!"
                # Resample if necessary
                if sr_x != 16000:
                    x_16k = resample(x, orig_sr=sr_x, target_sr=16000).squeeze()                 
                else:
                    x_16k = x
                x_hat = self.sample(x_16k, y, sr_x, self.dnn, pesq_func=pesq)
                if self.sr != 16000:
                    x_hat_16k = resample(x_hat, orig_sr=self.sr, target_sr=16000).squeeze()
                else:
                    x_hat_16k = x_hat    
                pesq_sum += pesq(16000, x_16k, x_hat_16k, 'wb') 
                si_sdr_sum += si_sdr(x, x_hat)
                estoi_sum += stoi(x, x_hat, self.sr, extended=True)

                pesq_avg = pesq_sum / len(clean_files)
                si_sdr_avg = si_sdr_sum / len(clean_files)
                estoi_avg = estoi_sum / len(clean_files)

        print("pesq_avg:", pesq_avg)
        print("si_sdr:", si_sdr_avg)
        print("estoi:", estoi_avg)
        self.log('pesq', pesq_avg, on_step=False, on_epoch=True, sync_dist=True)
        self.log('si_sdr', si_sdr_avg, on_step=False, on_epoch=True, sync_dist=True)
        self.log('estoi', estoi_avg, on_step=False, on_epoch=True, sync_dist=True)
            
        loss = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def sample(self, clean_16k, cond, sr_clean, score_model, sample_steps=2, cfg=2.0, pesq_func=pesq):
        T_orig = cond.size(1) 
        norm_factor = cond.abs().max().item()
        cond = cond / norm_factor
        cond = torch.unsqueeze(self._forward_transform(self._stft(cond.cuda())), 0)
        cond = pad_spec(cond)
        z = torch.randn_like(cond) 
        b = z.size(0)
        dt = 1 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        
        with torch.no_grad():
            for i in range(sample_steps, 0, -1):
                if i != 1:
                    t = i / sample_steps
                    r = t - dt
                    t = torch.tensor([t] * b).to(z.device)
                    r = torch.tensor([r] * b).to(z.device)
                    vc = self.dnn(z, t, r, cond) 
                    z = z - dt * vc
                else:
                    t = i / sample_steps
                    t = torch.tensor([t] * b).to(z.device)
                    r = torch.tensor([0.003] * b).to(z.device)
                    vc = self.dnn(z, t, r, cond) 
                    z = z - dt * vc
            x_hat = self.to_audio(z.squeeze(), T_orig)
            x_hat = x_hat * norm_factor
        return x_hat

    
    def samplefortest_COSE(self, cond, score_model, sample_steps=1, pesq_func=pesq):
        T_orig = cond.size(1) 
        norm_factor = cond.abs().max().item()
        cond = cond / norm_factor
        cond = torch.unsqueeze(self._forward_transform(self._stft(cond.cuda())), 0)
        cond = pad_spec(cond)
        z = torch.randn_like(cond).to(cond.device)  

        b = z.size(0)
        dt = 1 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])

        with torch.no_grad():
            for i in range(sample_steps, 0, -1):
                if i != 1:
                    t = i / sample_steps
                    r = t - dt
                    t = torch.tensor([t] * b).to(z.device)
                    r = torch.tensor([r] * b).to(z.device)
                    vc = self.dnn(z, t, r, cond) 
                    z = z - dt * vc
                else:
                    t = i / sample_steps
                    t = torch.tensor([t] * b).to(z.device)
                    r = torch.tensor([0.003] * b).to(z.device)
                    vc = self.dnn(z, t, r, cond) 
                    z = z - dt * vc
            x_hat = self.to_audio(z.squeeze(), T_orig)
            x_hat = x_hat * norm_factor
        return x_hat

    def forward(self, x_t, y, t, r):
        """
        The model forward pass. In [1] and [2], the model estimates the score function. In [3], the model estimates 
        either the score function or the target data for the Schrödinger bridge (loss_type='data_prediction').
        
        [1] Julius Richter, Simon Welker, Jean-Marie Lemercier, Bunlong Lay, and  Timo Gerkmann 
            "Speech Enhancement and Dereverberation with Diffusion-Based Generative Models"
            IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 2351-2364, 2023. 

        [2] Julius Richter, Yi-Chiao Wu, Steven Krenn, Simon Welker, Bunlong Lay, Shinji Watanabe, Alexander Richard, and Timo Gerkmann
            "EARS: An Anechoic Fullband Speech Dataset Benchmarked for Speech Enhancement and Dereverberation"
            ISCA Interspecch, Kos, Greece, Sept. 2024. 

        [3] Julius Richter, Danilo de Oliveira, and Timo Gerkmann
            "Investigating Training Objectives for Generative Speech Enhancement"
            https://arxiv.org/abs/2409.10753

        """

        # In [3], we use new code with backbone='ncsnpp_v2':
        if self.backbone == "ncsnpp_v2":
            F = self.dnn(self._c_in(t) * x_t, self._c_in(t) * y, t)
            
            # Scaling the network output, see below Eq. (7) in the paper
            if self.network_scaling == "1/sigma":
                std = self.sde._std(t)
                F = F / std[:, None, None, None]
            elif self.network_scaling == "1/t":
                F = F / t[:, None, None, None]

            # The loss type determines the output of the model
            if self.loss_type == "score_matching":
                score = self._c_skip(t) * x_t + self._c_out(t) * F
                return score
            elif self.loss_type == "denoiser":
                sigmas = self.sde._std(t)[:, None, None, None]
                score = (F - x_t) / sigmas.pow(2)
                return score
            elif self.loss_type == 'data_prediction':
                x_hat = self._c_skip(t) * x_t + self._c_out(t) * F
                return x_hat
            elif self.loss_type == 'rectified_flow':
                
                x_hat = self.dnn(x_t,t,r)
                return x_hat    
        # In [1] and [2], we use the old code:
        else:
      
            dnn_input = torch.cat([x_t, y], dim=1)          
            score = self.dnn(dnn_input, t, r)
            return score

    def _c_in(self, t):
        if self.c_in == "1":
            return 1.0
        elif self.c_in == "edm":
            sigma = self.sde._std(t)
            return (1.0 / torch.sqrt(sigma**2 + self.sigma_data**2))[:, None, None, None]
        else:
            raise ValueError("Invalid c_in type: {}".format(self.c_in))
    
    def _c_out(self, t):
        if self.c_out == "1":
            return 1.0
        elif self.c_out == "sigma":
            return self.sde._std(t)[:, None, None, None]
        elif self.c_out == "1/sigma":
            return 1.0 / self.sde._std(t)[:, None, None, None] 
        elif self.c_out == "edm":
            sigma = self.sde._std(t)
            return ((sigma * self.sigma_data) / torch.sqrt(self.sigma_data**2 + sigma**2))[:, None, None, None]
        else:
            raise ValueError("Invalid c_out type: {}".format(self.c_out))
    
    def _c_skip(self, t):
        if self.c_skip == "0":
            return 0.0
        elif self.c_skip == "edm":
            sigma = self.sde._std(t)
            return (self.sigma_data**2 / (sigma**2 + self.sigma_data**2))[:, None, None, None]
        else:
            raise ValueError("Invalid c_skip type: {}".format(self.c_skip))

    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def get_pc_sampler(self, predictor_name, corrector_name, y, N=None, minibatch=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns
            return batched_sampling_fn

    def get_ode_sampler(self, y, N=None, minibatch=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_ode_sampler(sde, self, y=y, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_ode_sampler(sde, self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return sample, ns
            return batched_sampling_fn

    def get_sb_sampler(self, sde, y, sampler_type="ode", N=None, **kwargs):
        N = sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N if N is not None else sde.N

        return sampling.get_sb_sampler(sde, self, y=y, sampler_type=sampler_type, **kwargs)

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)

    