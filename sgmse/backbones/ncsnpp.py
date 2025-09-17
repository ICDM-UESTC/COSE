# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file

from .ncsnpp_utils import layers, layerspp, normalization
import torch.nn as nn
import functools
import torch
import numpy as np

from .shared import BackboneRegistry

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init


@BackboneRegistry.register("ncsnpp")
class NCSNpp(nn.Module):
    """NCSN++ model, adapted from https://github.com/yang-song/score_sde repository"""

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--ch_mult",type=int, nargs='+', default=[1,1,2,2,2,2,2])
        parser.add_argument("--num_res_blocks", type=int, default=2)
        parser.add_argument("--attn_resolutions", type=int, nargs='+', default=[16])
        parser.add_argument("--no-centered", dest="centered", action="store_false", help="The data is not centered [-1, 1]")
        parser.add_argument("--centered", dest="centered", action="store_true", help="The data is centered [-1, 1]")
        parser.set_defaults(centered=True)
        return parser

    def __init__(self,
        scale_by_sigma = True,
        nonlinearity = 'swish',
        nf = 128,
        ch_mult = (1, 1, 2, 2, 2, 2, 2),
        num_res_blocks = 2,
        attn_resolutions = (16,),
        resamp_with_conv = True,
        conditional = True,
        fir = False,
        fir_kernel = [1, 3, 3, 1],
        skip_rescale = True,
        resblock_type = 'biggan',
        progressive = 'output_skip',
        progressive_input = 'input_skip',
        progressive_combine = 'sum',
        init_scale = 0.,
        fourier_scale = 16,
        image_size = 256,
        embedding_type = 'fourier',
        dropout = .0,
        centered = True,
        **unused_kwargs
    ):
        super().__init__()
        self.act = act = get_act(nonlinearity)

        self.nf = nf = nf
        ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions = attn_resolutions
        dropout = dropout
        resamp_with_conv = resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [image_size // (2 ** i) for i in range(num_resolutions)]

        self.conditional = conditional = conditional  # noise-conditional
        self.centered = centered
        self.scale_by_sigma = scale_by_sigma

        fir = fir
        fir_kernel = fir_kernel
        self.skip_rescale = skip_rescale = skip_rescale
        self.resblock_type = resblock_type = resblock_type.lower()
        self.progressive = progressive = progressive.lower()
        self.progressive_input = progressive_input = progressive_input.lower()
        self.embedding_type = embedding_type = embedding_type.lower()
        init_scale = init_scale
        assert progressive in ['none', 'output_skip', 'residual']
        assert progressive_input in ['none', 'input_skip', 'residual']
        assert embedding_type in ['fourier', 'positional']
        combine_method = progressive_combine.lower()
        combiner = functools.partial(Combine, method=combine_method)

        num_channels = 4  # x.real, x.imag, y.real, y.imag
        self.output_layer = nn.Conv2d(num_channels, 2, 1)

        modules = []
        # timestep/noise_level embedding
        if embedding_type == 'fourier':
            # Gaussian Fourier features embeddings for 't'
            self.fourier_projection_t = layerspp.GaussianFourierProjection(
                embedding_size=nf, scale=fourier_scale
            )
            # Gaussian Fourier features embeddings for 'r'
            self.fourier_projection_r = layerspp.GaussianFourierProjection(
                embedding_size=nf, scale=fourier_scale
            )
            embed_dim = 2 * nf * 2  # Double the embedding size to account for t and r
        elif embedding_type == 'positional':
            embed_dim = nf * 2 # Double for t and r
        else:
            raise ValueError(f'embedding type {embedding_type} unknown.')

        if conditional:
            # Linear layers for 't' embedding
            self.linear_t1 = nn.Linear(2 * nf, nf * 4)
            self.linear_t1.weight.data = default_initializer()(self.linear_t1.weight.shape)
            nn.init.zeros_(self.linear_t1.bias)
            self.linear_t2 = nn.Linear(nf * 4, nf * 4)
            self.linear_t2.weight.data = default_initializer()(self.linear_t2.weight.shape)
            nn.init.zeros_(self.linear_t2.bias)

            # Linear layers for 'r' embedding
            self.linear_r1 = nn.Linear(2 * nf, nf * 4)
            self.linear_r1.weight.data = default_initializer()(self.linear_r1.weight.shape)
            nn.init.zeros_(self.linear_r1.bias)
            self.linear_r2 = nn.Linear(nf * 4, nf * 4)
            self.linear_r2.weight.data = default_initializer()(self.linear_r2.weight.shape)
            nn.init.zeros_(self.linear_r2.bias)

            # Final combined linear layer
            self.final_linear = nn.Linear(nf * 4 * 2, nf * 4) # Input is 2 * (nf * 4) from t and r branches
            self.final_linear.weight.data = default_initializer()(self.final_linear.weight.shape)
            nn.init.zeros_(self.final_linear.bias)

        AttnBlock = functools.partial(layerspp.AttnBlockpp,
            init_scale=init_scale, skip_rescale=skip_rescale)

        Upsample = functools.partial(layerspp.Upsample,
            with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive == 'output_skip':
            self.pyramid_upsample = layerspp.Upsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive == 'residual':
            pyramid_upsample = functools.partial(layerspp.Upsample, fir=fir,
                fir_kernel=fir_kernel, with_conv=True)

        Downsample = functools.partial(layerspp.Downsample, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive_input == 'input_skip':
            self.pyramid_downsample = layerspp.Downsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive_input == 'residual':
            pyramid_downsample = functools.partial(layerspp.Downsample,
                fir=fir, fir_kernel=fir_kernel, with_conv=True)

        if resblock_type == 'ddpm':
            ResnetBlock = functools.partial(ResnetBlockDDPM, act=act,
                dropout=dropout, init_scale=init_scale,
                skip_rescale=skip_rescale, temb_dim=nf * 4) # temb_dim remains nf*4, as it's the combined embedding size

        elif resblock_type == 'biggan':
            ResnetBlock = functools.partial(ResnetBlockBigGAN, act=act,
                dropout=dropout, fir=fir, fir_kernel=fir_kernel,
                init_scale=init_scale, skip_rescale=skip_rescale, temb_dim=nf * 4) # temb_dim remains nf*4

        else:
            raise ValueError(f'resblock type {resblock_type} unrecognized.')

        # Downsampling block

        channels = num_channels
        if progressive_input != 'none':
            input_pyramid_ch = channels

        modules.append(conv3x3(channels, nf))
        hs_c = [nf]

        in_ch = nf
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                if resblock_type == 'ddpm':
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))

                if progressive_input == 'input_skip':
                    modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                    if combine_method == 'cat':
                        in_ch *= 2

                elif progressive_input == 'residual':
                    modules.append(pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch))
                    input_pyramid_ch = in_ch

                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        pyramid_ch = 0
        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):  # +1 blocks in upsampling because of skip connection from combiner (after downsampling)
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            if progressive != 'none':
                if i_level == num_resolutions - 1:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                            num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == 'residual':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, in_ch, bias=True))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name.')
                else:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                            num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, channels, bias=True, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == 'residual':
                        modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name')

            if i_level != 0:
                if resblock_type == 'ddpm':
                    modules.append(Upsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, up=True))

        assert not hs_c

        if progressive != 'output_skip':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                         num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, channels, init_scale=init_scale))

        self.all_modules = nn.ModuleList(modules)
        

    def forward(self, x, t_cond, r_cond, y = None): # Changed time_cond to t_cond and r_cond
        # timestep/noise_level embedding; only for continuous training
        x = torch.cat([x, y], dim=1)  
        
        modules = self.all_modules
        m_idx = 0
       
        # Convert real and imaginary parts of (x,y) into four channel dimensions
        x = torch.cat((x[:,[0],:,:].real, x[:,[0],:,:].imag,
                       x[:,[1],:,:].real, x[:,[1],:,:].imag), dim=1)

        if self.embedding_type == 'fourier':
            # Gaussian Fourier features embeddings for t and r
            used_sigmas_t = t_cond
            used_sigmas_r = r_cond

            temb_t = self.fourier_projection_t(torch.log(used_sigmas_t))
            temb_r = self.fourier_projection_r(torch.log(used_sigmas_r))

        elif self.embedding_type == 'positional':
            # Sinusoidal positional embeddings.
            temb_t = layers.get_timestep_embedding(t_cond, self.nf)
            temb_r = layers.get_timestep_embedding(r_cond, self.nf)

        else:
            raise ValueError(f'embedding type {self.embedding_type} unknown.')

        if self.conditional:
            temb_t = self.linear_t1(temb_t)
            temb_t = self.linear_t2(self.act(temb_t))

            temb_r = self.linear_r1(temb_r)
            temb_r = self.linear_r2(self.act(temb_r))
            
            # Concatenate t and r embeddings and pass through a final linear layer
            temb = self.final_linear(torch.cat([temb_t, temb_r], dim=-1))
            temb = self.act(temb) # Apply activation after final linear layer
        else:
            temb = None

        if not self.centered:
            # If input data is in [0, 1]
            x = 2 * x - 1.

        # Downsampling block
        input_pyramid = None
        if self.progressive_input != 'none':
            input_pyramid = x

        # Input layer: Conv2d: 4ch -> 128ch
        hs = [modules[m_idx](x)]
        m_idx += 1
        
        # Down path in U-Net
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                # Attention layer (optional)
                if h.shape[-2] in self.attn_resolutions: # edit: check H dim (-2) not W dim (-1)
                    h = modules[m_idx](h)
                    m_idx += 1
                hs.append(h)

            # Downsampling
            if i_level != self.num_resolutions - 1:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](hs[-1])
                    m_idx += 1
                else:
                    h = modules[m_idx](hs[-1], temb)
                    m_idx += 1

                if self.progressive_input == 'input_skip':   # Combine h with x
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1

                elif self.progressive_input == 'residual':
                    input_pyramid = modules[m_idx](input_pyramid)
                    m_idx += 1
                    if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid
                hs.append(h)

        h = hs[-1] # actualy equal to: h = h
        h = modules[m_idx](h, temb)  # ResNet block
        m_idx += 1
        h = modules[m_idx](h)  # Attention block
        m_idx += 1
        h = modules[m_idx](h, temb)  # ResNet block
        m_idx += 1

        pyramid = None
      
        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1

            # edit: from -1 to -2
            if h.shape[-2] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if self.progressive != 'none':
                if i_level == self.num_resolutions - 1:
                    if self.progressive == 'output_skip':
                        pyramid = self.act(modules[m_idx](h))  # GroupNorm
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)  # Conv2D: 256 -> 4
                        m_idx += 1
                    elif self.progressive == 'residual':
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    else:
                        raise ValueError(f'{self.progressive} is not a valid name.')
                else:
                    if self.progressive == 'output_skip':
                        pyramid = self.pyramid_upsample(pyramid)  # Upsample
                        pyramid_h = self.act(modules[m_idx](h))  # GroupNorm
                        m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h)
                        m_idx += 1
                        pyramid = pyramid + pyramid_h
                    elif self.progressive == 'residual':
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                        if self.skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.)
                        else:
                            pyramid = pyramid + h
                        h = pyramid
                    else:
                        raise ValueError(f'{self.progressive} is not a valid name')

            # Upsampling Layer
            if i_level != 0:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](h)
                    m_idx += 1
                else:
                    h = modules[m_idx](h, temb)  # Upspampling
                    m_idx += 1

        assert not hs

        if self.progressive == 'output_skip':
            h = pyramid
        else:
            h = self.act(modules[m_idx](h))
            m_idx += 1
            h = modules[m_idx](h)
            m_idx += 1

        assert m_idx == len(modules), "Implementation error"
        if self.scale_by_sigma:
            # Assuming t_cond is the relevant "sigma" for scaling the output
            used_sigmas_t = t_cond.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
            h = h / used_sigmas_t

        # Convert back to complex number
        h = self.output_layer(h)
        h = torch.permute(h, (0, 2, 3, 1)).contiguous()
        h = torch.view_as_complex(h)[:,None, :, :]
        return h