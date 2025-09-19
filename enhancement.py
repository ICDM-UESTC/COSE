import glob
import torch
from tqdm import tqdm
from os import makedirs
from soundfile import write
from torchaudio import load
from os.path import join, dirname, basename
from argparse import ArgumentParser
from librosa import resample
from pesq import pesq
# Set CUDA architecture list
from sgmse.util.other import set_torch_cuda_arch_list
set_torch_cuda_arch_list()
import os
import shutil
from sgmse.model import ScoreModel
from sgmse.util.other import pad_spec


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True, help='Directory containing the test data (noisy)')
    parser.add_argument("--enhanced_dir", type=str, required=True, help='Directory containing the enhanced data')
    parser.add_argument("--ckpt", type=str,  help='Path to model checkpoint')
    parser.add_argument("--N", type=int, default=1, help="Number of reverse steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference")
    parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum process time (0.03 by default)")
    args = parser.parse_args()

    if os.path.exists(args.enhanced_dir):
       shutil.rmtree(args.enhanced_dir)  # Delete the folder and all its contents
    os.makedirs(args.enhanced_dir) 
    # Load score model 
    model = ScoreModel.load_from_checkpoint(args.ckpt, map_location=args.device)
    model.t_eps = args.t_eps
    model.eval()

    # Get list of noisy files
    noisy_files = []
    noisy_files += sorted(glob.glob(join(args.test_dir, '*.wav')))
    noisy_files += sorted(glob.glob(join(args.test_dir, '**', '*.wav')))
    noisy_files += sorted(glob.glob(join(args.test_dir, '*.flac')))
    noisy_files += sorted(glob.glob(join(args.test_dir, '**', '*.flac')))

    # Check if the model is trained on 48 kHz data

    target_sr = 16000
    pad_mode = "zero_pad"

    # Enhance files
    for noisy_file in tqdm(noisy_files):
        filename = basename(noisy_file)

        
        # Load noisy and clean wavs
        y, sr = load(noisy_file)  # Noisy

        
        # Resample to target sample rate if needed
        if sr != target_sr:
            y = torch.tensor(resample(y.numpy(), orig_sr=sr, target_sr=target_sr))

        
        # Convert clean speech to numpy array
        # Reverse sampling with clean speech as first argument
        x_hat = model.samplefortest_COSE(y, model.dnn, sample_steps=args.N, pesq_func=pesq)
        
        # Write enhanced wav file
        makedirs(dirname(join(args.enhanced_dir, filename)), exist_ok=True)
        write(join(args.enhanced_dir, filename), x_hat.cpu().numpy(), target_sr)