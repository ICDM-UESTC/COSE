# Compose Yourself: Average-Velocity Flow Matching for One-Step Speech Enhancement
This is the implementation of Compose Yourself: Average-Velocity Flow Matching for One-Step Speech Enhancement.

# Paper Link

You can access the full paper on arXiv via the link below:

[View on arXiv](https://arxiv.org/abs/2509.15952)
## Environment Requirements
```
# create virtual environment
conda create --name COSE python=3.9.0

# activate environment
conda activate COSE

# install required packages
pip install -r requirements.txt
```
## How to train
```
python train.py --log_dir <path_to_model> --base_dir <path_to_dataset>
```
## How to evaluate
```
python enhancement.py --test_dir <path_to_noisy> --enhanced_dir <path_to_enhanced> --ckpt <path_to_model_checkpoint>

python calc_metrics.py --clean_dir <path_to_clean> --noisy_dir <path_to_noisy> --enhanced_dir <path_to_enhanced>
```
## Built Upon & Related Work

This repository  is built on previous outstanding works:

- 🔗 **[SGMSE]**-https://github.com/sp-uhh/sgmse

- 🔗 **[StoRM]**-https://github.com/sp-uhh/storm

- 🔗 **[FlowMSE]**-https://github.com/seongq/flowmse

**Note**: This work extends the above method through a one-step generation framework while retaining the complex STFT-based front-end data processing design.

## Pretrained Checkpoints (Coming Soon)

We will provide pretrained checkpoints trained on the **VoiceBank-DEMAND** dataset in the near future.
> **Stay tuned!** We will update this section as soon as the checkpoint is ready for public release.
