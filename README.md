# Fine-Tuning ViTs

## Configure venv using uv
- [uv](https://github.com/astral-sh/uv)
- `uv venv <VENV_PATH> --python 3.12`
- `source <VENV_PATH>/bin/activate`
- `uv pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121`
- `uv pip install matplotlib==3.10.1 datasets==3.3.2 transformers==4.49.0 scikit-learn==1.6.1 python-dotenv==1.0.1 accelerate==1.4.0`

## Configure local env

- `cp .env.sample .env`
- modify HF_ env variables as needed

### Confirm venv
- `python test_env.py`

## Results

### google/vit-base-patch16-224

- consumed ~ ? GB of VRAM

```bash
TrainOutput(global_step=1350, training_loss=0.15539069290514346, metrics={'train_runtime': 137.439, 'train_samples_per_second': 98.225, 'train_steps_per_second': 9.823, 'total_flos': 1.046216869705728e+18, 'train_loss': 0.15539069290514346, 'epoch': 3.0})
```

### dinov2 w reg - facebook/dinov2_with_registers-base

- consumed ~ ? GB of VRAM

- in progress

### apple/aimv2-large-patch14-224

- consumed ~ 7 GB of VRAM

---

## Refs
- [Fine-Tuning Vision Transformer with Hugging Face and PyTorch](https://medium.com/@supersjgk/fine-tuning-vision-transformer-with-hugging-face-and-pytorch-df19839d5396)