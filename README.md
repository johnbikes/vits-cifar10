**a work in progress**

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

- consumed ~ 3 GB of VRAM

```bash
TrainOutput(global_step=1350, training_loss=0.15539069290514346, metrics={'train_runtime': 137.439, 'train_samples_per_second': 98.225, 'train_steps_per_second': 9.823, 'total_flos': 1.046216869705728e+18, 'train_loss': 0.15539069290514346, 'epoch': 3.0})
```

```bash
{'test_loss': 0.06985098868608475, 'test_accuracy': 0.978, 'test_runtime': 3.4191, 'test_samples_per_second': 292.471, 'test_steps_per_second': 73.118}
```

### dinov2 w reg - facebook/dinov2_with_registers-base

- consumed ~ ? GB of VRAM

- in progress

### apple/aimv2-large-patch14-224

- consumed ~ 7 GB of VRAM

- **please ignore aimv2 results till the correct base classes are used!**

```bash
TrainOutput(global_step=1350, training_loss=1.853396900318287, metrics={'train_runtime': 494.6235, 'train_samples_per_second': 27.293, 'train_steps_per_second': 2.729, 'total_flos': 2.928337017403392e+18, 'train_loss': 1.853396900318287, 'epoch': 3.0})
```

---

## Refs
- [Fine-Tuning Vision Transformer with Hugging Face and PyTorch](https://medium.com/@supersjgk/fine-tuning-vision-transformer-with-hugging-face-and-pytorch-df19839d5396)
- [HF DINOv2 with Registers](https://huggingface.co/docs/transformers/en/model_doc/dinov2_with_registers)
- [HF AIMv2 large-patch14-224](https://huggingface.co/apple/aimv2-large-patch14-224)