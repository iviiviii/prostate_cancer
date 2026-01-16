# MNIST artifact demo

This folder contains a small demo to run the existing DDPM artifact pipeline on MNIST.

- `main_art_mnist.py` : main training script. Example:

```bash
python mnist_artifact/main_art_mnist.py --output-dir mnist_artifact/runs/test_run --num-train-images 16 --image-size 64 --train-steps 2000
```

The script reuses the `ddpm` package in the parent directory and writes results to `--output-dir`.
