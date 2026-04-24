# jax-jetson-orin-nano-wheels

JAX 0.5.3 built from source for Jetson Orin Nano

- JetPack 6.2, CUDA 12.6, cuDNN 9.20.0, compute capability sm_87
- Python 3.10, aarch64
- Compatible flax stack: `chex` 0.1.90, `orbax-checkpoint` 0.11.0, `optax` 0.2.5, `flax` 0.10.0

Install the three wheels first, then the flax stack in that sequence. Don't pip install `cuda-tools` in the same venv

## Prerequisites

The system default gcc/clang on JetPack 6.2 isn't able to compile jaxlib. clang 17 is required (clang 14, the apt default, fails on ARM SVE). If rebuilding from source:

```bash
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 17
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-17 100
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-17 100
```

Using the prebuilt wheels here means you skip this entirely.

## Install

```bash
pip install jaxlib-*.whl jax_cuda12_plugin-*.whl jax_cuda12_pjrt-*.whl
pip install chex==0.1.90
pip install orbax-checkpoint==0.11.0
pip install optax==0.2.5
pip install flax==0.10.0
```

## Verify

```bash
python -c "import jax; print(jax.__version__); print(jax.devices())"
```

Expected output: `[CudaDevice(id=0)]`
