# DeepSDF notes

## Notes on building

```bash
build
cmake ..
make -j
```

Ref: https://github.com/facebookresearch/DeepSDF/issues/88

## Pre-process

```bash
export MESA_GL_VERSION_OVERRIDE=3.3 && export PANGOLIN_WINDOW_URI=headless://

# Debug
./bin/PreprocessMesh -m /media/data/data_extract/shapenet/ShapeNetCore.v2/02958343/1a0bc9ab92c915167ae33d942430658c/models/model_normalized.obj -o temp.npz

# Run with short list
# However, only a subset will finish successfully
# https://github.com/facebookresearch/DeepSDF/issues/5#issuecomment-521430369
python preprocess_data.py --data_dir data \
    --source /media/data/data_extract/shapenet/ShapeNetCore.v2/ \
    --name ShapeNetV2 \
    --split examples/splits/sv2_cars_debug.json \
    --threads 20 \
    --skip
python preprocess_data.py --data_dir data \
    --source /media/data/data_extract/shapenet/ShapeNetCore.v2/ \
    --name ShapeNetV2 \
    --split examples/splits/sv2_cars_debug.json \
    --surface \
    --threads 20 \
    --skip

# Run full dataset
python preprocess_data.py --data_dir data \
    --source /media/data/data_extract/shapenet/ShapeNetCore.v2/ \
    --name ShapeNetV2 \
    --split examples/splits/sv2_cars_full.json \
    --threads 20 \
    --skip
python preprocess_data.py --data_dir data \
    --source /media/data/data_extract/shapenet/ShapeNetCore.v2/ \
    --name ShapeNetV2 \
    --split examples/splits/sv2_cars_full.json \
    --surface \
    --threads 20 \
    --skip

# Generate list:
# - examples/splits/sv2_cars_valid.json # All valid items
# - examples/splits/sv2_cars_train.json # First 95% of the valid items
# - examples/splits/sv2_cars_test.json  # Last 5% of the valid items
python gen_valid_splits.py --get_legit
python gen_valid_splits.py --render_legit
python gen_valid_splits.py --gen_train_test_split
```

## Train

```bash
# Modify checkpoint and parameters in examples/cars/specs.json

# Automatic parallel on 2x24G cards
python train_deep_sdf.py -e examples/cars
```

## Test

```bash
# Output training set latent codes and meshes
python reconstruct.py -e examples/cars -c 1500 --split examples/splits/sv2_cars_train.json -d data --resolution 128
python reconstruct.py -e examples/cars -c 2000 --split examples/splits/sv2_cars_train.json -d data --resolution 128

# Outputs to examples/cars/Reconstructions
python reconstruct.py -e examples/cars -c 100 --split examples/splits/sv2_cars_test.json -d data --skip

# Visualize results
python explore_results.py --explore

# Convert all training latent codes to meshes
# Save to: examples/cars/LatentMeshes/2000/xxx.ply
python explore_results.py --convert
```
