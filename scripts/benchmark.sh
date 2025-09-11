# Benchmarking the models on the turbulence dataset
python src/train.py model=turbulence_FFNet
python src/train.py model=turbulence_SIREN
python src/train.py model=turbulence_OCFFNet
python src/train.py model=turbulence_OCSIREN