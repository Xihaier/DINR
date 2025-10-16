python src/train.py -m \
  model=turbulence_FFNet \
  model.ntk_analysis=false \
  model.checkpoint_epochs=[] \
  model.ablation_noise=true \
  model.noise_type=gaussian \
  model.noise_level=0.1 \
  seed=3407,0,1,2,3,4,5,6,7,8

python src/train.py -m \
  model=turbulence_FFNet \
  model.ntk_analysis=false \
  model.checkpoint_epochs=[] \
  model.ablation_noise=true \
  model.noise_type=gaussian \
  model.noise_level=0.2 \
  seed=3407,0,1,2,3,4,5,6,7,8

python src/train.py -m \
  model=turbulence_FFNet \
  model.ntk_analysis=false \
  model.checkpoint_epochs=[] \
  model.ablation_noise=true \
  model.noise_type=gaussian \
  model.noise_level=0.3 \
  seed=3407,0,1,2,3,4,5,6,7,8

python src/train.py -m \
  model=turbulence_OCFFNet \
  model.ntk_analysis=false \
  model.checkpoint_epochs=[] \
  model.ablation_noise=true \
  model.noise_type=gaussian \
  model.noise_level=0.1 \
  seed=3407,0,1,2,3,4,5,6,7,8

python src/train.py -m \
  model=turbulence_OCFFNet \
  model.ntk_analysis=false \
  model.checkpoint_epochs=[] \
  model.ablation_noise=true \
  model.noise_type=gaussian \
  model.noise_level=0.2 \
  seed=3407,0,1,2,3,4,5,6,7,8

python src/train.py -m \
  model=turbulence_OCFFNet \
  model.ntk_analysis=false \
  model.checkpoint_epochs=[] \
  model.ablation_noise=true \
  model.noise_type=gaussian \
  model.noise_level=0.3 \
  seed=3407,0,1,2,3,4,5,6,7,8

python src/train.py -m \
  model=turbulence_OCFFNet \
  model.ntk_analysis=false \
  model.checkpoint_epochs=[] \
  model.ablation_ot_loss=true \
  model.ablation_noise=true \
  model.noise_type=gaussian \
  model.noise_level=0.1 \
  seed=3407,0,1,2,3,4,5,6,7,8

python src/train.py -m \
  model=turbulence_OCFFNet \
  model.ntk_analysis=false \
  model.checkpoint_epochs=[] \
  model.ablation_ot_loss=true \
  model.ablation_noise=true \
  model.noise_type=gaussian \
  model.noise_level=0.2 \
  seed=3407,0,1,2,3,4,5,6,7,8

python src/train.py -m \
  model=turbulence_OCFFNet \
  model.ntk_analysis=false \
  model.checkpoint_epochs=[] \
  model.ablation_ot_loss=true \
  model.ablation_noise=true \
  model.noise_type=gaussian \
  model.noise_level=0.3 \
  seed=3407,0,1,2,3,4,5,6,7,8