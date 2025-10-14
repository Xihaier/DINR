# python src/train.py \
#   model=turbulence_FFNet \
#   model.ntk_analysis=False \
#   model.checkpoint_epochs=[] \
#   model.ablation_noise=False

python src/train.py -m \
  model=turbulence_FFNet \
  model.ntk_analysis=false \
  model.checkpoint_epochs=[] \
  model.ablation_noise=true \
  model.noise_type=gaussian \
  model.noise_level=0.01,0.05,0.1 \
  seed=0,1,2,3,4,5,6,7,8,3407

python src/train.py -m \
  model=turbulence_OCFFNet \
  model.ntk_analysis=false \
  model.checkpoint_epochs=[] \
  model.ablation_noise=true \
  model.noise_type=gaussian \
  model.noise_level=0.01,0.05,0.1 \
  seed=0,1,2,3,4,5,6,7,8,3407

python src/train.py -m \
  model=turbulence_OCFFNet \
  model.ntk_analysis=false \
  model.checkpoint_epochs=[] \
  model.ablation_ot_loss=true \
  model.ablation_noise=true \
  model.noise_type=gaussian \
  model.noise_level=0.01,0.05,0.1 \
  seed=0,1,2,3,4,5,6,7,8,3407