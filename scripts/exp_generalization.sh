python src/train.py -m \
  data.generalization_test=true \
  data.generalization_train_percentage=0.75 \
  model=turbulence_FFNet \
  model.checkpoint_epochs=[] \
  seed=3407,0,1,2,3,4,5,6,7,8

python src/train.py -m \
  data.generalization_test=true \
  data.generalization_train_percentage=0.5 \
  model=turbulence_FFNet \
  model.checkpoint_epochs=[] \
  seed=3407,0,1,2,3,4,5,6,7,8

python src/train.py -m \
  data.generalization_test=true \
  data.generalization_train_percentage=0.25 \
  model=turbulence_FFNet \
  model.checkpoint_epochs=[] \
  seed=3407,0,1,2,3,4,5,6,7,8

python src/train.py -m \
  data.generalization_test=true \
  data.generalization_train_percentage=0.75 \
  model=turbulence_OCFFNet \
  model.checkpoint_epochs=[] \
  seed=3407,0,1,2,3,4,5,6,7,8

python src/train.py -m \
  data.generalization_test=true \
  data.generalization_train_percentage=0.5 \
  model=turbulence_OCFFNet \
  model.checkpoint_epochs=[] \
  seed=3407,0,1,2,3,4,5,6,7,8

python src/train.py -m \
  data.generalization_test=true \
  data.generalization_train_percentage=0.25 \
  model=turbulence_OCFFNet \
  model.checkpoint_epochs=[] \
  seed=3407,0,1,2,3,4,5,6,7,8

python src/train.py -m \
  data.generalization_test=true \
  data.generalization_train_percentage=0.75 \
  model=turbulence_OCFFNet \
  model.checkpoint_epochs=[] \
  model.ablation_ot_loss=true \
  seed=3407,0,1,2,3,4,5,6,7,8

python src/train.py -m \
  data.generalization_test=true \
  data.generalization_train_percentage=0.5 \
  model=turbulence_OCFFNet \
  model.checkpoint_epochs=[] \
  model.ablation_ot_loss=true \
  seed=3407,0,1,2,3,4,5,6,7,8

python src/train.py -m \
  data.generalization_test=true \
  data.generalization_train_percentage=0.25 \
  model=turbulence_OCFFNet \
  model.checkpoint_epochs=[] \
  model.ablation_ot_loss=true \
  seed=3407,0,1,2,3,4,5,6,7,8