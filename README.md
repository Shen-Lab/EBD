# Equivariant Blurring Diffusion


## Environments
```bash
conda env create -f environment.yml
# Activate the environment
conda activate ebd
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter==2.0.6 -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
pip install torch-sparse==0.6.11 -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
pip install torch-cluster==1.5.9 -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
pip install torch-spline-conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
pip install torch-geometric==1.7.2
pip install easydict==1.11
pip install tensorboard==2.11.2
pip install rdkit==2023.3.2
pip install rmsd==1.5.0
```

## Preprocessing
1. Make folder `logs/`.
2. Download folder from [[here]](https://drive.google.com/file/d/1SZN7py721pN6Vo3Khwv1o839BJCAnlkM/view?usp=share_link) and unzip in `data/GEOM/`.
3. Download folder from [[here]](https://drive.google.com/file/d/1AVXvmOL89IXUNh7CO4TUIfoZfSEVB28w/view?usp=share_link) and unzip in `trained_model/`.


## Train
```bash
python train_atom.py ./config/drugs_default.yml
```


## Generation
Use the provided checkpoint or the checkpoint from your own trained model to generate conformers.
Example when using the provided checkpoint:
```bash
python test.py ./trained_model/checkpoints/drugs_ckpt.pt
```


## Evaluation
Use the provided conformers from the provided checkpoint or from your own trained model to evaluate conformers.
```bash
python eval_covmat.py ./trained_model/samples/samples_all.pkl
```
