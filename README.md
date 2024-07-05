# Automated Prompting for Non-overlapping Cross-domain Sequential Recommendation
## How to Install
This code is built on top of the awesome toolbox [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) so you need to install the `dassl` environment first. Simply follow the instructions described [here](https://github.com/KaiyangZhou/Dassl.pytorch#installation) to install `dassl` as well as PyTorch. After that, run `pip install -r requirements.txt` (this should be done when `dassl` is activated). Then, you are ready to go.

## How to Run

python train.py 
--root DATA 
--seed 3 
--trainer CoOp 
--dataset-config-file configs/datasets/oxford_pets.yaml 
--config-file configs/trainers/CoOp/rn50_ep50_ctxv1.yaml 
--output-dir output/oxford_pets/CoOp/rn50_ep50_ctxv1_1shots/nctx16_cscFalse_ctpmiddle/seed3



