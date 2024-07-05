# Automated Prompting for Non-overlapping Cross-domain Sequential Recommendation
## How to Install
You need to install the `dassl` environment first. Simply follow the instructions below to install `dassl` as well as PyTorch. 
### Installation

Make sure [conda](https://www.anaconda.com/distribution/) is installed properly.

```bash
# Clone this repo
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/

# Create a conda environment
conda create -y -n dassl python=3.8

# Activate the environment
conda activate dassl

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
```
After that, run `pip install -r requirements.txt` (this should be done when `dassl` is activated). Then, you are ready to go.

## How to Run

python train.py 
--root DATA 
--seed 3 
--trainer CoOp 
--dataset-config-file configs/datasets/oxford_pets.yaml 
--config-file configs/trainers/CoOp/rn50_ep50_ctxv1.yaml 
--output-dir output/oxford_pets/CoOp/rn50_ep50_ctxv1_1shots/nctx16_cscFalse_ctpmiddle/seed3



