# setups and brings up venv.  dg 1/30/24
module load python3/3.8.10
# module load pytorch/1.9.0
source venv/bin/activate
export PYTHONPATH="/projectnb/textconv/distill/mdistiller:$PYTHONPATH"
cd /projectnb/textconv/distill/mdistiller
python3 setup.py develop
pip install -r requirements.txt
#  python3 tools/train.py --cfg configs/cifar100/sld.yaml # this is me!
#  python3 tools/train.py --cfg configs/cifar100/kd.yaml
