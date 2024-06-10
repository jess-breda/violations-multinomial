# Violations Multinomial

### Set Up
* clone repository into desrired folder `git clone https://github.com/jess-breda/violations-multinomial.git`
* if on cluster, activate module with python 3.10 e.g. `module load anacondapy/2024.02`
* create environment `conda create -n viol-multi python==3.10`
* activate environemnt `conda activate viol-multi`
* change direcotry to where `setup.py` is located `cd violations-multinomial/src`
* install the package and libraries `pip install -e .`
* install SSM, directions from [here](https://github.com/lindermanlab/ssm)
  * clone repository into desired folder (doesn't need to be the same as above) `git clone https://github.com/lindermanlab/ssm`
  * change direcotry to where `setup.py` is located `cd ssm`
  * `pip install -e .`
    * note numpy, cython already installed in viol-multi


