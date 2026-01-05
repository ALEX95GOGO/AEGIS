# AEGIS: Human Attention-based Explainable Guidance for Intelligent Vehicle Systems

**[ACM SIGCHI 2025]**  
Official repository for the paper: *AEGIS: Human Attention-based Explainable Guidance for Intelligent Vehicle Systems*, presented at ACM SIGCHI 2025.

---

## ⚙️ Create the Environment

Make sure you have [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

Then, run the following command in your terminal:

```bash
conda env create -f environment.yml
```
or 
```bash
conda create -n aegis python==3.7
conda activate aegis
pip install -r requirements.txt
```
## What do AEGIS provide? Please click to view the video &#x2193;&#x2193;&#x2193;

[![Watch the video](asset/thumbnail.png)](https://www.youtube.com/watch?v=RiyZsicPuQ0)


## 🚗 Running the CARLA Simulator

```bash
# Create and navigate to the CARLA directory
mkdir carla
cd carla

# Download and extract CARLA 0.9.14
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.14.tar.gz
tar -xvf CARLA_0.9.14.tar.gz

# Launch the simulator
./CarlaUE4.sh --world-port=2000
```

## 🧠 Training AEGIS for the Car-Following Scenario
```
python train_car_following.py --simulator_port 2000
```

Training AEGIS for the Left-Turn Scenario
```
python train_left_turn.py --simulator_port 2000
```

## Evaluate AEGIS for the Car-Following Scenario
```
python eval_car_following_save.py
```
You should be able to visualize the machine attention using this evalutation script.

## Dataset

The eye-tracking data and corresponding frame images can be downloaded from the following link:

https://huggingface.co/datasets/zzhuan/AEGIS_dataset/tree/main

## 📄 Paper

- __Zhuang Z__, Lu CY, Wang YK, Chang YC, Thomas Do, Lin CT. *"AEGIS: Human Attention-based Explainable Guidance for Intelligent Vehicle Systems"*. **ACM CHI Conference on Human Factors in Computing Systems**, 2025.

You can read our paper on arXiv here:  
[**AEGIS: Human Attention-based Explainable Guidance for Intelligent Vehicle Systems** (arXiv:2504.05950)](https://arxiv.org/abs/2504.05950)

[![Download PDF](https://img.shields.io/badge/PDF-Download-blue)](https://arxiv.org/pdf/2504.05950)

