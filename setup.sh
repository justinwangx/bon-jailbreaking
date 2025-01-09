ssh-keygen -t ed25519
mamba create -n bon python=3.11.7
sudo apt install ffmpeg
pip install -r requirements.txt
pip install vllm==0.6.2
cd ..
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip install -e ".[model_worker]"
cd ../bon-jailbreaking