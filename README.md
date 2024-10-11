# llm-jailbreaks
Contains code useful for jailbreaking LLMs.

# Repository setup

### Environment

To set up the development environment for this project,
follow the steps below:

1. First install `micromamba` if you haven't already.
You can follow the installation instructions here:
https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html. The Homebrew / automatic installation method is recommended.
Or just run:
```
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```
2. Next, you need to run the following from the root of the repository:
    ```bash
    micromamba env create -n almj python=3.11.7
    micromamba activate almj
    pip install -r requirements.txt
    pip install -e .
    (git clone git@github.com:facebookresearch/WavAugment.git && cd WavAugment && python setup.py develop)
    ```

3. Install Kaldi if on a new machine: ./scripts/install_kaldi.sh

4. Pull submodules (optional but required for DiVA)
    ```bash
    git submodule update --init --recursive
    ```

### Secrets

You should create a file called `SECRETS` at the root of the repository
with the following contents:
```
# Required
OPENAI_API_KEY=<your-key>
OPENAI_ORG=<openai-org-id>
GOOGLE_API_KEY=<your-key>
GOOGLE_PROJECT_ID=<GCP-project-name>
GOOGLE_PROJECT_REGION=<GCP-project-region>

# Optional
RUNPOD_API_KEY=<your-key> # required for GPU access for DiVA
ELEVENLABS_API_KEY=<your-key> # required for ElevenLabs TTS
ANTHROPIC_API_KEY=<your-key>
```

### Replicate experiments

To replicate the experiments in the paper, run the scripts in the `paper_runs` directory. For example, to replicate Figure 1, run the following:

```bash
bash paper_runs/1_run_bon.sh
```