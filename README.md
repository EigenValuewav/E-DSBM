# E-DSBM: ENVELOPE-CONDITIONED SCHRÖDINGER BRIDGE MATCHING WITH TEMPORAL DYNAMICS FOR SPEECH ENHANCEMENT
This repository contains the official PyTorch implementation for the paper "Envelope-Conditioned Schrödinger Bridge Matching with Temporal Dynamics for Speech Enhancement" (ICASSP 2026 under review).

E-DSBM is a novel conditional generative model that optimizes the Diffusion Schrödinger Bridge Matching (DSBM) framework for speech enhancement. By conditioning the generative process on the temporal envelope of the noisy speech, E-DSBM preserves the intrinsic characteristics of the signal while restoring fine spectral details. Its efficient, unidirectional, and non-iterative design.

The inferred audio samples can be found at the [following link](https://eigenvaluewav.github.io/E-DSBM_Page/E-DSBM_page.html)



## Model Architecture:
The E-DSBM architecture consists of a Conditional U-Net that takes the noisy spectrogram, a time step, and the temporal envelope as input. It has a multi-output structure, predicting both the drift required for enhancement and the speech envelope itself to reinforce the conditional learning.
<img width="1601" height="774" alt="image" src="https://github.com/user-attachments/assets/1ca3fe42-5dad-4f77-8e04-e7f901976f83" />


### Install Dependencies
Install the required Python(**python version>= 3.9**) packages. For PyTorch, please follow the official instructions for your specific CUDA version
```
pip install -r requirements.txt
```



### How to Train
First, create the **runs**, **checkpoints**, and **eval_results** directories within the E-DSBM root directory.
Before starting the training, you will need to prepare the dataset. This project is designed for the VoiceBank+DEMAND dataset, which was used in our paper. Please download the dataset from the [VoiceBank+DEMAND Dataset](https://datashare.ed.ac.uk/handle/10283/2791) and resample all audio files to 16kHz.
After preparing the dataset, open the config.json file. Revise the paths for **clean_dir, noisy_dir, train_file, and test_file** to match your directory structure.

We train the model via running
``` 
python main.py --mode train
```



### How to Inference on Single File
To enhance a single noisy audio file, use the inference mode.
```
python main.py --mode inference --checkpoint_path ./checkpoints/best_model.pth --test_clean_dir ./PATH/TO/CLEAN/SPEECH/ --test_noisy_dir ./PATH/TO/NOISY/SPEECH/ --inference_steps 7
```




### How to Evaluate
To evaluate a trained model on the test set, provide the path to the checkpoint.
```
python main.py --mode evaluate --checkpoint_path ./checkpoints/best_model.pth --test_clean_dir ./PATH/TO/CLEAN/SPEECH/ --test_noisy_dir ./PATH/TO/NOISY/SPEECH/ --inference_steps 7
```




