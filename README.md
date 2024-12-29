# Speaker Verification System

This repository implements a speaker verification system using audio features and deep learning techniques. The system verifies whether a given audio sample matches a target speaker.

## Features
- **Real-time speaker verification** using microphone input.
- **MFCC feature extraction** along with delta and delta-delta coefficients.
- **LSTM-based model** for speaker verification.
- **Model training** using target and non-target speaker recordings.

## Requirements
- `librosa` for audio processing
- `sounddevice` for recording audio
- `torch` for the neural network

## Usage

1. **Training**: Use target and non-target audio recordings to train the speaker verification model.
2. **Verification**: Record real-time audio to verify if it matches the target speaker.
