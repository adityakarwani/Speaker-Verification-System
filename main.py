import numpy as np
import librosa
import sounddevice as sd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader

class AudioProcessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate 

    def extract_features(self, audio):
        """
        Extract MFCC features and their deltas (change over time) and delta-delta (second-order change).
        The features are concatenated and returned as a single array.
        """
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=20)
        delta = librosa.feature.delta(mfccs)
        delta2 = librosa.feature.delta(mfccs, order=2)
        features = np.concatenate([mfccs, delta, delta2]) 
        return features

    def record_audio(self, duration=5):
        """
        Record audio for the specified duration.
        Returns the recorded audio as a 1D numpy array.
        """
        recording = sd.rec(int(duration * self.sample_rate), 
                           samplerate=self.sample_rate, channels=1)
        sd.wait()
        return recording.flatten()  

class SpeakerDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)  
        self.labels = torch.LongTensor(labels)  

    def __len__(self):
        return len(self.labels)  

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]  

class SpeakerVerificationModel(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)  
        self.fc = nn.Linear(hidden_size, 2)  

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  
        output = self.fc(lstm_out[:, -1, :])  
        return output

class SpeakerVerifier:
    def __init__(self, sample_rate=16000):
        self.processor = AudioProcessor(sample_rate)  
        self.model = None  

    def train(self, target_recordings, non_target_recordings, epochs=50):
        """
        Train the speaker verification model using the provided target and non-target recordings.
        """
        features = []
        labels = []

        for audio in target_recordings:
            feat = self.processor.extract_features(audio)
            features.append(feat.T)  
            labels.append(1)  

        for audio in non_target_recordings:
            feat = self.processor.extract_features(audio)
            features.append(feat.T)
            labels.append(0)  

        dataset = SpeakerDataset(features, labels)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        input_size = features[0].shape[1]
        self.model = SpeakerVerificationModel(input_size)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters())

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_features, batch_labels in dataloader:
                optimizer.zero_grad()  
                outputs = self.model(batch_features) 
                loss = criterion(outputs, batch_labels)  
                loss.backward()  
                optimizer.step()  
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}')

    def verify_speaker(self, audio, threshold=0.5):
        """
        Verify if the given audio belongs to the target speaker.
        Returns a boolean (True if target, False otherwise) and the confidence score.
        """
        self.model.eval()  
        with torch.no_grad():
            features = self.processor.extract_features(audio)
            features = torch.FloatTensor(features.T).unsqueeze(0)  
            output = torch.softmax(self.model(features), dim=1) 
            probability = output[0][1].item()  
            return probability > threshold, probability  

    def real_time_verification(self, duration=5):
        """
        Perform real-time speaker verification in a loop, allowing the user to quit anytime.
        """
        print("Press 'q' and hit Enter to quit real-time verification.")
        while True:
            user_input = input("Press Enter to record, or 'q' to quit: ").strip().lower()
            if user_input == 'q':
                print("Exiting real-time verification.")
                break

            print("Recording...")
            audio = self.processor.record_audio(duration)  
            is_target, confidence = self.verify_speaker(audio)  
            result = "Target" if is_target else "Non-target"  
            print(f"Speaker: {result} (confidence: {confidence:.2f})")

def evaluate_system(verifier, test_target, test_non_target):
    """
    Evaluate the system on test data and return accuracy and F1 score.
    """
    true_labels = []
    pred_labels = []

    for audio in test_target:
        is_target, _ = verifier.verify_speaker(audio)
        true_labels.append(1)
        pred_labels.append(1 if is_target else 0)

    for audio in test_non_target:
        is_target, _ = verifier.verify_speaker(audio)
        true_labels.append(0)
        pred_labels.append(1 if is_target else 0)

    return {
        'accuracy': accuracy_score(true_labels, pred_labels),  
        'f1': f1_score(true_labels, pred_labels)  
    }

if __name__ == "__main__":
    verifier = SpeakerVerifier()

    target_recordings = [np.random.randn(16000*5) for _ in range(10)]
    non_target_recordings = [np.random.randn(16000*5) for _ in range(10)]  

    verifier.train(target_recordings, non_target_recordings)

    verifier.real_time_verification()

