import pyaudio
import numpy as np
import librosa
import threading
import time
from collections import deque
from src.neural_fabric import NeuralFabric

class AudioCortex:
    def __init__(self, fabric: NeuralFabric, zone_name: str, sample_rate=44100,
                 chunk_size=2048, n_mfcc=13, n_bins_per_mfcc=8):
        self.fabric = fabric
        self.zone_name = zone_name
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.n_mfcc = n_mfcc
        self.n_bins_per_mfcc = n_bins_per_mfcc
        self.num_neurons_required = self.n_mfcc * self.n_bins_per_mfcc
        
        if self.zone_name not in self.fabric.zones:
            raise ValueError(f"Zone '{self.zone_name}' not found. Please add neurons to it first.")
        
        audio_zone_neurons = self.fabric.zones[self.zone_name]
        if len(audio_zone_neurons) < self.num_neurons_required:
            raise ValueError(f"Audio zone needs {self.num_neurons_required} neurons, but only {len(audio_zone_neurons)} are allocated.")

        self.neuron_map = np.array(sorted(list(audio_zone_neurons))[:self.num_neurons_required]).reshape((self.n_mfcc, self.n_bins_per_mfcc))
        
        self.mfcc_min = -500
        self.mfcc_max = 500
        
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.is_streaming = False
        self.audio_buffer = deque()
        self.processing_thread = None
        print(f"AudioCortex initialized. Mapped {self.num_neurons_required} neurons.")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        self.audio_buffer.append(in_data)
        return (in_data, pyaudio.paContinue)

    def _process_audio_thread(self):
        while self.is_streaming:
            if self.audio_buffer:
                data = self.audio_buffer.popleft()
                self.process_chunk(data)
            else:
                time.sleep(0.01)

    def start_stream(self):
        if self.is_streaming:
            return
        try:
            self.stream = self.p.open(format=pyaudio.paFloat32, channels=1,
                                      rate=self.sample_rate, input=True,
                                      frames_per_buffer=self.chunk_size,
                                      stream_callback=self._audio_callback)
            self.is_streaming = True
            self.processing_thread = threading.Thread(target=self._process_audio_thread, daemon=True)
            self.processing_thread.start()
            print("Audio stream started.")
        except Exception as e:
            print(f"AUDIO_ERROR: Could not start audio stream: {e}. Is a microphone connected?")
            self.is_streaming = False
            self.stream = None

    def stop_stream(self):
        if not self.is_streaming: return
        self.is_streaming = False
        if self.processing_thread and self.processing_thread.is_alive(): self.processing_thread.join()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
        print("Audio stream stopped.")
        
    def _mfccs_to_sparse_activations(self, mfccs: np.ndarray) -> set:
        activated_neuron_uids = set()
        mfcc_vector = mfccs[:, mfccs.shape[1] // 2]
        normalized_mfccs = (mfcc_vector - self.mfcc_min) / (self.mfcc_max - self.mfcc_min)
        clipped_mfccs = np.clip(normalized_mfccs, 0.0, 1.0)
        bin_indices = (clipped_mfccs * (self.n_bins_per_mfcc - 1)).astype(int)

        for i_mfcc, bin_idx in enumerate(bin_indices):
            neuron_uid = self.neuron_map[i_mfcc, bin_idx]
            activated_neuron_uids.add(neuron_uid)
            
        return activated_neuron_uids
        
    def process_chunk(self, audio_data: bytes) -> set:
        audio_np = np.frombuffer(audio_data, dtype=np.float32)
        if np.abs(audio_np).mean() < 0.005: return set()
        mfccs = librosa.feature.mfcc(y=audio_np, sr=self.sample_rate, n_mfcc=self.n_mfcc, n_fft=self.chunk_size)
        activated_uids = self._mfccs_to_sparse_activations(mfccs)
        if activated_uids:
            self.fabric.activate_pattern(activated_uids, signal_strength=1.1)
        return activated_uids