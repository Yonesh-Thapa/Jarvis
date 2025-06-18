# full, runnable code here
import pyaudio
import numpy as np
import librosa
import threading
import time
from collections import deque

from src.neural_fabric import NeuralFabric

class AudioCortex:
    """
    Processes a live audio stream into sparse neural activations.
    It controls the 'audio' zone of a NeuralFabric instance.
    """
    def __init__(self, fabric: NeuralFabric, zone_name: str, sample_rate=44100,
                 chunk_size=2048, n_mfcc=13, n_bins_per_mfcc=8):
        """
        Initializes the Audio Cortex.

        Args:
            fabric (NeuralFabric): The shared neural fabric.
            zone_name (str): The name of the zone this cortex controls.
            sample_rate (int): Audio sample rate in Hz.
            chunk_size (int): Number of audio samples per processing buffer.
            n_mfcc (int): Number of MFCCs to compute, representing sound features.
            n_bins_per_mfcc (int): Number of neurons to represent the value range of a single MFCC.
        """
        self.fabric = fabric
        self.zone_name = zone_name
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.n_mfcc = n_mfcc
        self.n_bins_per_mfcc = n_bins_per_mfcc
        self.num_neurons_required = self.n_mfcc * self.n_bins_per_mfcc
        
        # --- Principle: Verify resource allocation ---
        if self.zone_name not in self.fabric.zones:
            raise ValueError(f"Zone '{self.zone_name}' not found. Please add neurons to it first.")
        
        audio_zone_neurons = self.fabric.zones[self.zone_name]
        if len(audio_zone_neurons) < self.num_neurons_required:
            raise ValueError(f"Audio zone needs {self.num_neurons_required} neurons, but only {len(audio_zone_neurons)} are allocated.")

        # --- Map neurons to MFCC bins for discretization ---
        self.neuron_map = np.array(sorted(list(audio_zone_neurons))[:self.num_neurons_required]).reshape((self.n_mfcc, self.n_bins_per_mfcc))
        
        # Define the expected range of MFCC values for normalization.
        # These are empirical values, good for general purpose audio.
        self.mfcc_min = -500
        self.mfcc_max = 500
        
        # Audio stream handling
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.is_streaming = False
        self.audio_buffer = deque()
        self.processing_thread = None

        print(f"AudioCortex initialized. Mapped {self.num_neurons_required} neurons to {self.n_mfcc} MFCCs with {self.n_bins_per_mfcc} bins each.")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Non-blocking callback called by PyAudio when new audio data is available."""
        self.audio_buffer.append(in_data)
        return (in_data, pyaudio.paContinue)

    def _process_audio_thread(self):
        """Dedicated thread to process audio from the buffer to avoid blocking the audio callback."""
        while self.is_streaming:
            if self.audio_buffer:
                data = self.audio_buffer.popleft()
                self.process_chunk(data)
            else:
                time.sleep(0.01) # Wait for more data

    def start_stream(self):
        """Opens the microphone stream and starts the processing thread."""
        if self.is_streaming:
            print("Stream is already running.")
            return

        self.stream = self.p.open(format=pyaudio.paFloat32,
                                  channels=1,
                                  rate=self.sample_rate,
                                  input=True,
                                  frames_per_buffer=self.chunk_size,
                                  stream_callback=self._audio_callback)
        self.is_streaming = True
        self.processing_thread = threading.Thread(target=self._process_audio_thread, daemon=True)
        self.processing_thread.start()
        print("Audio stream started.")

    def stop_stream(self):
        """Stops the audio stream and shuts down."""
        if not self.is_streaming:
            return
        
        self.is_streaming = False
        if self.processing_thread:
            self.processing_thread.join() # Wait for thread to finish
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        self.p.terminate()
        print("Audio stream stopped.")
        
    def _mfccs_to_sparse_activations(self, mfccs: np.ndarray) -> set:
        """
        Maps MFCC vectors to a sparse set of neuron UIDs using binning.
        
        Returns:
            set: A set of neuron UIDs to be activated.
        """
        activated_neuron_uids = set()
        
        # We only process the MFCC of the middle time frame of the chunk
        # to get a representative activation for that moment.
        mfcc_vector = mfccs[:, mfccs.shape[1] // 2]
        
        # Normalize and clip the values to be within [0, 1]
        normalized_mfccs = (mfcc_vector - self.mfcc_min) / (self.mfcc_max - self.mfcc_min)
        clipped_mfccs = np.clip(normalized_mfccs, 0.0, 1.0)
        
        # --- Principle: Map features to neurons ---
        # Determine the bin index for each MFCC value.
        # `(self.n_bins_per_mfcc - 1)` ensures the index is within `[0, n_bins-1]`.
        bin_indices = (clipped_mfccs * (self.n_bins_per_mfcc - 1)).astype(int)

        for i_mfcc, bin_idx in enumerate(bin_indices):
            # Get the neuron UID assigned to this MFCC and this specific bin.
            neuron_uid = self.neuron_map[i_mfcc, bin_idx]
            activated_neuron_uids.add(neuron_uid)
            
        return activated_neuron_uids
        
    def process_chunk(self, audio_data: bytes) -> set:
        """
        The main processing logic. Takes a chunk of audio, computes MFCCs,
        and activates neurons in the fabric. Designed to be testable.
        
        Args:
            audio_data (bytes): A chunk of raw audio data.

        Returns:
            set: The set of neuron UIDs that were activated.
        """
        # Convert byte data to a numpy array of floats.
        audio_np = np.frombuffer(audio_data, dtype=np.float32)
        
        # --- Principle: "Hear once", compress, and discard raw data ---
        if np.abs(audio_np).mean() < 0.005: # Simple silence detection
             return set()

        # 1. Compute MFCCs - the compressed, symbolic representation.
        # The raw `audio_np` is discarded after this step.
        mfccs = librosa.feature.mfcc(y=audio_np, sr=self.sample_rate, n_mfcc=self.n_mfcc, n_fft=self.chunk_size)
        
        # 2. Convert features to sparse neural activations.
        activated_uids = self._mfccs_to_sparse_activations(mfccs)
        
        # 3. Excite the corresponding neurons in the fabric.
        if activated_uids:
            self.fabric.activate_pattern(activated_uids, signal_strength=1.1)

        return activated_uids