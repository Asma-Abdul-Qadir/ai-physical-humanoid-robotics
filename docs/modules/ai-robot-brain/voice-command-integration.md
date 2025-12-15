---
sidebar_position: 5
---

# Voice Command Integration: Audio Processing for Humanoid Robots

Welcome to the Voice Command Integration module, which focuses on implementing robust speech recognition and natural language processing systems for humanoid robots. This chapter covers the integration of audio processing, speech recognition, natural language understanding, and voice synthesis to enable natural human-robot interaction through voice commands.

## Learning Objectives

By the end of this section, you will be able to:
- Implement speech recognition systems using modern ASR models
- Process audio input for voice command recognition
- Integrate natural language processing for command understanding
- Create voice command grammars and parsing systems
- Implement text-to-speech for robot responses
- Handle noise, accents, and environmental challenges
- Design voice user interfaces for humanoid robots
- Evaluate and optimize voice command systems

## Introduction to Voice Command Systems

Voice command systems enable natural and intuitive interaction between humans and humanoid robots. These systems process spoken language to understand user intentions and execute appropriate actions. For humanoid robots, voice command integration is essential for creating social and accessible interactions.

### Voice Command Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Audio Input   │    │   Speech        │    │   Language      │
│   (Microphones) │───▶│   Recognition   │───▶│   Understanding │
│   • Microphone  │    │   • ASR Model   │    │   • NLU Engine  │
│   • Array       │    │   • VAD         │    │   • Intent      │
│   • Beamforming │    │   • STT         │    │   • Entities    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Audio         │    │   Command       │    │   Action        │
│   Preprocessing │    │   Parsing       │    │   Execution     │
│   • Noise       │    │   • Grammar     │    │   • Navigation  │
│   • Enhancement │    │   • Validation  │    │   • Manipulation│
│   • Filtering   │    │   • Context     │    │   • Interaction │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Voice Command Processing Pipeline

The voice command processing pipeline involves multiple stages working together to transform speech into actionable commands:

1. **Audio Capture**: Capturing speech from the environment
2. **Audio Preprocessing**: Enhancing and cleaning the audio signal
3. **Speech Recognition**: Converting speech to text
4. **Natural Language Understanding**: Extracting meaning and intent
5. **Command Parsing**: Converting natural language to structured commands
6. **Action Execution**: Executing the appropriate robot behavior
7. **Response Generation**: Providing feedback to the user

## Audio Processing and Preprocessing

### Microphone Arrays and Audio Capture

For humanoid robots, audio quality is crucial for reliable voice command recognition. Multiple microphones can be used to improve signal quality:

```python
#!/usr/bin/env python3

import pyaudio
import numpy as np
import webrtcvad
import threading
import queue
import time
from scipy import signal
from collections import deque


class AudioCaptureSystem:
    def __init__(self):
        # Audio parameters
        self.sample_rate = 16000  # Standard for ASR
        self.channels = 4         # Multiple microphones for beamforming
        self.chunk_size = 1024
        self.format = pyaudio.paInt16

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()

        # Initialize WebRTC VAD (Voice Activity Detection)
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(2)  # More aggressive VAD mode

        # Audio buffers
        self.audio_buffer = deque(maxlen=100)  # Store recent audio chunks
        self.processed_audio_queue = queue.Queue()

        # Audio processing parameters
        self.silence_threshold = 0.01
        self.min_speech_duration = 0.3  # seconds
        self.max_buffer_duration = 5.0  # seconds

        # State variables
        self.is_listening = False
        self.is_recording = False
        self.recording_start_time = 0
        self.recording_buffer = []

        # Audio processing components
        self.noise_reducer = NoiseReducer(self.sample_rate)
        self.beamformer = Beamformer(self.channels, self.sample_rate)

        self.get_logger().info('Audio capture system initialized')

    def start_capture(self):
        """Start audio capture from microphone array"""
        # Open audio stream with multiple channels
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

        # Start capture thread
        self.capture_thread = threading.Thread(target=self.capture_audio)
        self.capture_thread.daemon = True
        self.capture_thread.start()

        self.get_logger().info('Audio capture started')

    def capture_audio(self):
        """Capture audio from microphone array"""
        while True:
            try:
                # Read audio data from all channels
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)

                # Convert to numpy array and separate channels
                audio_array = np.frombuffer(data, dtype=np.int16)
                audio_array = audio_array.astype(np.float32).reshape(-1, self.channels)

                # Normalize audio
                audio_array /= 32768.0

                # Apply beamforming to focus on speaker direction
                focused_audio = self.beamformer.apply_beamforming(audio_array)

                # Apply noise reduction
                enhanced_audio = self.noise_reducer.reduce_noise(focused_audio)

                # Add to buffer
                self.audio_buffer.append(enhanced_audio)

                # Check for voice activity
                if self.is_voice_activity(enhanced_audio):
                    if not self.is_recording:
                        self.start_recording(enhanced_audio)
                    else:
                        self.add_to_recording(enhanced_audio)
                else:
                    if self.is_recording:
                        self.check_stop_recording()

            except Exception as e:
                self.get_logger().error(f'Audio capture error: {e}')
                break

    def is_voice_activity(self, audio_chunk):
        """Check for voice activity using WebRTC VAD"""
        # Convert to 16-bit PCM for VAD (use first channel)
        chunk_16bit = (audio_chunk[:, 0] * 32768).astype(np.int16)

        # VAD requires specific frame sizes (10, 20, or 30 ms)
        frame_size = int(self.sample_rate * 0.01)  # 10ms frame
        if len(chunk_16bit) >= frame_size:
            frame = chunk_16bit[:frame_size].tobytes()
            return self.vad.is_speech(frame, self.sample_rate)

        return False

    def start_recording(self, audio_chunk):
        """Start recording speech for command processing"""
        self.is_recording = True
        self.recording_start_time = time.time()
        self.recording_buffer = [audio_chunk]

    def add_to_recording(self, audio_chunk):
        """Add audio chunk to current recording"""
        if self.is_recording:
            self.recording_buffer.append(audio_chunk)

    def check_stop_recording(self):
        """Check if we should stop recording and process the command"""
        if time.time() - self.recording_start_time >= self.min_speech_duration:
            # Process the recorded speech
            if self.recording_buffer:
                full_recording = np.concatenate(self.recording_buffer, axis=0)

                # Add to processing queue
                self.processed_audio_queue.put(full_recording)

                # Log the recording
                duration = len(full_recording) / self.sample_rate
                self.get_logger().info(f'Recording complete: {duration:.2f}s')

            self.is_recording = False

    def get_processed_audio(self, timeout=1.0):
        """Get processed audio from the queue"""
        try:
            return self.processed_audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_logger(self):
        """Simple logger for simulation"""
        class Logger:
            def info(self, msg):
                print(f'[INFO] {msg}')
            def error(self, msg):
                print(f'[ERROR] {msg}')
        return Logger()

    def stop_capture(self):
        """Stop audio capture"""
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()


class NoiseReducer:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.noise_buffer = deque(maxlen=20000)  # 20 seconds of noise samples
        self.noise_estimate = None
        self.spectral_subtraction_factor = 0.5

    def reduce_noise(self, audio_data):
        """Apply noise reduction to audio data"""
        # Update noise estimate if this appears to be silence
        if self.is_silence(audio_data):
            # Flatten multi-channel audio for noise estimation
            flat_audio = audio_data.flatten()
            self.noise_buffer.extend(flat_audio)

            if len(self.noise_buffer) == self.noise_buffer.maxlen:
                self.noise_estimate = np.array(self.noise_buffer)

        # Apply noise reduction if we have an estimate
        if self.noise_estimate is not None and len(self.noise_estimate) > 0:
            # Apply spectral subtraction
            enhanced_audio = self.spectral_subtraction(audio_data)
            return enhanced_audio

        return audio_data

    def is_silence(self, audio_data):
        """Check if audio segment is likely silence"""
        # Calculate RMS across all channels
        rms = np.sqrt(np.mean(audio_data ** 2))
        return rms < 0.005  # Adjust threshold as needed

    def spectral_subtraction(self, audio_data):
        """Apply spectral subtraction noise reduction"""
        # Flatten for processing, then reshape back
        original_shape = audio_data.shape
        flat_audio = audio_data.flatten()

        # Apply FFT
        fft = np.fft.fft(flat_audio)
        magnitude = np.abs(fft)
        phase = np.angle(fft)

        # Get noise estimate magnitude
        if len(self.noise_estimate) > 0:
            # Pad or truncate noise estimate to match audio length
            if len(self.noise_estimate) > len(flat_audio):
                noise_estimate = self.noise_estimate[:len(flat_audio)]
            elif len(self.noise_estimate) < len(flat_audio):
                # Repeat noise estimate to match length
                reps = int(np.ceil(len(flat_audio) / len(self.noise_estimate)))
                noise_estimate = np.tile(self.noise_estimate, reps)[:len(flat_audio)]
            else:
                noise_estimate = self.noise_estimate

            noise_fft = np.fft.fft(noise_estimate)
            noise_magnitude = np.abs(noise_fft)

            # Apply spectral subtraction
            enhanced_magnitude = np.maximum(
                magnitude - self.spectral_subtraction_factor * noise_magnitude,
                0.1 * magnitude
            )

            # Reconstruct signal
            enhanced_fft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = np.real(np.fft.ifft(enhanced_fft))

            # Reshape back to original shape
            enhanced_audio = enhanced_audio.reshape(original_shape)
            return enhanced_audio.astype(np.float32)

        return audio_data


class Beamformer:
    def __init__(self, num_channels, sample_rate):
        self.num_channels = num_channels
        self.sample_rate = sample_rate

        # Simple delay-and-sum beamformer parameters
        # Assuming microphones are arranged in a linear array
        self.mic_distance = 0.05  # 5cm between microphones
        self.speed_of_sound = 343.0  # m/s

        # Default look direction (0 degrees = forward)
        self.look_direction = 0.0

    def apply_beamforming(self, multi_channel_audio):
        """Apply beamforming to focus on specific direction"""
        if self.num_channels <= 1:
            # No beamforming needed for single channel
            return multi_channel_audio.flatten() if multi_channel_audio.ndim > 1 else multi_channel_audio

        # For simplicity, use the first channel as the main signal
        # In practice, implement proper delay-and-sum or other beamforming
        return multi_channel_audio[:, 0]  # Use first channel for now

    def set_look_direction(self, angle_degrees):
        """Set the beamforming look direction"""
        self.look_direction = np.radians(angle_degrees)
```

### Audio Enhancement and Filtering

For humanoid robots operating in noisy environments, audio enhancement is critical:

```python
#!/usr/bin/env python3

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, resample


class AudioEnhancer:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

        # Pre-emphasis filter parameters
        self.pre_emphasis_coeff = 0.97

        # Initialize AGC parameters
        self.agc_target_level = 0.5
        self.agc_attack_time = 0.01  # seconds
        self.agc_release_time = 0.2  # seconds

        # Initialize noise reduction parameters
        self.noise_floor = 0.01
        self.speech_threshold = 0.05

    def enhance_audio(self, audio_data):
        """Apply comprehensive audio enhancement"""
        # Apply pre-emphasis to boost high frequencies
        emphasized_audio = self.pre_emphasis_filter(audio_data)

        # Apply AGC (Automatic Gain Control)
        agc_audio = self.apply_agc(emphasized_audio)

        # Apply noise reduction
        denoised_audio = self.reduce_noise(agc_audio)

        # Apply high-pass filter to remove DC offset
        filtered_audio = self.high_pass_filter(denoised_audio)

        return filtered_audio

    def pre_emphasis_filter(self, audio_data, coeff=None):
        """Apply pre-emphasis filter to boost high frequencies"""
        if coeff is None:
            coeff = self.pre_emphasis_coeff

        # Pre-emphasis: y[n] = x[n] - coeff * x[n-1]
        return np.append(audio_data[0], audio_data[1:] - coeff * audio_data[:-1])

    def apply_agc(self, audio_data):
        """Apply Automatic Gain Control"""
        # Simple AGC implementation
        window_size = int(0.01 * self.sample_rate)  # 10ms window

        if len(audio_data) < window_size:
            return audio_data

        # Calculate RMS in overlapping windows
        rms_values = []
        for i in range(0, len(audio_data) - window_size, window_size // 2):
            window = audio_data[i:i + window_size]
            rms = np.sqrt(np.mean(window ** 2))
            rms_values.append(max(rms, 0.001))  # Avoid division by zero

        # Interpolate RMS values to match original length
        if rms_values:
            target_rms = np.interp(
                np.linspace(0, len(rms_values) - 1, len(audio_data)),
                np.arange(len(rms_values)),
                rms_values
            )

            # Apply gain to achieve target level
            gain = self.agc_target_level / (target_rms + 0.001)
            # Limit gain to prevent excessive amplification
            gain = np.clip(gain, 0.1, 10.0)

            return audio_data * gain

        return audio_data

    def reduce_noise(self, audio_data):
        """Apply noise reduction using spectral subtraction"""
        # Calculate noise floor estimate from low-energy segments
        frame_size = 512
        overlap = 0.5

        if len(audio_data) < frame_size:
            return audio_data

        # Segment the audio
        frames = []
        for i in range(0, len(audio_data) - frame_size, int(frame_size * (1 - overlap))):
            frames.append(audio_data[i:i + frame_size])

        # Estimate noise from low-energy frames
        frame_energies = [np.mean(frame ** 2) for frame in frames]
        noise_frames = [frames[i] for i, energy in enumerate(frame_energies)
                       if energy < self.noise_floor]

        if noise_frames:
            avg_noise_frame = np.mean(noise_frames, axis=0)

            # Apply spectral subtraction
            enhanced_frames = []
            for frame in frames:
                frame_fft = np.fft.fft(frame)
                noise_fft = np.fft.fft(avg_noise_frame)

                frame_mag = np.abs(frame_fft)
                noise_mag = np.abs(noise_fft)

                enhanced_mag = np.maximum(frame_mag - 0.5 * noise_mag, 0.1 * frame_mag)
                enhanced_frame = np.real(np.fft.ifft(enhanced_mag * np.exp(1j * np.angle(frame_fft))))
                enhanced_frames.append(enhanced_frame)

            # Reconstruct audio (simplified overlap-add)
            result = np.zeros_like(audio_data)
            frame_idx = 0
            for i in range(0, len(audio_data) - frame_size, int(frame_size * (1 - overlap))):
                if frame_idx < len(enhanced_frames):
                    result[i:i + frame_size] += enhanced_frames[frame_idx]
                    frame_idx += 1

            return result

        return audio_data

    def high_pass_filter(self, audio_data, cutoff_freq=100):
        """Apply high-pass filter to remove DC offset and low-frequency noise"""
        nyquist = self.sample_rate / 2.0
        normalized_cutoff = cutoff_freq / nyquist

        # Create Butterworth high-pass filter
        b, a = butter(4, normalized_cutoff, btype='high', analog=False)

        # Apply filter
        filtered_audio = filtfilt(b, a, audio_data)

        return filtered_audio

    def band_pass_filter(self, audio_data, low_freq=300, high_freq=3400):
        """Apply band-pass filter to focus on speech frequencies"""
        nyquist = self.sample_rate / 2.0
        low = low_freq / nyquist
        high = high_freq / nyquist

        # Create Butterworth band-pass filter
        b, a = butter(4, [low, high], btype='band', analog=False)

        # Apply filter
        filtered_audio = filtfilt(b, a, audio_data)

        return filtered_audio
```

## Speech Recognition Systems

### Implementing ASR with Modern Models

For humanoid robots, accurate speech recognition is essential. We'll implement both local and cloud-based ASR systems:

```python
#!/usr/bin/env python3

import torch
import whisper
import speech_recognition as sr
import numpy as np
import threading
import queue
import time
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class SpeechRecognitionResult:
    """Data class for speech recognition results"""
    text: str
    confidence: float
    language: str
    timestamp: float
    processing_time: float


class LocalASRSystem:
    def __init__(self, model_size="base"):
        self.model_size = model_size
        self.model = None
        self.recognizer = sr.Recognizer()

        # Initialize the Whisper model
        self.load_model()

        # Audio processing parameters
        self.sample_rate = 16000
        self.energy_threshold = 300  # For speech detection
        self.dynamic_energy_threshold = True

        # Processing queue for async recognition
        self.recognition_queue = queue.Queue()
        self.result_queue = queue.Queue()

        self.get_logger().info(f'Local ASR system initialized with {model_size} model')

    def load_model(self):
        """Load the Whisper ASR model"""
        try:
            self.get_logger().info(f'Loading Whisper {self.model_size} model...')
            self.model = whisper.load_model(self.model_size)
            self.get_logger().info('Whisper model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load Whisper model: {e}')
            self.model = None

    def transcribe_audio(self, audio_data):
        """Transcribe audio data using Whisper"""
        if self.model is None:
            return SpeechRecognitionResult(
                text="",
                confidence=0.0,
                language="",
                timestamp=time.time(),
                processing_time=0.0
            )

        start_time = time.time()

        try:
            # Ensure audio is at correct sample rate (16kHz for Whisper)
            # In practice, resample if needed
            audio_np = audio_data.astype(np.float32)

            # Pad audio if too short (Whisper needs at least 1 second)
            if len(audio_np) < self.sample_rate:
                pad_length = self.sample_rate - len(audio_np)
                audio_np = np.pad(audio_np, (0, pad_length), mode='constant')

            # Run transcription
            result = self.model.transcribe(audio_np)

            processing_time = time.time() - start_time

            return SpeechRecognitionResult(
                text=result.get('text', ''),
                confidence=result.get('avg_logprob', 0.0),
                language=result.get('language', 'unknown'),
                timestamp=time.time(),
                processing_time=processing_time
            )

        except Exception as e:
            self.get_logger().error(f'Transcription error: {e}')
            processing_time = time.time() - start_time
            return SpeechRecognitionResult(
                text="",
                confidence=0.0,
                language="",
                timestamp=time.time(),
                processing_time=processing_time
            )

    def transcribe_from_file(self, audio_file_path):
        """Transcribe from an audio file"""
        if self.model is None:
            return None

        try:
            result = self.model.transcribe(audio_file_path)
            return SpeechRecognitionResult(
                text=result.get('text', ''),
                confidence=result.get('avg_logprob', 0.0),
                language=result.get('language', 'unknown'),
                timestamp=time.time(),
                processing_time=result.get('processing_time', 0.0)
            )
        except Exception as e:
            self.get_logger().error(f'File transcription error: {e}')
            return None

    def start_async_recognition(self):
        """Start asynchronous recognition thread"""
        recognition_thread = threading.Thread(target=self.async_recognition_worker)
        recognition_thread.daemon = True
        recognition_thread.start()

    def async_recognition_worker(self):
        """Worker thread for asynchronous recognition"""
        while True:
            try:
                # Get audio data from queue
                audio_data = self.recognition_queue.get(timeout=1.0)

                # Transcribe the audio
                result = self.transcribe_audio(audio_data)

                # Put result in result queue
                self.result_queue.put(result)

                self.recognition_queue.task_done()

            except queue.Empty:
                continue  # Timeout, continue loop
            except Exception as e:
                self.get_logger().error(f'Async recognition error: {e}')

    def get_recognition_result(self, timeout=1.0):
        """Get recognition result from queue"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_logger(self):
        """Simple logger for simulation"""
        class Logger:
            def info(self, msg):
                print(f'[INFO] {msg}')
            def error(self, msg):
                print(f'[ERROR] {msg}')
        return Logger()


class CloudASRSystem:
    def __init__(self, api_key=None, provider='google'):
        self.api_key = api_key
        self.provider = provider
        self.recognizer = sr.Recognizer()

        # Configure energy threshold for speech detection
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True

        # Audio processing parameters
        self.sample_rate = 16000

        self.get_logger().info(f'Cloud ASR system initialized for {provider}')

    def transcribe_audio(self, audio_data):
        """Transcribe audio using cloud ASR service"""
        try:
            # Convert numpy array to audio data format expected by speech_recognition
            # This is a simplified approach - in practice, you'd need proper audio format conversion
            import io
            import wave

            # Create a temporary WAV file from numpy array
            audio_io = io.BytesIO()
            with wave.open(audio_io, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)

                # Convert float32 to int16
                int16_audio = (audio_data * 32767).astype(np.int16)
                wav_file.writeframes(int16_audio.tobytes())

            # Reset buffer position
            audio_io.seek(0)

            # Create AudioData object
            audio_data_obj = sr.AudioData(audio_io.read(), self.sample_rate, 2)

            # Use Google Speech Recognition (as example)
            text = self.recognizer.recognize_google(audio_data_obj)

            return SpeechRecognitionResult(
                text=text,
                confidence=0.8,  # Google's API doesn't provide confidence directly
                language="en-US",
                timestamp=time.time(),
                processing_time=0.5  # Estimate
            )

        except sr.UnknownValueError:
            self.get_logger().info('Cloud ASR - Could not understand audio')
            return SpeechRecognitionResult(
                text="",
                confidence=0.0,
                language="",
                timestamp=time.time(),
                processing_time=0.0
            )
        except sr.RequestError as e:
            self.get_logger().error(f'Cloud ASR request error: {e}')
            return SpeechRecognitionResult(
                text="",
                confidence=0.0,
                language="",
                timestamp=time.time(),
                processing_time=0.0
            )
        except Exception as e:
            self.get_logger().error(f'Cloud ASR error: {e}')
            return SpeechRecognitionResult(
                text="",
                confidence=0.0,
                language="",
                timestamp=time.time(),
                processing_time=0.0
            )

    def get_logger(self):
        """Simple logger for simulation"""
        class Logger:
            def info(self, msg):
                print(f'[INFO] {msg}')
            def error(self, msg):
                print(f'[ERROR] {msg}')
        return Logger()


class HybridASRSystem:
    def __init__(self, local_model_size="base", cloud_provider="google"):
        self.local_asr = LocalASRSystem(local_model_size)
        self.cloud_asr = CloudASRSystem(provider=cloud_provider)

        # Configuration
        self.use_local_first = True  # Try local first, fallback to cloud
        self.local_confidence_threshold = 0.7  # Minimum confidence for local results
        self.fallback_enabled = True  # Whether to fallback to cloud

        self.get_logger().info('Hybrid ASR system initialized')

    def transcribe_audio(self, audio_data):
        """Transcribe audio using hybrid approach"""
        if self.use_local_first:
            # Try local ASR first
            local_result = self.local_asr.transcribe_audio(audio_data)

            # If confidence is high enough, return local result
            if local_result.confidence >= self.local_confidence_threshold:
                return local_result

            # Otherwise, fallback to cloud if enabled
            if self.fallback_enabled:
                cloud_result = self.cloud_asr.transcribe_audio(audio_data)

                # Return the result with higher confidence
                if cloud_result.confidence > local_result.confidence:
                    return cloud_result
                else:
                    return local_result
            else:
                return local_result
        else:
            # Try cloud first, fallback to local
            cloud_result = self.cloud_asr.transcribe_audio(audio_data)

            if cloud_result.confidence >= self.local_confidence_threshold:
                return cloud_result

            if self.fallback_enabled:
                local_result = self.local_asr.transcribe_audio(audio_data)
                if local_result.confidence > cloud_result.confidence:
                    return local_result
                else:
                    return cloud_result
            else:
                return cloud_result

    def get_logger(self):
        """Simple logger for simulation"""
        class Logger:
            def info(self, msg):
                print(f'[INFO] {msg}')
            def error(self, msg):
                print(f'[ERROR] {msg}')
        return Logger()
```

## Natural Language Understanding

### Command Parsing and Intent Recognition

For humanoid robots, understanding the user's intent from spoken commands is crucial:

```python
#!/usr/bin/env python3

import re
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class IntentType(Enum):
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    INFORMATION = "information"
    SOCIAL = "social"
    SYSTEM = "system"


@dataclass
class ParsedCommand:
    intent: IntentType
    entities: Dict[str, str]
    confidence: float
    original_text: str
    action: str


class CommandParser:
    def __init__(self):
        # Define command patterns and their corresponding intents
        self.command_patterns = {
            # Navigation commands
            IntentType.NAVIGATION: [
                (r"go to (the )?(?P<destination>\w+)", "go_to_destination"),
                (r"move to (the )?(?P<destination>\w+)", "move_to_destination"),
                (r"navigate to (the )?(?P<destination>\w+)", "navigate_to_destination"),
                (r"walk to (the )?(?P<destination>\w+)", "walk_to_destination"),
                (r"go (forward|backward|left|right)", "move_direction"),
                (r"turn (left|right)", "turn_direction"),
                (r"come (here|to me|over)", "come_to_user"),
                (r"follow me", "follow_user"),
            ],

            # Manipulation commands
            IntentType.MANIPULATION: [
                (r"get (me )?(the )?(?P<object>\w+)", "fetch_object"),
                (r"bring (me )?(the )?(?P<object>\w+)", "fetch_object"),
                (r"pick up (the )?(?P<object>\w+)", "fetch_object"),
                (r"grab (the )?(?P<object>\w+)", "fetch_object"),
                (r"take (the )?(?P<object>\w+)", "fetch_object"),
                (r"hand me (the )?(?P<object>\w+)", "hand_object"),
                (r"give me (the )?(?P<object>\w+)", "hand_object"),
            ],

            # Information commands
            IntentType.INFORMATION: [
                (r"what (is|are) (there|in here)", "describe_environment"),
                (r"tell me about (the )?surroundings", "describe_environment"),
                (r"how are you", "check_wellbeing"),
                (r"what time is it", "get_time"),
                (r"what day is it", "get_date"),
                (r"what is the weather", "get_weather"),
                (r"how many people", "count_people"),
            ],

            # Social commands
            IntentType.SOCIAL: [
                (r"hello", "greet"),
                (r"hi", "greet"),
                (r"good morning", "greet"),
                (r"good afternoon", "greet"),
                (r"good evening", "greet"),
                (r"goodbye", "farewell"),
                (r"bye", "farewell"),
                (r"see you", "farewell"),
                (r"thank you", "acknowledge_gratitude"),
                (r"thanks", "acknowledge_gratitude"),
            ],

            # System commands
            IntentType.SYSTEM: [
                (r"stop", "stop_action"),
                (r"halt", "stop_action"),
                (r"pause", "pause_action"),
                (r"resume", "resume_action"),
                (r"help", "show_help"),
                (r"repeat", "repeat_last_action"),
            ]
        }

        # Define entity extraction patterns
        self.entity_patterns = {
            'destination': [
                'kitchen', 'living room', 'bedroom', 'office', 'bathroom',
                'dining room', 'hallway', 'entrance', 'exit'
            ],
            'object': [
                'bottle', 'cup', 'book', 'phone', 'keys', 'ball',
                'water', 'coffee', 'tea', 'food', 'medicine'
            ],
            'direction': ['forward', 'backward', 'left', 'right', 'up', 'down'],
            'person': ['person', 'people', 'human', 'man', 'woman', 'child']
        }

        # Synonym mappings
        self.synonyms = {
            'bottle': ['water', 'drink', 'liquid', 'container'],
            'cup': ['mug', 'glass', 'coffee', 'tea', 'drink'],
            'kitchen': ['cooking', 'food', 'eat'],
            'living room': ['sofa', 'couch', 'tv', 'relax'],
            'bedroom': ['sleep', 'bed', 'rest'],
            'office': ['work', 'computer', 'desk']
        }

    def parse_command(self, text: str) -> Optional[ParsedCommand]:
        """Parse natural language command and extract intent and entities"""
        text_lower = text.lower().strip()

        # Try to match against each intent type
        for intent_type, patterns in self.command_patterns.items():
            for pattern, action in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    # Extract entities from the match
                    entities = match.groupdict()

                    # Enhance entity extraction with additional patterns
                    enhanced_entities = self.enhance_entities(entities, text_lower)

                    # Calculate confidence based on match quality
                    confidence = self.calculate_confidence(pattern, match, text_lower)

                    return ParsedCommand(
                        intent=intent_type,
                        entities=enhanced_entities,
                        confidence=confidence,
                        original_text=text,
                        action=action
                    )

        # If no pattern matches, return None
        return None

    def enhance_entities(self, entities: Dict[str, str], text: str) -> Dict[str, str]:
        """Enhance entity extraction with additional patterns"""
        enhanced = entities.copy()

        # Look for additional entities that weren't captured by regex
        for entity_type, possible_values in self.entity_patterns.items():
            if entity_type not in enhanced:
                # Check for synonyms and related terms
                for value in possible_values:
                    if value in text:
                        enhanced[entity_type] = value
                        break

                # Check synonyms
                for main_value, synonyms in self.synonyms.items():
                    if main_value in text or any(syn in text for syn in synonyms):
                        enhanced[entity_type] = main_value
                        break

        return enhanced

    def calculate_confidence(self, pattern: str, match: re.Match, text: str) -> float:
        """Calculate confidence score for the match"""
        # Base confidence on pattern match
        base_confidence = 0.8

        # Boost confidence for longer, more specific matches
        match_length = len(match.group(0))
        text_length = len(text)

        if text_length > 0:
            length_ratio = match_length / text_length
            base_confidence += min(0.2, length_ratio * 0.1)

        # Additional boosts based on match quality
        if match.group(0).strip() == text.strip():
            # Exact match of the whole text
            base_confidence += 0.1

        return min(1.0, base_confidence)

    def extract_entities_by_type(self, text: str, entity_type: str) -> List[str]:
        """Extract entities of a specific type from text"""
        entities = []
        text_lower = text.lower()

        if entity_type in self.entity_patterns:
            for value in self.entity_patterns[entity_type]:
                if value in text_lower:
                    entities.append(value)

        # Check synonyms
        for main_value, synonyms in self.synonyms.items():
            if main_value in text_lower or any(syn in text_lower for syn in synonyms):
                entities.append(main_value)

        return entities


class NaturalLanguageUnderstanding:
    def __init__(self):
        self.command_parser = CommandParser()

        # Context management
        self.context = {
            'current_location': 'unknown',
            'last_interaction': None,
            'user_preferences': {},
            'environment_state': {}
        }

        # Intent handlers
        self.intent_handlers = {
            IntentType.NAVIGATION: self.handle_navigation_intent,
            IntentType.MANIPULATION: self.handle_manipulation_intent,
            IntentType.INFORMATION: self.handle_information_intent,
            IntentType.SOCIAL: self.handle_social_intent,
            IntentType.SYSTEM: self.handle_system_intent
        }

    def process_command(self, text: str) -> Optional[Dict]:
        """Process natural language command and return executable action"""
        # Parse the command
        parsed_command = self.command_parser.parse_command(text)

        if not parsed_command:
            return {
                'action': 'unknown_command',
                'response': "I didn't understand that command. Could you please rephrase?",
                'confidence': 0.0
            }

        # Handle the intent
        handler = self.intent_handlers.get(parsed_command.intent)
        if handler:
            result = handler(parsed_command)
        else:
            result = {
                'action': 'unknown_intent',
                'response': f"I'm not sure how to handle {parsed_command.intent.value} commands.",
                'confidence': parsed_command.confidence
            }

        # Add parsed command info to result
        result['parsed_command'] = parsed_command
        result['original_text'] = text

        return result

    def handle_navigation_intent(self, command: ParsedCommand) -> Dict:
        """Handle navigation-related intents"""
        entities = command.entities

        if command.action == 'go_to_destination':
            destination = entities.get('destination', 'unknown')
            return {
                'action': 'navigate',
                'destination': destination,
                'response': f"Okay, I'll go to the {destination}.",
                'confidence': command.confidence
            }
        elif command.action == 'move_direction':
            direction = entities.get('direction', 'forward')
            return {
                'action': 'move',
                'direction': direction,
                'response': f"Moving {direction}.",
                'confidence': command.confidence
            }
        elif command.action == 'come_to_user':
            return {
                'action': 'navigate_to_user',
                'response': "I'm coming to you.",
                'confidence': command.confidence
            }
        elif command.action == 'follow_user':
            return {
                'action': 'follow',
                'response': "I'll follow you.",
                'confidence': command.confidence
            }
        else:
            return {
                'action': 'navigation_unknown',
                'response': "I'm not sure where you want me to go.",
                'confidence': command.confidence
            }

    def handle_manipulation_intent(self, command: ParsedCommand) -> Dict:
        """Handle manipulation-related intents"""
        entities = command.entities

        if command.action in ['fetch_object', 'hand_object']:
            obj = entities.get('object', 'unknown')
            return {
                'action': 'fetch',
                'object': obj,
                'response': f"I'll get the {obj} for you.",
                'confidence': command.confidence
            }
        else:
            return {
                'action': 'manipulation_unknown',
                'response': "I'm not sure what you want me to do with objects.",
                'confidence': command.confidence
            }

    def handle_information_intent(self, command: ParsedCommand) -> Dict:
        """Handle information-related intents"""
        if command.action == 'describe_environment':
            return {
                'action': 'describe_environment',
                'response': "I can see several objects around us, including furniture and possible interaction points.",
                'confidence': command.confidence
            }
        elif command.action == 'check_wellbeing':
            return {
                'action': 'self_check',
                'response': "I'm functioning well, thank you for asking!",
                'confidence': command.confidence
            }
        elif command.action == 'get_time':
            from datetime import datetime
            current_time = datetime.now().strftime("%H:%M")
            return {
                'action': 'provide_time',
                'response': f"The current time is {current_time}.",
                'confidence': command.confidence
            }
        else:
            return {
                'action': 'information_unknown',
                'response': "I don't have that information available right now.",
                'confidence': command.confidence
            }

    def handle_social_intent(self, command: ParsedCommand) -> Dict:
        """Handle social interaction intents"""
        if command.action == 'greet':
            return {
                'action': 'greet',
                'response': "Hello! It's nice to meet you.",
                'confidence': command.confidence
            }
        elif command.action == 'farewell':
            return {
                'action': 'farewell',
                'response': "Goodbye! It was nice talking with you.",
                'confidence': command.confidence
            }
        elif command.action == 'acknowledge_gratitude':
            return {
                'action': 'acknowledge_gratitude',
                'response': "You're welcome! I'm happy to help.",
                'confidence': command.confidence
            }
        else:
            return {
                'action': 'social_unknown',
                'response': "I'm here to interact with you. How can I help?",
                'confidence': command.confidence
            }

    def handle_system_intent(self, command: ParsedCommand) -> Dict:
        """Handle system-related intents"""
        if command.action == 'stop_action':
            return {
                'action': 'stop',
                'response': "Stopping current action.",
                'confidence': command.confidence
            }
        elif command.action == 'show_help':
            return {
                'action': 'show_help',
                'response': "I can help with navigation, object fetching, answering questions, and social interaction. What would you like me to do?",
                'confidence': command.confidence
            }
        else:
            return {
                'action': 'system_unknown',
                'response': "I'm not sure how to handle that system command.",
                'confidence': command.confidence
            }

    def update_context(self, key: str, value):
        """Update context with new information"""
        self.context[key] = value

    def get_context(self, key: str, default=None):
        """Get context value"""
        return self.context.get(key, default)
```

## Voice Command Grammar and Validation

### Structured Command Processing

For humanoid robots, it's important to validate voice commands and provide structured responses:

```python
#!/usr/bin/env python3

import re
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum


class CommandValidationResult(Enum):
    VALID = "valid"
    INVALID_SYNTAX = "invalid_syntax"
    MISSING_ENTITY = "missing_entity"
    UNKNOWN_INTENT = "unknown_intent"
    LOW_CONFIDENCE = "low_confidence"


@dataclass
class ValidatedCommand:
    intent: str
    entities: Dict[str, str]
    validation_result: CommandValidationResult
    confidence: float
    suggestions: List[str]


class VoiceCommandGrammar:
    def __init__(self):
        # Define grammar rules for different command types
        self.grammar_rules = {
            'navigation': {
                'required_entities': ['destination'],
                'optional_entities': ['distance', 'speed'],
                'pattern': r'go to (?P<destination>\w+)( at (?P<speed>\w+) speed)?'
            },
            'manipulation': {
                'required_entities': ['action', 'object'],
                'optional_entities': ['location'],
                'pattern': r'(?P<action>get|bring|pick up|grab) (?P<object>\w+)( from (?P<location>\w+))?'
            },
            'information': {
                'required_entities': ['query_type'],
                'optional_entities': ['details'],
                'pattern': r'what (is|are) (?P<query_type>\w+)( (?P<details>.*))?'
            }
        }

        # Define entity validation rules
        self.entity_validators = {
            'destination': self.validate_destination,
            'object': self.validate_object,
            'action': self.validate_action,
            'query_type': self.validate_query_type
        }

        # Define valid values for entities
        self.valid_destinations = [
            'kitchen', 'living room', 'bedroom', 'office', 'bathroom',
            'dining room', 'hallway', 'entrance', 'exit'
        ]

        self.valid_objects = [
            'bottle', 'cup', 'book', 'phone', 'keys', 'ball',
            'water', 'coffee', 'tea', 'food', 'medicine'
        ]

        self.valid_actions = [
            'get', 'bring', 'pick up', 'grab', 'take', 'hand', 'give'
        ]

        self.valid_query_types = [
            'time', 'date', 'weather', 'temperature', 'news',
            'surroundings', 'environment', 'people'
        ]

    def validate_command(self, text: str, parsed_command: Dict) -> ValidatedCommand:
        """Validate a parsed command against grammar rules"""
        intent = parsed_command.get('intent', 'unknown')
        entities = parsed_command.get('entities', {})
        confidence = parsed_command.get('confidence', 0.0)

        # Check if intent is recognized
        if intent not in self.grammar_rules:
            return ValidatedCommand(
                intent=intent,
                entities=entities,
                validation_result=CommandValidationResult.UNKNOWN_INTENT,
                confidence=confidence,
                suggestions=self.get_intent_suggestions(text)
            )

        # Get grammar rule for this intent
        grammar_rule = self.grammar_rules[intent]

        # Check required entities
        missing_entities = []
        for required_entity in grammar_rule['required_entities']:
            if required_entity not in entities or not entities[required_entity]:
                missing_entities.append(required_entity)

        if missing_entities:
            return ValidatedCommand(
                intent=intent,
                entities=entities,
                validation_result=CommandValidationResult.MISSING_ENTITY,
                confidence=confidence,
                suggestions=self.get_entity_suggestions(intent, missing_entities)
            )

        # Validate individual entities
        invalid_entities = []
        for entity_name, entity_value in entities.items():
            if entity_name in self.entity_validators:
                is_valid = self.entity_validators[entity_name](entity_value)
                if not is_valid:
                    invalid_entities.append((entity_name, entity_value))

        if invalid_entities:
            return ValidatedCommand(
                intent=intent,
                entities=entities,
                validation_result=CommandValidationResult.INVALID_SYNTAX,
                confidence=confidence,
                suggestions=self.get_correction_suggestions(invalid_entities)
            )

        # Check confidence threshold
        if confidence < 0.6:  # Adjust threshold as needed
            return ValidatedCommand(
                intent=intent,
                entities=entities,
                validation_result=CommandValidationResult.LOW_CONFIDENCE,
                confidence=confidence,
                suggestions=['Could you please repeat that?', 'I didn\'t catch that clearly.']
            )

        # Command is valid
        return ValidatedCommand(
            intent=intent,
            entities=entities,
            validation_result=CommandValidationResult.VALID,
            confidence=confidence,
            suggestions=[]
        )

    def validate_destination(self, destination: str) -> bool:
        """Validate destination entity"""
        return destination.lower() in [d.lower() for d in self.valid_destinations]

    def validate_object(self, obj: str) -> bool:
        """Validate object entity"""
        return obj.lower() in [o.lower() for o in self.valid_objects]

    def validate_action(self, action: str) -> bool:
        """Validate action entity"""
        return action.lower() in [a.lower() for a in self.valid_actions]

    def validate_query_type(self, query_type: str) -> bool:
        """Validate query type entity"""
        return query_type.lower() in [q.lower() for q in self.valid_query_types]

    def get_intent_suggestions(self, text: str) -> List[str]:
        """Get suggestions for unknown intents"""
        suggestions = []

        # Check if text contains keywords related to known intents
        text_lower = text.lower()

        if any(keyword in text_lower for keyword in ['go', 'move', 'navigate', 'walk']):
            suggestions.append('Try a navigation command like "go to kitchen"')

        if any(keyword in text_lower for keyword in ['get', 'bring', 'pick', 'grab']):
            suggestions.append('Try a manipulation command like "get the cup"')

        if any(keyword in text_lower for keyword in ['what', 'tell', 'how', 'when']):
            suggestions.append('Try an information command like "what time is it"')

        return suggestions or ['I didn\'t understand that. Try commands like "go to kitchen", "get the cup", or "what time is it".']

    def get_entity_suggestions(self, intent: str, missing_entities: List[str]) -> List[str]:
        """Get suggestions for missing entities"""
        suggestions = []

        for entity in missing_entities:
            if entity == 'destination':
                suggestions.append(f'Please specify a destination. Valid options: {", ".join(self.valid_destinations[:5])}')
            elif entity == 'object':
                suggestions.append(f'Please specify an object. Valid options: {", ".join(self.valid_objects[:5])}')
            elif entity == 'action':
                suggestions.append(f'Please specify an action. Valid options: {", ".join(self.valid_actions[:5])}')
            elif entity == 'query_type':
                suggestions.append(f'Please specify what information you want. Valid options: {", ".join(self.valid_query_types[:5])}')

        return suggestions

    def get_correction_suggestions(self, invalid_entities: List[tuple]) -> List[str]:
        """Get suggestions for correcting invalid entities"""
        suggestions = []

        for entity_name, entity_value in invalid_entities:
            if entity_name == 'destination':
                suggestions.append(f'Did you mean one of these destinations: {", ".join(self.valid_destinations[:3])}?')
            elif entity_name == 'object':
                suggestions.append(f'Did you mean one of these objects: {", ".join(self.valid_objects[:3])}?')
            elif entity_name == 'action':
                suggestions.append(f'Did you mean one of these actions: {", ".join(self.valid_actions[:3])}?')

        return suggestions


class CommandValidator:
    def __init__(self):
        self.grammar = VoiceCommandGrammar()

    def validate_and_process(self, text: str, parsed_command: Dict) -> Dict:
        """Validate command and return processing result"""
        validated = self.grammar.validate_command(text, parsed_command)

        result = {
            'original_text': text,
            'intent': validated.intent,
            'entities': validated.entities,
            'confidence': validated.confidence,
            'is_valid': validated.validation_result == CommandValidationResult.VALID,
            'validation_result': validated.validation_result.value,
            'suggestions': validated.suggestions
        }

        # If command is invalid, add appropriate response
        if not result['is_valid']:
            result['response'] = self.generate_validation_response(validated)
        else:
            result['response'] = f"Command validated successfully: {validated.intent}"

        return result

    def generate_validation_response(self, validated: ValidatedCommand) -> str:
        """Generate appropriate response for validation result"""
        if validated.validation_result == CommandValidationResult.UNKNOWN_INTENT:
            return "I didn't understand that command. " + (validated.suggestions[0] if validated.suggestions else "")
        elif validated.validation_result == CommandValidationResult.MISSING_ENTITY:
            return "Your command is missing required information. " + (validated.suggestions[0] if validated.suggestions else "")
        elif validated.validation_result == CommandValidationResult.INVALID_SYNTAX:
            return "I didn't understand that part of your command. " + (validated.suggestions[0] if validated.suggestions else "")
        elif validated.validation_result == CommandValidationResult.LOW_CONFIDENCE:
            return "I'm not confident I understood that correctly. " + (validated.suggestions[0] if validated.suggestions else "")
        else:
            return "Command processed successfully."
```

## Text-to-Speech and Voice Synthesis

### Implementing Natural Robot Responses

For humanoid robots, natural-sounding responses are important for good user experience:

```python
#!/usr/bin/env python3

import pyttsx3
import threading
import queue
import time
from typing import Optional


class TextToSpeechSystem:
    def __init__(self):
        # Initialize pyttsx3 engine
        self.engine = pyttsx3.init()

        # Configure voice properties
        self.configure_voice()

        # Queues for speech processing
        self.speech_queue = queue.Queue()
        self.is_speaking = False

        # Voice properties
        self.rate = 200  # Words per minute
        self.volume = 0.9  # Volume level (0.0 to 1.0)
        self.voice_type = 'default'  # Can be 'male', 'female', 'child', etc.

        self.get_logger().info('Text-to-speech system initialized')

    def configure_voice(self):
        """Configure the voice properties"""
        try:
            # Get available voices
            voices = self.engine.getProperty('voices')

            # Set a female voice if available (usually index 1), otherwise use default
            if len(voices) > 1:
                self.engine.setProperty('voice', voices[1].id)  # Female voice
            else:
                self.engine.setProperty('voice', voices[0].id)  # Default voice

            # Set speech rate
            self.engine.setProperty('rate', self.rate)

            # Set volume
            self.engine.setProperty('volume', self.volume)

        except Exception as e:
            self.get_logger().error(f'Error configuring voice: {e}')

    def speak_text(self, text: str, blocking: bool = False):
        """Speak the given text"""
        if blocking:
            # Speak directly without queuing
            self.engine.say(text)
            self.engine.runAndWait()
        else:
            # Add to queue for non-blocking speech
            self.speech_queue.put(text)

            # Start speaking thread if not already running
            if not hasattr(self, 'speaking_thread') or not self.speaking_thread.is_alive():
                self.speaking_thread = threading.Thread(target=self.speech_worker)
                self.speaking_thread.daemon = True
                self.speaking_thread.start()

    def speech_worker(self):
        """Worker thread for processing speech queue"""
        while True:
            try:
                # Get text from queue
                text = self.speech_queue.get(timeout=1.0)

                # Set speaking flag
                self.is_speaking = True

                # Speak the text
                self.engine.say(text)
                self.engine.runAndWait()

                # Clear speaking flag
                self.is_speaking = False

                # Mark task as done
                self.speech_queue.task_done()

            except queue.Empty:
                # Check if we should continue
                if self.speech_queue.empty():
                    break
                continue
            except Exception as e:
                self.get_logger().error(f'Speech worker error: {e}')
                self.is_speaking = False
                break

    def is_busy(self) -> bool:
        """Check if the TTS system is currently speaking"""
        return self.is_speaking or not self.speech_queue.empty()

    def stop_speaking(self):
        """Stop current speech"""
        self.engine.stop()

    def set_voice_properties(self, rate: int = None, volume: float = None, voice_type: str = None):
        """Set voice properties"""
        if rate is not None:
            self.rate = rate
            self.engine.setProperty('rate', rate)

        if volume is not None:
            self.volume = volume
            self.engine.setProperty('volume', volume)

        if voice_type is not None:
            self.voice_type = voice_type
            # In a real implementation, you would select different voices based on type
            # For now, just store the type

    def get_available_voices(self):
        """Get list of available voices"""
        try:
            voices = self.engine.getProperty('voices')
            return [voice.name for voice in voices]
        except Exception as e:
            self.get_logger().error(f'Error getting voices: {e}')
            return []

    def get_logger(self):
        """Simple logger for simulation"""
        class Logger:
            def info(self, msg):
                print(f'[INFO] {msg}')
            def error(self, msg):
                print(f'[ERROR] {msg}')
        return Logger()


class ContextualResponseGenerator:
    def __init__(self):
        self.tts_system = TextToSpeechSystem()

        # Context for generating appropriate responses
        self.context = {
            'user_name': None,
            'interaction_history': [],
            'current_task': None,
            'robot_state': 'idle',
            'environment_context': {}
        }

        # Response templates
        self.response_templates = {
            'navigation_success': [
                "I have reached the {destination}.",
                "Here we are at the {destination}.",
                "Arrived at the {destination}."
            ],
            'navigation_in_progress': [
                "I am going to the {destination}.",
                "On my way to the {destination}.",
                "Navigating to the {destination}."
            ],
            'object_fetch_success': [
                "I have picked up the {object}.",
                "Got the {object} for you.",
                "Successfully retrieved the {object}."
            ],
            'object_fetch_in_progress': [
                "I am getting the {object}.",
                "Fetching the {object} now.",
                "Looking for the {object}."
            ],
            'acknowledgment': [
                "I understand.",
                "Got it.",
                "I'll take care of that.",
                "Understood."
            ],
            'error_generic': [
                "I'm sorry, I couldn't do that.",
                "Something went wrong with that request.",
                "I couldn't complete that task."
            ]
        }

    def generate_response(self, action_result: Dict) -> str:
        """Generate contextual response based on action result"""
        action = action_result.get('action', 'unknown')
        entities = action_result.get('entities', {})

        # Generate response based on action type
        if action == 'navigate':
            destination = entities.get('destination', 'unknown')
            import random
            template = random.choice(self.response_templates['navigation_in_progress'])
            return template.format(destination=destination)

        elif action == 'fetch':
            obj = entities.get('object', 'unknown')
            import random
            template = random.choice(self.response_templates['object_fetch_in_progress'])
            return template.format(object=obj)

        elif action == 'greet':
            return "Hello! How can I assist you today?"

        elif action == 'farewell':
            return "Goodbye! Have a great day!"

        elif action == 'unknown_command':
            return "I didn't understand that command. Could you please rephrase?"

        else:
            # Default acknowledgment
            import random
            return random.choice(self.response_templates['acknowledgment'])

    def speak_response(self, response: str, context_aware: bool = True):
        """Speak the response with optional context awareness"""
        if context_aware:
            # Add personalization if user name is known
            if self.context['user_name']:
                response = f"{self.context['user_name']}, {response}"

        # Add to interaction history
        self.context['interaction_history'].append({
            'type': 'response',
            'text': response,
            'timestamp': time.time()
        })

        # Speak the response
        self.tts_system.speak_text(response)

    def update_context(self, key: str, value):
        """Update context with new information"""
        self.context[key] = value

    def get_context(self, key: str, default=None):
        """Get context value"""
        return self.context.get(key, default)
```

## Complete Voice Command Integration System

### Putting It All Together

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
import numpy as np
import time
from typing import Dict, Optional


class VoiceCommandIntegrationSystem(Node):
    def __init__(self):
        super().__init__('voice_command_integration_system')

        # Initialize subsystems
        self.audio_capture = AudioCaptureSystem()
        self.asr_system = HybridASRSystem()
        self.nlu_system = NaturalLanguageUnderstanding()
        self.command_validator = CommandValidator()
        self.response_generator = ContextualResponseGenerator()

        # ROS2 publishers and subscribers
        self.speech_sub = self.create_subscription(
            String, '/speech_input', self.speech_callback, 10)
        self.voice_command_pub = self.create_publisher(
            String, '/voice_command', 10)
        self.robot_action_pub = self.create_publisher(
            String, '/robot_action', 10)
        self.response_pub = self.create_publisher(
            String, '/response_output', 10)

        # System state
        self.is_active = True
        self.last_command_time = time.time()
        self.command_history = []

        # Start audio capture
        self.audio_capture.start_capture()

        self.get_logger().info('Voice command integration system initialized')

    def speech_callback(self, msg):
        """Process incoming speech commands"""
        speech_text = msg.data
        self.get_logger().info(f'Received speech: {speech_text}')

        # Process the speech command through the pipeline
        result = self.process_voice_command(speech_text)

        if result:
            # Publish the processed command
            command_msg = String()
            command_msg.data = str(result)
            self.voice_command_pub.publish(command_msg)

            # Generate and speak response
            response = result.get('response', 'Command processed')
            self.response_generator.speak_response(response)

            # Publish response
            response_msg = String()
            response_msg.data = response
            self.response_pub.publish(response_msg)

            # Log the interaction
            self.log_interaction(speech_text, result)

    def process_voice_command(self, text: str) -> Optional[Dict]:
        """Process voice command through the complete pipeline"""
        try:
            # Step 1: Natural Language Understanding
            nlu_result = self.nlu_system.process_command(text)

            if not nlu_result:
                return {
                    'action': 'unknown',
                    'response': "I didn't understand that command.",
                    'confidence': 0.0,
                    'original_text': text
                }

            # Step 2: Command Validation
            validation_result = self.command_validator.validate_and_process(
                text, nlu_result
            )

            if not validation_result['is_valid']:
                return {
                    'action': 'validation_failed',
                    'response': validation_result['response'],
                    'suggestions': validation_result['suggestions'],
                    'original_text': text
                }

            # Step 3: Execute action (in simulation, just return the command)
            action_result = self.execute_command(nlu_result)

            # Combine results
            result = {
                'action': nlu_result['action'],
                'entities': nlu_result.get('entities', {}),
                'confidence': nlu_result.get('confidence', 0.0),
                'original_text': text,
                'validation': validation_result,
                'execution_result': action_result
            }

            # Generate appropriate response
            response = self.response_generator.generate_response(nlu_result)
            result['response'] = response

            return result

        except Exception as e:
            self.get_logger().error(f'Error processing voice command: {e}')
            return {
                'action': 'error',
                'response': "I encountered an error processing your command.",
                'original_text': text,
                'error': str(e)
            }

    def execute_command(self, nlu_result: Dict) -> Dict:
        """Execute the parsed command (simulation)"""
        action = nlu_result.get('action', 'unknown')
        entities = nlu_result.get('entities', {})

        # In a real system, this would interface with the robot's action system
        # For simulation, we'll just return a success result
        return {
            'status': 'success',
            'action': action,
            'entities': entities,
            'execution_time': 0.1  # Simulated execution time
        }

    def log_interaction(self, input_text: str, result: Dict):
        """Log the voice command interaction"""
        interaction = {
            'input': input_text,
            'output': result,
            'timestamp': time.time()
        }

        self.command_history.append(interaction)

        # Keep only recent history
        if len(self.command_history) > 50:
            self.command_history = self.command_history[-50:]

    def get_logger(self):
        """Get ROS2 logger"""
        return self.get_logger()


def main(args=None):
    rclpy.init(args=args)
    voice_system = VoiceCommandIntegrationSystem()

    try:
        # Start the system
        rclpy.spin(voice_system)
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        voice_system.audio_capture.stop_capture()
        voice_system.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Performance Optimization and Testing

### Optimizing Voice Command Systems

```python
#!/usr/bin/env python3

import time
import threading
import queue
from collections import defaultdict, deque
import psutil


class VoiceCommandOptimizer:
    def __init__(self):
        self.component_times = defaultdict(list)
        self.resource_usage = {}
        self.performance_thresholds = {
            'asr_latency': 1.0,      # seconds
            'nlu_latency': 0.1,      # seconds
            'tts_latency': 0.5,      # seconds
            'cpu_threshold': 80.0,   # percent
            'memory_threshold': 80.0 # percent
        }

        # Processing queues
        self.processing_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue()

        # Adaptive parameters
        self.adaptive_params = {
            'asr_quality': 'base',  # 'tiny', 'base', 'small', 'medium', 'large'
            'recognition_rate': 10,  # Hz
            'buffer_size': 1024
        }

        self.get_logger().info('Voice command optimizer initialized')

    def monitor_performance(self):
        """Monitor system performance and adjust parameters"""
        while True:
            # Monitor resource usage
            self.resource_usage = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'timestamp': time.time()
            }

            # Adjust parameters based on load
            self.adapt_to_load()

            time.sleep(1.0)  # Monitor every second

    def adapt_to_load(self):
        """Adapt processing parameters based on system load"""
        cpu_usage = self.resource_usage.get('cpu_percent', 0)
        memory_usage = self.resource_usage.get('memory_percent', 0)

        # Adjust ASR model quality based on CPU usage
        if cpu_usage > self.performance_thresholds['cpu_threshold']:
            # Reduce ASR quality to save CPU
            current_quality = self.adaptive_params['asr_quality']
            quality_levels = ['large', 'medium', 'small', 'base', 'tiny']
            current_idx = quality_levels.index(current_quality) if current_quality in quality_levels else 3

            if current_idx < len(quality_levels) - 1:
                self.adaptive_params['asr_quality'] = quality_levels[current_idx + 1]
        elif cpu_usage < 50:
            # Increase ASR quality if CPU is underutilized
            current_quality = self.adaptive_params['asr_quality']
            quality_levels = ['large', 'medium', 'small', 'base', 'tiny']
            current_idx = quality_levels.index(current_quality) if current_quality in quality_levels else 4

            if current_idx > 0:
                self.adaptive_params['asr_quality'] = quality_levels[current_idx - 1]

    def measure_component_performance(self, component_name: str, func, *args, **kwargs):
        """Measure performance of a component"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        processing_time = end_time - start_time

        # Record timing
        self.component_times[component_name].append(processing_time)

        # Keep only recent measurements
        if len(self.component_times[component_name]) > 100:
            self.component_times[component_name] = self.component_times[component_name][-100:]

        return result, processing_time

    def get_average_component_time(self, component_name: str) -> float:
        """Get average processing time for a component"""
        times = self.component_times[component_name]
        return sum(times) / len(times) if times else 0.0

    def get_throughput(self, component_name: str) -> float:
        """Get processing throughput for a component"""
        times = self.component_times[component_name]
        if len(times) > 1:
            total_time = times[-1] - times[0]  # Approximate time span
            return len(times) / total_time if total_time > 0 else 0.0
        return 0.0

    def start_monitoring(self):
        """Start performance monitoring in a separate thread"""
        monitor_thread = threading.Thread(target=self.monitor_performance)
        monitor_thread.daemon = True
        monitor_thread.start()

    def get_logger(self):
        """Simple logger for simulation"""
        class Logger:
            def info(self, msg):
                print(f'[INFO] {msg}')
            def error(self, msg):
                print(f'[ERROR] {msg}')
        return Logger()


class VoiceCommandTester:
    def __init__(self):
        self.test_results = []
        self.test_scenarios = [
            {
                'name': 'basic_navigation',
                'input': 'go to kitchen',
                'expected_intent': 'navigation',
                'expected_entities': {'destination': 'kitchen'}
            },
            {
                'name': 'object_fetching',
                'input': 'get me the red cup',
                'expected_intent': 'manipulation',
                'expected_entities': {'object': 'cup'}
            },
            {
                'name': 'information_request',
                'input': 'what time is it',
                'expected_intent': 'information',
                'expected_entities': {'query_type': 'time'}
            },
            {
                'name': 'social_interaction',
                'input': 'hello how are you',
                'expected_intent': 'social',
                'expected_entities': {}
            }
        ]

    def run_tests(self, voice_system):
        """Run comprehensive tests on the voice command system"""
        self.get_logger().info('Starting voice command system tests')

        for scenario in self.test_scenarios:
            result = self.test_scenario(voice_system, scenario)
            self.test_results.append(result)

        self.get_logger().info(f'Completed {len(self.test_results)} tests')
        self.print_test_summary()

    def test_scenario(self, voice_system, scenario):
        """Test a specific scenario"""
        start_time = time.time()

        try:
            # Process the command
            result = voice_system.process_voice_command(scenario['input'])

            # Check if results match expectations
            success = self.validate_result(result, scenario)

            end_time = time.time()
            processing_time = end_time - start_time

            test_result = {
                'scenario_name': scenario['name'],
                'input': scenario['input'],
                'expected': {
                    'intent': scenario['expected_intent'],
                    'entities': scenario['expected_entities']
                },
                'actual': {
                    'intent': result.get('action', 'unknown') if result else 'none',
                    'entities': result.get('entities', {}) if result else {}
                },
                'success': success,
                'processing_time': processing_time,
                'timestamp': start_time
            }

        except Exception as e:
            test_result = {
                'scenario_name': scenario['name'],
                'input': scenario['input'],
                'expected': {
                    'intent': scenario['expected_intent'],
                    'entities': scenario['expected_entities']
                },
                'actual': {'error': str(e)},
                'success': False,
                'processing_time': time.time() - start_time,
                'timestamp': start_time
            }

        return test_result

    def validate_result(self, result, scenario):
        """Validate if the result matches expected values"""
        if not result:
            return False

        # Check intent
        actual_intent = result.get('action', 'unknown')
        expected_intent = scenario['expected_intent']

        # Simple intent matching (in practice, use more sophisticated matching)
        intent_match = actual_intent.startswith(expected_intent.lower()) or \
                      expected_intent.lower().startswith(actual_intent.lower())

        # Check entities
        actual_entities = result.get('entities', {})
        expected_entities = scenario['expected_entities']

        entity_match = True
        for key, expected_value in expected_entities.items():
            actual_value = actual_entities.get(key)
            if actual_value and expected_value.lower() in actual_value.lower():
                continue
            else:
                entity_match = False
                break

        return intent_match and entity_match

    def print_test_summary(self):
        """Print summary of test results"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests

        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        self.get_logger().info(f'Test Summary:')
        self.get_logger().info(f'  Total tests: {total_tests}')
        self.get_logger().info(f'  Passed: {passed_tests}')
        self.get_logger().info(f'  Failed: {failed_tests}')
        self.get_logger().info(f'  Success rate: {success_rate:.1f}%')

        # Print details for failed tests
        for result in self.test_results:
            if not result['success']:
                self.get_logger().info(f'  FAILED: {result["scenario_name"]} - "{result["input"]}"')

    def get_logger(self):
        """Simple logger for simulation"""
        class Logger:
            def info(self, msg):
                print(f'[INFO] {msg}')
            def error(self, msg):
                print(f'[ERROR] {msg}')
        return Logger()
```

## Troubleshooting Common Issues

### Common Voice Command Problems and Solutions

#### Issue 1: Poor Recognition Accuracy
**Symptoms**: Commands frequently misunderstood or not recognized
**Causes**:
- Background noise
- Poor microphone quality
- Accents or speech patterns not handled
- Insufficient training data
**Solutions**:
- Implement noise reduction preprocessing
- Use beamforming microphone arrays
- Train models on diverse speech data
- Implement confidence thresholds
- Add command confirmation mechanisms

#### Issue 2: High Latency
**Symptoms**: Long delays between speech input and robot response
**Causes**:
- Large ASR models
- Network delays (for cloud services)
- Inefficient processing pipelines
- Resource constraints
**Solutions**:
- Use smaller, faster models for real-time processing
- Implement local processing for critical commands
- Optimize processing pipelines
- Use edge computing solutions

#### Issue 3: Context Loss
**Symptoms**: Robot doesn't remember previous interactions or context
**Causes**:
- State not properly maintained
- Conversational context not tracked
- Multi-turn dialogue not supported
**Solutions**:
- Implement proper context management
- Use dialogue state tracking
- Support follow-up commands
- Maintain interaction history

#### Issue 4: False Activations
**Symptoms**: Robot responds to background speech or non-commands
**Causes**:
- Poor voice activity detection
- Wake word detection too sensitive
- Environmental noise triggering responses
**Solutions**:
- Improve voice activity detection
- Implement wake word functionality
- Use speaker identification
- Add confirmation prompts

## Best Practices for Voice Command Systems

### 1. User Experience Design
- Provide clear feedback for recognized commands
- Use natural, conversational language
- Implement graceful error handling
- Support multi-modal interaction (voice + gesture)

### 2. Robustness and Reliability
- Implement fallback mechanisms
- Handle edge cases gracefully
- Provide alternative interaction methods
- Include error recovery procedures

### 3. Performance Optimization
- Optimize for real-time processing
- Use adaptive quality based on system load
- Implement efficient resource management
- Monitor and log performance metrics

### 4. Privacy and Security
- Protect user voice data
- Implement secure communication
- Provide data privacy controls
- Consider local processing for sensitive commands

## Key Takeaways

- Voice command systems enable natural human-robot interaction
- Audio preprocessing improves recognition accuracy
- Natural language understanding extracts intent from speech
- Command validation ensures reliable execution
- Text-to-speech provides natural robot responses
- Performance optimization is critical for real-time operation
- Testing ensures system reliability and usability

In the next section, we'll explore AI Robot Brain exercises with working examples to practice voice command integration.