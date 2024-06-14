from google.cloud import speech_v1p1beta1 as speech
from google.cloud import storage
import wave
import numpy as np
import io

# Separate the speakers out.
def diarize_audio(bucket_title, file_title, speaker_count: int):
    client = speech.SpeechClient()

    audio = speech.RecognitionAudio(uri=rf"gs://{bucket_title}/{file_title}")
    config = speech.RecognitionConfig(
     encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
     language_code="en-GB",
     enable_speaker_diarization=True,
     diarization_speaker_count=speaker_count,
     audio_channel_count=2,
     enable_separate_recognition_per_channel=True,
    )

    operation = client.long_running_recognize(config=config, audio=audio)
    response = operation.result(timeout=1200)  # 20 minute timeout

    #  Diarization result in milliseconds
    diarization_result = []
    for result in response.results:
     for word_info in result.alternatives[0].words:
      diarization_result.append((word_info.start_time.total_seconds() * 1000, 
         word_info.end_time.total_seconds() * 1000, 
         word_info.speaker_tag))
    return diarization_result

def download_from_gcs(bucket_name, source_blob_name):
    storage_client = storage.Client(project='auto-diarizer')
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    data = blob.download_as_bytes()
    return io.BytesIO(data)

def read_wave(data):
    with wave.open(data, 'rb') as wf:
     params = wf.getparams()
     frames = wf.readframes(params.nframes)
     audio_data = np.frombuffer(frames, dtype=np.int16)
    return params, audio_data

def write_wave(file_path, params, audio_data):
    with wave.open(file_path, 'wb') as wf:
     wf.setparams(params)
     wf.writeframes(audio_data.tobytes())

def mute_segments(audio_data, params, segments_to_mute):
    sample_rate = params.framerate
    audio_data_writable = np.copy(audio_data)


    for start, end in segments_to_mute:
     start_frame = int(start * sample_rate / 1000)  # Convert ms to frames
     end_frame = int(end * sample_rate / 1000)      # Convert ms to frames
     audio_data_writable[start_frame:end_frame] = 0
    return audio_data_writable

# Silence the audio sections which don’t correspond to the main speaker.
def process_audio(bucket_title, file_title, speaker_to_keep, speaker_count):
    diarization_result = diarize_audio(bucket_title, file_title, speaker_count)

    audio_data_io = download_from_gcs(bucket_title, file_title)
    params, audio_data = read_wave(audio_data_io)
    
    segments_to_mute = [(start, end) for start, end, speaker in diarization_result if speaker not in speaker_to_keep]
    muted_audio_data = mute_segments(audio_data, params, segments_to_mute)
    
    return params, muted_audio_data

# Example usage
bucket_title = 'diarization-audio-storage' 
file_title = 'RAW_02_Katie_Audio_1.wav'
# Adjust the below based on your needs
speaker_to_keep = [1]  
speaker_count = 2

params, processed_audio_data = process_audio(bucket_title, file_title, speaker_to_keep, speaker_count)

output_file_path = rf"C:\Users\hugo\OneDrive\Pulpit\processed_{file_title}"
write_wave(output_file_path, params, processed_audio_data)