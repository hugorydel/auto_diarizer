from google.cloud import speech_v1p1beta1 as speech
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor
import wave
import numpy as np
import io
import os

# Google Cloud Storage Functions
def upload_to_gcs(bucket_title, local_file_path, file_title):
    storage_client = storage.Client(project='auto-diarizer')
    bucket = storage_client.bucket(bucket_title)
    blob = bucket.blob(file_title)
    blob.upload_from_filename(local_file_path)
    print(f"File {local_file_path} uploaded to gs://{bucket_title}/{file_title}.")
def download_from_gcs(bucket_title, file_title):
    storage_client = storage.Client(project='auto-diarizer')
    bucket = storage_client.bucket(bucket_title)
    blob = bucket.blob(file_title)
    data = blob.download_as_bytes()
    return io.BytesIO(data)

# Function used to call the Google Cloud Speech-To-Text Diarization
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
    response = operation.result(timeout=2400)  # 40 minute timeout

    diarization_result = []
    for result in response.results:
        for word_info in result.alternatives[0].words:
            diarization_result.append((word_info.start_time.total_seconds() * 1000,
               word_info.end_time.total_seconds() * 1000,
               word_info.speaker_tag))
    return diarization_result

# Audio Editing Functions - Convert audio into a format that can be directly edited programmatically
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

# Function used to silence parts of the audio
def mute_segments(audio_data, params, segments_to_mute):
    sample_rate = params.framerate
    audio_data_writable = np.copy(audio_data)

    for start, end in segments_to_mute:
        start_frame = int(start * sample_rate / 1000)  # Convert ms to frames
        end_frame = int(end * sample_rate / 1000)      # Convert ms to frames
        audio_data_writable[start_frame:end_frame] = 0
    return audio_data_writable

# Functions used to select quietest speaker
def calculate_loudness(audio_data, start_frame, end_frame):
    segment = audio_data[start_frame:end_frame]
    loudness = 20 * np.log10(np.sqrt(np.mean(segment**2)))
    return loudness
def get_quietest_speaker(diarization_result, audio_data, params):
    speaker_loudness = {}
    sample_rate = params.framerate

    for start, end, speaker in diarization_result:
        start_frame = int(start * sample_rate / 1000)
        end_frame = int(end * sample_rate / 1000)
        loudness = calculate_loudness(audio_data, start_frame, end_frame)
        
        if speaker in speaker_loudness:
            speaker_loudness[speaker].append(loudness)
        else:
            speaker_loudness[speaker] = [loudness]

    average_loudness = {speaker: np.mean(loudnesses) for speaker, loudnesses in speaker_loudness.items()}
    quietest_speaker = min(average_loudness, key=average_loudness.get)
    return quietest_speaker

# Program settings, including the bucket to use the files to upload, and the amount of speakers per file.
bucket_title = 'diarization-audio-storage'
local_file_paths = [rf"C:\Users\hugos\OneDrive\Pulpit\02_Pod_Katie Moran_1\RAW_02_Katie_Moran_1-01.wav", rf"C:\Users\hugos\OneDrive\Pulpit\02_Pod_Katie Moran_1\RAW_02_Katie_Moran_1-02.wav"]
speaker_count = 2


def process_file(bucket_title, local_file_path, speaker_count):
    file_title = os.path.basename(local_file_path)

    # Step 1: Upload local file to GCS
    upload_to_gcs(bucket_title, local_file_path, file_title)
    
    # Step 2: Using Google Cloud's Speech-To-Text diarize the audio files you uploaded, getting the specific timestamps at which each speaker is talking
    diarization_result = diarize_audio(bucket_title, file_title, speaker_count)

    # Step 3: Download the uploaded audio files to prepare them for editing
    audio_data_io = download_from_gcs(bucket_title, file_title)

    # Step 4: Read the audio data from the downloaded file
    params, audio_data = read_wave(audio_data_io)

    # Step 5: In the audio file, identify the quietest speaker (each speaker has a number assigned to them, so this will identify the number which is quietest)
    quietest_speaker = get_quietest_speaker(diarization_result, audio_data, params)
    
    # Step 6: Mute all the audio segments associated with the quietest speaker.
    segments_to_mute = [(start, end) for start, end, speaker in diarization_result if speaker == quietest_speaker]
    processed_audio_data = mute_segments(audio_data, params, segments_to_mute)

    # Step 7: Save the edited audio file (i.e. with the noise from the quietest speaker removed) to your local machine
    output_file_path = os.path.join(os.path.dirname(local_file_path), f"processed_{file_title}")
    write_wave(output_file_path, params, processed_audio_data)
    
    print(f"Processed file saved to {output_file_path}")


# Process files concurrently
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_file, bucket_title, local_file_path, speaker_count) for local_file_path in local_file_paths]
    for future in futures:
        future.result()  # Wait for all tasks to complete