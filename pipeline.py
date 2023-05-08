import kfp
import kfp.dsl as dsl

from kfp import compiler
from kfp.dsl import Dataset, Input, Output


FFMPEG_IMAGE = "linuxserver/ffmpeg:5.1.2"
GCS_BUCKET = "example"
MP4_FILE = "example.mp4"
WAV_FILE = "example.wav"

@dsl.container_component
def mp4_to_wav(mp4_file: Input[Dataset], wav_file: str, local_wav_file: str):
    return dsl.ContainerSpec(
        image=FFMPEG_IMAGE,
        args=[
            "-i", mp4_file.metadata["local_gcs_uri"],
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "48000",
            "-ac", "2",
            local_wav_file
        ]
    )

@dsl.component(
    base_image='python:3.11',
    packages_to_install=['google-cloud-speech', 'appengine-python-standard']
)
def google_speech_to_text(wav_file: str):
    import json
    import os
    import re

    from google.cloud import speech

    client = speech.SpeechClient()

    audio = speech.RecognitionAudio(
        uri = wav_file
    )

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=48000,
        language_code="en-US",
        audio_channel_count=2,
        model="video",
    )

    operation = client.long_running_recognize(config=config, audio=audio)
    response = operation.result()

    with open(wav_file.replace("gs://", "/gcs/").replace(".wav", ".txt"), 'w') as file:
        for result in response.results:
            file.write(result.alternatives[0].transcript + "\n")

@dsl.pipeline(
    name="transcript-extraction"
)
def transcript_extraction():
    mp4_file = dsl.importer(
        artifact_uri=f'gs://{GCS_BUCKET}/{MP4_FILE}',
        artifact_class=Dataset,
        reimport=False,
        metadata={
            "local_gcs_uri": f'/gcs/{GCS_BUCKET}/{MP4_FILE}'
        }
    )
    mp4_to_wav_task = mp4_to_wav(
        mp4_file=mp4_file.output,
        wav_file=f"gs://{GCS_BUCKET}/{WAV_FILE}",
        local_wav_file=f"/gcs/{GCS_BUCKET}/{WAV_FILE}"
    )
    mp4_to_wav_task.set_cpu_request("4")
    mp4_to_wav_task.set_cpu_limit("4")
    mp4_to_wav_task.set_memory_request("32Gi")
    mp4_to_wav_task.set_memory_limit("32Gi")
    google_speech_to_text_task = google_speech_to_text(
        wav_file=f"gs://{GCS_BUCKET}/{WAV_FILE}"
    )
    google_speech_to_text_task.after(mp4_to_wav_task)

compiler.Compiler().compile(transcript_extraction, 'pipeline.yaml')