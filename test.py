import sounddevice as sd
from scipy.io.wavfile import write
import assemblyai as aai
import time
duration = 20  # Duration of recording in seconds
sample_rate = 44100  # Sample rate (Hz)
output_filename = 'output.wav'  # Filename to save recorded audio

print("Recording...")
audio_data = sd.rec(int(duration * sample_rate),
                    samplerate=sample_rate, channels=1)
sd.wait()  # Wait until the recording is finished
print("Recording complete")

write(output_filename, sample_rate, audio_data)

aai.settings.api_key = "6ef4e563df034841b4763e398813d40e"

config = aai.TranscriptionConfig(language_code="fr")
transcription_start_time = time.time()

with open(output_filename, 'rb') as audio_file:
    # Upload the audio file to AssemblyAI
    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(
        audio_file,
    )

    transcription_end_time = time.time()

    # Print the full transcription text
    print("Transcription:")
    print(transcript.text)
    print(
        f"Time taken to transcribe: {transcription_end_time - transcription_start_time:.2f} seconds")
