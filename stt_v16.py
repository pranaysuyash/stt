import os
from moviepy.editor import VideoFileClip
from pydub import AudioSegment, silence
import logging
import numpy as np
import noisereduce as nr
from dotenv import load_dotenv
import whisperx
import torch
from tqdm import tqdm
import json
import pandas as pd
from openai import OpenAI
import matplotlib.pyplot as plt
from scipy.io import wavfile
from jsonschema import validate, ValidationError
import time
import random
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken



# Ensure correct import for ffmpeg-python
try:
    import ffmpeg as ffmpeg_lib
except ModuleNotFoundError:
    logging.error("ffmpeg-python is not installed or not found in the current environment.")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(dotenv_path='/Users/pranay/Projects/LLM/video/proj1/scripts/.env')

# Set up API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Initialize OpenAI client
client = OpenAI()
SEED = 12345


# Detect the available device for WhisperX
device = "cpu"  # For macOS and MPS, we will run everything on the CPU
compute_type = "int8"  # Explicitly set to int8 for compatibility
logger.info(f"Running on {device} with {compute_type} compute type.")


# Load WhisperX Model
whisper_model = whisperx.load_model("large-v2", device, compute_type=compute_type)
# --- Schemas for Output Structure ---


# Prepare the system and user prompt
system_prompt = """
You are an advanced medical transcription analysis assistant. Your task is to analyze the provided medical transcription and return a structured output based on the predefined schema.
    
The analysis should be thorough and comprehensive, taking into account all nuances in the conversation, including non-verbal cues. If there is any important information or insights that fall outside the schema, dynamically add a section in the analysis labeled 'additional_analysis' with relevant details.
    
Ensure the final output adheres to the predefined schema but allows flexibility for additional insights and findings as needed.
"""
# Utility function to save JSON data
def save_json_to_file(data, filepath):
    try:
        # Validate if the data is JSON-serializable
        json.dumps(data)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Results saved as {filepath}")
    except (TypeError, ValueError) as e:
        logger.error(f"Data is not JSON-serializable: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error saving JSON file: {str(e)}")

# --- Step 1: Video Segmentation ---
def segment_large_video(video_path, segment_duration=1800):
    try:
        if not os.path.exists(video_path):
            logger.error(f"File not found: {video_path}")
            raise FileNotFoundError(f"File not found: {video_path}")

        logger.info(f"Processing video file: {video_path}")
        probe = ffmpeg_lib.probe(video_path)
        duration = float(probe['format']['duration'])

        segments = []
        logger.info(f"Segmenting video into {segment_duration} second segments...")
        for start in tqdm(range(0, int(duration), segment_duration), desc="Segmenting video"):
            end = min(start + segment_duration, duration)
            segment_folder = f"{os.path.splitext(video_path)[0]}_segment_{start}_{end}"
            os.makedirs(segment_folder, exist_ok=True)
            segment_path = os.path.join(segment_folder, f"segment_{start}_{end}.mp4")

            try:
                ffmpeg_lib.input(video_path, ss=start, t=segment_duration).output(segment_path, c='copy').run()
                segments.append(segment_path)
                logger.info(f"Segment saved: {segment_path}")
            except ffmpeg_lib.Error as e:
                logger.error(f"Error writing segment {start}-{end}: {str(e)}")
                raise e

        return segments

    except Exception as e:
        logger.error(f"Error during video segmentation: {str(e)}")
        raise e

# --- Step 2: Audio Extraction ---
def extract_audio_from_segment(video_path):
    logger.info(f"Extracting audio from {video_path}...")
    audio_output = "temp_audio.wav"
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_output)
    logger.info(f"Audio extracted and saved as {audio_output}")
    return audio_output

# --- Step 3: Remove Silence ---
def remove_silence(audio_path, silence_thresh=-40, min_silence_len=500, padding=300):
    logger.info(f"Removing silence from audio file: {audio_path}...")
    sound = AudioSegment.from_file(audio_path, format="wav")
    non_silent_segments = silence.detect_nonsilent(sound, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    
    cleaned_audio = AudioSegment.silent(duration=0)
    for start, end in non_silent_segments:
        cleaned_audio += sound[start-padding:end+padding]
    
    cleaned_audio_path = os.path.splitext(audio_path)[0] + "_cleaned.wav"
    cleaned_audio.export(cleaned_audio_path, format="wav")
    logger.info(f"Silence removed and audio saved as {cleaned_audio_path}")
    return cleaned_audio_path

# --- Step 4: Chunk Audio ---
def chunk_audio(cleaned_audio_path, chunk_length_ms=1800_000):  # 1800 seconds = 30 minutes
    logger.info(f"Chunking audio file: {cleaned_audio_path} into {chunk_length_ms // 1000 // 60} minute chunks...")
    sound = AudioSegment.from_file(cleaned_audio_path)
    chunks = [sound[i:i + chunk_length_ms] for i in range(0, len(sound), chunk_length_ms)]
    chunk_paths = []
    for i, chunk in enumerate(tqdm(chunks, desc="Chunking audio")):
        chunk_path = f"chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunk_paths.append(chunk_path)
        logger.info(f"Created chunk {i+1} with duration {chunk.duration_seconds} seconds")
    return chunk_paths

# --- Step 5: Normalize and Reduce Noise ---
def normalize_and_reduce_noise(chunk, noise_reduction_level=0.5):
    logger.info(f"Normalizing and reducing noise for audio chunk with noise_reduction_level={noise_reduction_level}...")
    try:
        target_dBFS = -20.0
        change_in_dBFS = target_dBFS - chunk.dBFS
        normalized_chunk = chunk.apply_gain(change_in_dBFS)

        samples = np.array(normalized_chunk.get_array_of_samples())
        reduced_noise_samples = nr.reduce_noise(y=samples, sr=normalized_chunk.frame_rate, prop_decrease=noise_reduction_level)
        reduced_noise_chunk = AudioSegment(
            reduced_noise_samples.tobytes(),
            frame_rate=normalized_chunk.frame_rate,
            sample_width=normalized_chunk.sample_width,
            channels=normalized_chunk.channels
        )
        logger.info(f"Noise reduced for chunk with level {noise_reduction_level}.")
        return reduced_noise_chunk
    
    except Exception as e:
        logger.error(f"Error during normalization or noise reduction: {str(e)}")
        return chunk

# --- Step 6: Save Cleaned Chunks ---
def save_cleaned_chunks(cleaned_chunks, segment_folder):
    logger.info(f"Saving cleaned audio chunks to {segment_folder}...")
    try:
        output_dir = os.path.join(segment_folder, "cleaned_chunks")
        os.makedirs(output_dir, exist_ok=True)

        chunk_paths = []
        for i, chunk in enumerate(cleaned_chunks):
            output_file = os.path.join(output_dir, f"cleaned_chunk_{i + 1}.wav")
            chunk.export(output_file, format="wav")
            logger.info(f"Cleaned chunk {i + 1} saved as {output_file}")
            chunk_paths.append(output_file)  # Save the file path

        return chunk_paths  # Return the paths of saved files
    except Exception as e:
        logger.error(f"Error saving cleaned chunks: {str(e)}")
        raise e

# Function to visualize the waveform of the audio
def visualize_audio_waveform(audio_file_path, save_path=None):
    logger.info(f"Visualizing audio waveform for: {audio_file_path}")
    
    # Read the audio file
    sample_rate, audio_data = wavfile.read(audio_file_path)
    
    # Check if the audio is stereo or mono
    if len(audio_data.shape) == 2:
        # Stereo: Take the mean of both channels
        audio_data = np.mean(audio_data, axis=1)
    
    # Generate time axis in seconds
    time_axis = np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data))
    
    # Plot the waveform
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, audio_data, color='blue')
    plt.title(f"Waveform of {os.path.basename(audio_file_path)}")
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Waveform saved as {save_path}")
    else:
        plt.show()

    plt.close()

# --- Step 7: WhisperX Transcription ---
def transcribe_with_whisperx(audio_file_path, whisper_model, batch_size=16):
    logger.info(f"Transcribing audio chunk: {audio_file_path} using WhisperX...")
    
    try:
        # Load audio
        audio = whisperx.load_audio(audio_file_path)
        
        # Transcribe with WhisperX
        result = whisper_model.transcribe(audio, batch_size=batch_size)
        
        # Print the raw segments before alignment
        logger.info(f"Initial transcription result for {audio_file_path}: {result['segments']}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        raise e

# --- Step 8: WhisperX Alignment ---
def align_with_whisperx(result, audio_file_path, whisper_model, device="cpu"):
    logger.info(f"Aligning transcription for {audio_file_path}...")
    
    try:
        # Load alignment model
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        
        # Align results
        aligned_result = whisperx.align(result["segments"], model_a, metadata, audio_file_path, device)
        
        logger.info(f"Alignment completed for {audio_file_path}: {aligned_result}.")
        return aligned_result
    
    except Exception as e:
        logger.error(f"Error during alignment: {str(e)}")
        raise e

# --- Step 9: WhisperX Diarization ---
def diarize_with_whisperx(audio_file_path, hf_token, retries=3):
    logger.info(f"Performing speaker diarization for {audio_file_path}...")
    try:
        diarization_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
        
        # Retry mechanism
        for attempt in range(retries):
            try:
                diarization_result = diarization_model(audio_file_path)
                logger.info(f"Diarization completed for {audio_file_path}.")
                if isinstance(diarization_result, pd.DataFrame):
                    diarization_segments = diarization_result.to_dict('records')
                else:
                    diarization_segments = []
                return diarization_segments
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}: Error during diarization: {str(e)}")
                if attempt == retries - 1:
                    logger.error("Max retries reached for diarization.")
                    return None
    except Exception as e:
        logger.error(f"Error during diarization: {str(e)}")
        return None

# --- Step 10: WhisperX Speaker Assignment ---
def assign_speakers_to_transcription(diarization_segments, aligned_transcription):
    logger.info(f"Assigning speaker labels to transcription segments...")

    try:
        # Convert diarization_segments to DataFrame if it's not already
        if not isinstance(diarization_segments, pd.DataFrame):
            diarization_df = pd.DataFrame(diarization_segments)
        else:
            diarization_df = diarization_segments

        # Ensure the aligned_transcription contains 'segments' key
        if "segments" not in aligned_transcription:
            raise KeyError("Aligned transcription does not contain 'segments' key.")
        
        logger.info(f"Matching diarization segments with transcription...")

        # Use WhisperX's assign_word_speakers function for speaker assignment
        final_result = whisperx.assign_word_speakers(diarization_df, aligned_transcription)
        
        # Ensure that the final result contains the required fields
        for segment in final_result['segments']:
            if 'speaker' not in segment:
                logger.warning(f"No speaker label found for segment starting at {segment['start']}")

        logger.info(f"Speaker labels assigned successfully.")
        return final_result

    except KeyError as e:
        logger.error(f"Missing key during speaker assignment: {str(e)}")
        return aligned_transcription  # Return without speaker assignment if error occurs
    except Exception as e:
        logger.error(f"Error during speaker assignment: {str(e)}")
        return aligned_transcription  # Fallback to unmodified transcription if error occurs


    # return partial_response, None
def load_transcription_file(transcription_file_path):
    """Load transcription data from the file."""
    if os.path.exists(transcription_file_path):
        with open(transcription_file_path, 'r') as f:
            return json.load(f)
    else:
        logger.error(f"Transcription file not found: {transcription_file_path}")
        return None

def analyze_transcription_basic(transcription_chunk):
    """Send transcription result to OpenAI API and get a response."""
    try:
        # Send transcription result with a basic system and user prompt
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",  # You can update to any available model
            messages=[
                {"role": "system", "content": "You are a medical transcription analysis assistant. Your task is to analyze the provided transcription result and return insights, summaries, or important details from the conversation."},
                {"role": "user", "content": transcription_chunk}
            ],
            temperature=0.3,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        
        # Extract the response content
        raw_response = response.choices[0].message.content
        return raw_response

    except Exception as e:
        logger.error(f"Error during OpenAI API call: {str(e)}")
        return None

def main():
    # Path to your transcription file (you can modify it based on your file structure)
    transcription_file_path = "/Users/pranay/Projects/LLM/video/proj1/data/Chiranjeevi_Video_Dec_21_segment_0_1401.583333/segment_0_1401.583333/transcription_chunk_cleaned_chunk_1.wav.json"

    # Load the transcription data
    transcription_data = load_transcription_file(transcription_file_path)

    if transcription_data:
        # Assuming the transcription chunk is part of the loaded JSON data
        transcription_chunk = json.dumps(transcription_data)  # Convert dict to JSON string if necessary

        # Analyze transcription and get the result
        analysis_result = analyze_transcription_basic(transcription_chunk)

        if analysis_result:
            # Output the result
            print("Analysis Result:")
            print(analysis_result)
            
            # Save result to a file (optional)
            output_file_path = os.path.join(os.path.dirname(transcription_file_path), "final_analysis_cleaned_chunk_1.json")
            with open(output_file_path, 'w') as outfile:
                json.dump({"analysis": analysis_result}, outfile)
            logger.info(f"Results saved as {output_file_path}")

if __name__ == "__main__":
    main()