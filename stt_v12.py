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
import pandas as pd  # Ensure this is imported for DataFrame handling

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

# Detect the available device for WhisperX
device = "cpu"  # For macOS and MPS, we will run everything on the CPU
compute_type = "int8"  # Explicitly set to int8 for compatibility
logger.info(f"Running on {device} with {compute_type} compute type.")

# Load WhisperX Model
whisper_model = whisperx.load_model("large-v2", device, compute_type=compute_type)

# Utility function to save JSON data
def save_json_to_file(data, filepath, description):
    try:
        # Convert DataFrame to a list of dicts if necessary
        if isinstance(data, pd.DataFrame):
            data = data.to_dict(orient='records')
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"{description} saved as {filepath}")
    except Exception as e:
        logger.error(f"Error saving {description}: {str(e)}")

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
def normalize_and_reduce_noise(chunk):
    logger.info(f"Normalizing and reducing noise for audio chunk...")
    try:
        target_dBFS = -20.0
        change_in_dBFS = target_dBFS - chunk.dBFS
        normalized_chunk = chunk.apply_gain(change_in_dBFS)

        samples = np.array(normalized_chunk.get_array_of_samples())
        reduced_noise_samples = nr.reduce_noise(y=samples, sr=normalized_chunk.frame_rate)
        reduced_noise_chunk = AudioSegment(
            reduced_noise_samples.tobytes(),
            frame_rate=normalized_chunk.frame_rate,
            sample_width=normalized_chunk.sample_width,
            channels=normalized_chunk.channels
        )
        logger.info(f"Noise reduced for chunk.")
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

# --- Step 7: WhisperX Transcription ---
def transcribe_with_whisperx(audio_file_path, whisper_model, batch_size=16):
    logger.info(f"Transcribing audio chunk: {audio_file_path} using WhisperX...")
    
    # Load audio
    audio = whisperx.load_audio(audio_file_path)
    
    # Transcribe with WhisperX
    result = whisper_model.transcribe(audio, batch_size=batch_size)
    
    # Print the raw segments before alignment
    logger.info(f"Initial transcription result for {audio_file_path}: {result['segments']}")
    
    # Save the raw transcription result immediately
    transcription_file = os.path.splitext(audio_file_path)[0] + "_raw_transcription.json"
    save_json_to_file(result, transcription_file, "Raw transcription result")

    return result

# --- Step 8: WhisperX Alignment ---
def align_with_whisperx(result, audio_file_path, whisper_model):
    logger.info(f"Aligning transcription for {audio_file_path}...")
    
    # Load alignment model
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    
    # Align results
    aligned_result = whisperx.align(result["segments"], model_a, metadata, audio_file_path, device)
    
    logger.info(f"Alignment completed for {audio_file_path}.")
    # Save the aligned transcription result
    alignment_file = os.path.splitext(audio_file_path)[0] + "_aligned_transcription.json"
    save_json_to_file(aligned_result, alignment_file, "Aligned transcription result")

    return aligned_result   

# --- Step 9: WhisperX Diarization ---
def diarize_with_whisperx(audio_file_path, hf_token):
    logger.info(f"Performing speaker diarization for {audio_file_path}...")
    try:
        diarization_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
        
        # Perform diarization
        diarization_segments = diarization_model(audio_file_path)
        
        logger.info(f"Diarization completed for {audio_file_path}.")
        
        # Convert diarization segments to a JSON-serializable format
        diarization_result = [
            {
                "start": segment['start'],
                "end": segment['end'],
                "speaker": segment['speaker']
            }
            for segment in diarization_segments if isinstance(segment, dict)
        ]
        
        logger.info(f"Diarization segments converted for {audio_file_path}.")
        return diarization_result
    
    except Exception as e:
        logger.error(f"Error during diarization: {str(e)}")
        return None



# --- Step 10: WhisperX Speaker Assignment ---
def assign_speakers_to_transcription(diarization_segments, aligned_transcription):
    """
    Assign speaker labels to the transcription segments.
    """
    logger.info(f"Assigning speaker labels to transcription segments...")
    try:
        result_with_speakers = whisperx.assign_word_speakers(diarization_segments, aligned_transcription)
        logger.info(f"Speaker labels assigned.")
        return result_with_speakers
    except Exception as e:
        logger.error(f"Error during speaker assignment: {str(e)}")
        return aligned_transcription

# --- Main Processing Flow ---
def main(video_file_path, hf_token):
    logger.info(f"Starting processing for video: {video_file_path}")
    
    # Step 1: Segment the video
    segments = segment_large_video(video_file_path)

    for segment_file_path in segments:
        segment_folder = os.path.splitext(segment_file_path)[0]
        os.makedirs(segment_folder, exist_ok=True)

        # Step 2: Extract audio
        audio_file_path = extract_audio_from_segment(segment_file_path)

        # Step 3: Remove silence
        cleaned_audio_path = remove_silence(audio_file_path)

        # Step 4: Chunk audio
        chunks = chunk_audio(cleaned_audio_path)

        # Step 5: Normalize and reduce noise in chunks
        cleaned_chunks = [normalize_and_reduce_noise(AudioSegment.from_file(chunk)) for chunk in chunks]

        # Step 6: Save cleaned chunks and get their paths
        cleaned_chunk_paths = save_cleaned_chunks(cleaned_chunks, segment_folder)

        # Step 7: Transcribe using WhisperX
        for cleaned_chunk_path in tqdm(cleaned_chunk_paths, desc="Processing cleaned chunks"):
            transcription_result = transcribe_with_whisperx(cleaned_chunk_path, whisper_model)

            # Save the raw transcription result to a file
            transcription_file = os.path.join(segment_folder, f"transcription_chunk_{os.path.basename(cleaned_chunk_path)}.json")
            save_json_to_file(transcription_result, transcription_file, "Raw transcription result")
            
            # Step 8: Align transcriptions with word-level timestamps
            aligned_transcription = align_with_whisperx(transcription_result, cleaned_chunk_path, whisper_model)

            # Save the aligned transcription result to a file
            alignment_file = os.path.join(segment_folder, f"aligned_transcription_chunk_{os.path.basename(cleaned_chunk_path)}.json")
            save_json_to_file(aligned_transcription, alignment_file, "Aligned transcription result")

            # Step 9: Perform speaker diarization
            diarization_segments = diarize_with_whisperx(cleaned_chunk_path, hf_token)

            # Save the diarization result to a file
            diarization_file = os.path.join(segment_folder, f"diarization_chunk_{os.path.basename(cleaned_chunk_path)}.json")
            save_json_to_file(diarization_segments, diarization_file, "Diarization result")

            # Step 10: Assign speaker labels to transcription
            final_result = assign_speakers_to_transcription(diarization_segments, aligned_transcription)

            # Save the final transcription with speaker labels
            final_result_file = os.path.join(segment_folder, f"final_transcription_with_speakers_chunk_{os.path.basename(cleaned_chunk_path)}.json")
            save_json_to_file(final_result, final_result_file, "Final transcription with speakers")

if __name__ == "__main__":
    video_file_path = '/Users/pranay/Projects/LLM/video/proj1/data/Chiranjeevi_Video_Dec_21.mp4'
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise ValueError("Hugging Face token not found. Please set it in the environment.")
    print("Using Hugging Face token:", hf_token)
    main(video_file_path, hf_token)