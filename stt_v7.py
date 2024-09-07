import os
from moviepy.editor import VideoFileClip
from pydub import AudioSegment, silence
import logging
import numpy as np
import noisereduce as nr
import matplotlib.pyplot as plt
import google.generativeai as genai
import json
from dotenv import load_dotenv
from datetime import datetime
from io import BytesIO
import tempfile

# Ensure correct import for ffmpeg-python
try:
    import ffmpeg as ffmpeg_lib
except ModuleNotFoundError:
    logging.error("ffmpeg-python is not installed or not found in the current environment.")



# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Load environment variables and configure Gemini
load_dotenv(dotenv_path='/Users/pranay/Projects/LLM/video/proj1/scripts/.env')
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")
genai.configure(api_key=api_key)

# --- System Prompt ---
system_prompt = """
You are an AI assistant specialized in transcribing medical consultations from audio/video content. Your task is to provide an accurate, verbatim transcription of the conversation, including speaker identification and timestamps.
Key Responsibilities:

Transcribe all speech content accurately.
Identify speakers (e.g., Doctor, Patient, Nurse).
Include timestamps at regular intervals or when the speaker changes.
Note any significant non-verbal sounds (e.g., coughing, long pauses, background noises) that may be relevant to understanding the context.

Do not provide any analysis, summaries, or additional information beyond the transcription itself.
"""
# --- User Prompt ---
user_prompt = """
Please transcribe the following medical consultation recording:
<video_content>
{{VIDEO_CONTENT}}
</video_content>
Provide your transcription in the following JSON format:
{
  "transcription": [
    {
      "timestamp": "MM:SS",
      "speaker": "SPEAKER_ROLE",
      "text": "SPOKEN_TEXT",
      "non_verbal": "NON_VERBAL_SOUNDS (if any)"
    },
    ...
  ]
}

Ensure your transcription is thorough and captures all spoken content. If there are parts of the audio that are unclear or inaudible, indicate this in the transcription.
"""


def segment_large_video(video_path, segment_duration=1800):
    """
    Segments a large video into smaller clips of specified duration using ffmpeg.
    """
    try:
        if not os.path.exists(video_path):
            logger.error(f"File not found: {video_path}")
            raise FileNotFoundError(f"File not found: {video_path}")

        logger.info(f"Processing video file: {video_path}")

        # Get video information
        probe = ffmpeg_lib.probe(video_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        duration = float(probe['format']['duration'])

        segments = []
        for start in range(0, int(duration), segment_duration):
            end = min(start + segment_duration, duration)
            segment_folder = f"{os.path.splitext(video_path)[0]}_segment_{start}_{end}"
            os.makedirs(segment_folder, exist_ok=True)
            segment_path = os.path.join(segment_folder, f"segment_{start}_{end}.mp4")
            
            try:
                (
                    ffmpeg_lib
                    .input(video_path, ss=start, t=segment_duration)
                    .output(segment_path, c='copy')
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )
                segments.append(segment_path)
                logger.info(f"Segment saved: {segment_path}")
            except ffmpeg_lib.Error as e:
                logger.error(f"Error writing segment {start}-{end}: {str(e)}")
                raise e

        return segments

    except Exception as e:
        logger.error(f"Error during video segmentation: {str(e)}")
        raise e

def extract_and_chunk_audio(file_path, chunk_length_ms=30 * 60 * 1000):
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.endswith(('.mp4', '.mkv', '.avi')):
            logger.info(f"Processing video file: {file_path}")
            clip = VideoFileClip(file_path)
            audio_clip = clip.audio
            temp_audio_path = "temp_audio.wav"
            audio_clip.write_audiofile(temp_audio_path)
            audio = AudioSegment.from_wav(temp_audio_path)
            logger.info(f"Extracted audio from video: {file_path}")
        else:
            logger.info(f"Processing audio file: {file_path}")
            audio = AudioSegment.from_file(file_path)

        logger.info(f"Total audio duration: {len(audio) / 1000} seconds")

        chunks = []
        for i in range(0, len(audio), chunk_length_ms):
            chunk = audio[i:i + chunk_length_ms]
            if len(chunk) > 0:  # Only add non-empty chunks
                chunks.append(chunk)
                logger.info(f"Created chunk {len(chunks)} with duration {len(chunk) / 1000} seconds")

        logger.info(f"Chunked audio into {len(chunks)} pieces, each up to {chunk_length_ms / 1000} seconds long.")
        return chunks

    except Exception as e:
        logger.error(f"Error during audio extraction and chunking: {str(e)}")
        raise e

def detect_silence_in_chunk(chunk, silence_thresh=-50.0, min_silence_len=1000):
    """
    Detects if a chunk is mostly silent.
    """
    try:
        silence_chunks = silence.detect_silence(chunk, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
        total_silence_duration = sum([end - start for start, end in silence_chunks])
        return total_silence_duration > (len(chunk) * 0.7)  # If more than 70% of the chunk is silent, consider it silent
    except Exception as e:
        logger.error(f"Error detecting silence: {str(e)}")
        raise e

def normalize_audio(chunk):
    """
    Normalize the audio chunk to a target dBFS level.
    """
    try:
        target_dBFS = -20.0
        change_in_dBFS = target_dBFS - chunk.dBFS
        return chunk.apply_gain(change_in_dBFS)
    except Exception as e:
        logger.error(f"Error normalizing audio: {str(e)}")
        raise e

def reduce_noise(chunk):
    """
    Apply noise reduction to an audio chunk.
    """
    try:
        samples = np.array(chunk.get_array_of_samples())
        reduced_noise_samples = nr.reduce_noise(y=samples, sr=chunk.frame_rate)
        reduced_noise_chunk = AudioSegment(
            reduced_noise_samples.tobytes(),
            frame_rate=chunk.frame_rate,
            sample_width=chunk.sample_width,
            channels=chunk.channels
        )
        return reduced_noise_chunk
    except ValueError as e:
        logger.warning(f"Noise reduction issue: {e}. Skipping chunk.")
        return chunk
    except Exception as e:
        logger.error(f"Error during noise reduction: {str(e)}")
        raise e

def save_cleaned_chunks(cleaned_chunks, segment_folder, indices=None):
    """
    Save specified or all cleaned audio chunks to the specified segment folder.
    """
    try:
        output_dir = os.path.join(segment_folder, "cleaned_chunks")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if indices is None:
            indices = range(1, len(cleaned_chunks) + 1)

        for output_index, i in enumerate(indices):
            if 1 <= i <= len(cleaned_chunks):
                output_file = os.path.join(output_dir, f"cleaned_chunk_{output_index + 1}.wav")
                cleaned_chunks[i-1].export(output_file, format="wav")
                logger.info(f"Cleaned Chunk {output_index + 1} saved as {output_file}")
            else:
                logger.warning(f"Index {i} is out of range. No chunk saved for this index.")
    except Exception as e:
        logger.error(f"Error saving cleaned chunks: {str(e)}")
        raise e

def clean_audio_chunks(chunks, segment_folder):
    """
    Clean audio chunks by checking for silence, normalizing, and applying noise reduction.
    Save each chunk as it's processed.
    """
    cleaned_chunks = []
    for i, chunk in enumerate(chunks):
        try:
            if detect_silence_in_chunk(chunk):
                logger.info(f"Chunk {i+1} is mostly silent. Skipping.")
                continue
            
            # Check if chunk is less than 1 second
            if len(chunk) < 1000:  # Length is in milliseconds
                logger.info(f"Chunk {i+1} is less than 1 second long. Skipping.")
                continue
            chunk = normalize_audio(chunk)
            chunk = reduce_noise(chunk)
            
            # Save the cleaned chunk immediately
            output_dir = os.path.join(segment_folder, "cleaned_chunks")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"cleaned_chunk_{i+1}.wav")
            chunk.export(output_file, format="wav")
            
            cleaned_chunks.append(chunk)
            logger.info(f"Chunk {i+1} cleaned and saved as {output_file}")
        except Exception as e:
            logger.error(f"Error cleaning chunk {i+1}: {str(e)}")
            continue
    return cleaned_chunks

def visualize_audio_chunks(chunks, indices=None, title_prefix=""):
    """
    Visualize waveforms of the specified or all audio chunks.
    """
    try:
        if indices is None:
            indices = range(1, len(chunks) + 1)

        for i in indices:
            if 1 <= i <= len(chunks):
                chunk = chunks[i - 1]
                samples = np.array(chunk.get_array_of_samples())
                
                plt.figure(figsize=(10, 4))
                plt.plot(samples)
                plt.title(f"{title_prefix} Waveform of Chunk {i} (Duration: {len(chunk) / 1000:.2f} seconds)")
                plt.xlabel("Sample Number")
                plt.ylabel("Amplitude")
                plt.show()
            else:
                logger.warning(f"Index {i} is out of range. No visualization available for this index.")
    except Exception as e:
        logger.error(f"Error visualizing audio chunks: {str(e)}")
        raise e
    

def transcribe_chunk_with_gemini(model, cleaned_chunk, user_prompt):
    """Transcribe an audio chunk using Gemini."""
    try:
        # Create a temporary file to store the audio chunk
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            cleaned_chunk.export(temp_audio_file.name, format="wav")
            temp_audio_path = temp_audio_file.name

        # Upload the temporary audio file to Gemini
        uploaded_file = genai.upload_file(path=temp_audio_path)

        # Generate content with the uploaded file using appropriate prompts
        response = model.generate_content(
            [user_prompt, uploaded_file]
        )

        # Return the raw API response
        return response.text
    
    except Exception as e:
        logger.error(f"Error transcribing audio chunk: {str(e)}")
        return None
    
    # finally:
    #     # Clean up the temporary file
    #     os.remove(temp_audio_path)


def process_chunks_with_gemini(model, cleaned_chunks, user_prompt):
    raw_outputs = []
    
    for chunk in cleaned_chunks:
        raw_output = transcribe_chunk_with_gemini(model, chunk, user_prompt)
        if raw_output:
            raw_outputs.append(raw_output)
    
    return raw_outputs

def extract_json_parts(json_data, parent_key=''):
    """
    Recursively extracts all parts from inside a JSON structure and returns them,
    maintaining array structures with numerical indices.
    
    Args:
        json_data (dict or list): The JSON data structure to parse.
        parent_key (str): The base string for nested keys.
        
    Returns:
        dict: A dictionary with all keys and their corresponding values.
    """
    parts = {}

    if isinstance(json_data, dict):
        for key, value in json_data.items():
            full_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, (dict, list)):
                parts.update(extract_json_parts(value, full_key))
            else:
                parts[full_key] = value

    elif isinstance(json_data, list):
        for index, item in enumerate(json_data):
            full_key = f"{parent_key}[{index}]"
            if isinstance(item, str) and item.strip().startswith("```json"):
                try:
                    json_str = item.strip().removeprefix("```json").removesuffix("```").strip()
                    parsed_json = json.loads(json_str)
                    parts.update(extract_json_parts(parsed_json, full_key))
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing JSON from string at {full_key}: {str(e)}")
                    parts[full_key] = item  # Fall back to storing the raw string
            elif isinstance(item, (dict, list)):
                parts.update(extract_json_parts(item, full_key))
            else:
                parts[full_key] = item

    return parts

def save_transcription_output(raw_outputs, segment_folder):
    try:
        # Save the raw output
        raw_output_path = os.path.join(segment_folder, "raw_transcription_output.json")
        with open(raw_output_path, 'w') as f:
            json.dump(raw_outputs, f, indent=4)
        logger.info(f"Raw transcription output saved as {raw_output_path}")

        # Create a formatted JSON structure
        formatted_output = {
            "transcriptions": raw_outputs,
            "metadata": {
                "total_chunks": len(raw_outputs),
                "generated_on": str(datetime.now())
            }
        }

        formatted_output_path = os.path.join(segment_folder, "formatted_transcription_output.json")
        with open(formatted_output_path, 'w') as f:
            json.dump(formatted_output, f, indent=4)
        logger.info(f"Formatted transcription output saved as {formatted_output_path}")

        # Extract data from the raw output and save it as extracted_data.json
        extracted_data = extract_json_parts(raw_outputs)
        extracted_data_path = os.path.join(segment_folder, "extracted_data.json")
        with open(extracted_data_path, 'w') as f:
            json.dump(extracted_data, f, indent=4)
        logger.info(f"Extracted data saved as {extracted_data_path}")
        
    except Exception as e:
        logger.error(f"Error saving transcription output: {str(e)}")
        raise e
# Initialize the Gemini model
model = genai.GenerativeModel(
    model_name='gemini-1.5-pro-exp-0827',
    tools=[],
    system_instruction=system_prompt
)
# chat = model.start_chat(enable_automatic_function_calling=True)

# Segment the large video file into smaller videos
video_file_path = '/Users/pranay/Projects/LLM/video/proj1/data/Chiranjeevi_Video_Dec_21.mp4'
segments = segment_large_video(video_file_path)

# Process each video segment individually and save both raw, formatted transcription outputs, and extracted data
for segment_file_path in segments:
    segment_folder = os.path.splitext(segment_file_path)[0]
    os.makedirs(segment_folder, exist_ok=True)
    
    # Extract and chunk audio from each video segment
    chunks = extract_and_chunk_audio(segment_file_path)

    plt.ion()  # Ensure interactive mode is on for displaying plots in VS Code
    
    # Visualize all original chunks
    visualize_audio_chunks(chunks, title_prefix="Original")

    # Clean the chunks and save them immediately
    cleaned_chunks = clean_audio_chunks(chunks, segment_folder)
    
    # Visualize all cleaned chunks
    visualize_audio_chunks(cleaned_chunks, title_prefix="Cleaned")
    
    # Process transcriptions for each cleaned chunk
    raw_outputs = process_chunks_with_gemini(model, cleaned_chunks, user_prompt)
    
    # Save both raw and formatted transcription outputs, then extract and save extracted data
    save_transcription_output(raw_outputs, segment_folder)