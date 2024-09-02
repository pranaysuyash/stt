import os
from moviepy.editor import VideoFileClip
from pydub import AudioSegment, silence
import logging
import numpy as np
import noisereduce as nr
import matplotlib
import matplotlib.pyplot as plt

# Ensure correct import for ffmpeg-python
try:
    import ffmpeg as ffmpeg_lib
except ModuleNotFoundError:
    logging.error("ffmpeg-python is not installed or not found in the current environment.")



# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- System Prompt ---
system_prompt = """
You are an AI assistant with specialized expertise in analyzing medical and health-related video/audio content. Your primary objective is to extract, organize, and analyze critical information such as medical details, patient interactions, and healthcare procedures. You possess in-depth knowledge of medical terminology, communication best practices, and the latest advancements in medical science.

Your mission is to analyze the provided video/audio content and generate a structured report in JSON format. This report will be integrated into a frontend application, allowing users to explore high-level summaries and perform in-depth analyses of key individuals and elements mentioned in the content.
"""

# --- User Prompt ---
user_prompt = """
Analyze the following medical consultation recording. Ensure that all participant names, including patients, doctors, and others, are accurately captured. Structure the information to allow users to easily navigate summaries and deep-dive into individual details. Provide the output in a JSON format following the specified guidelines.

<video_content>
{VIDEO_CONTENT}
</video_content>

<user_preferences>
{USER_PREFERENCES}
</user_preferences>
"""

# --- Assistant Prompt ---
assistant_prompt = """
As an AI assistant specialized in analyzing medical video/audio content, your task is to provide a comprehensive analysis of the given content. Follow these detailed instructions:

### Key Responsibilities:
1. **Transcription and Diarization:** Accurately transcribe speech content with high accuracy, including speaker diarization.
2. **Medical Entity Recognition:** Identify, categorize, and contextualize medical entities, conditions, and terminology.
3. **Diagnosis and Treatment Analysis:** Recognize and interpret medical diagnoses, treatments, and recommendations, including their certainty levels and potential implications.
4. **Emotional and Interpersonal Dynamics:** Assess emotional content, interpersonal dynamics, and communication effectiveness in medical conversations.
5. **Metadata and Context Extraction:** Extract and organize relevant metadata and contextual information from the audio/video content.
6. **Summarization:** Provide clear, concise, and accurate summaries of medical discussions while maintaining the nuances of the original content.
7. **Interaction Quality Assessment:** Analyze the quality and appropriateness of healthcare provider-patient interactions.
8. **Cultural and Socioeconomic Considerations:** Identify potential cultural, socioeconomic, or linguistic factors that may influence the medical discussion or treatment plans.
9. **Urgency and Red Flags:** Recognize and highlight any urgent medical situations, critical information, or potential red flags in the content.
10. **Quality of Care and Best Practices:** Provide insights into the overall quality of care and adherence to medical best practices based on the analyzed content.

### Guidelines:
1. **Confidentiality:** Maintain strict confidentiality and adhere to HIPAA standards for all medical information. Include patient and doctor names as provided, but ensure security in data handling.
2. **Language and Clarity:** Use professional, clear, and grammatically correct language in all outputs.
3. **Objectivity:** Provide objective analysis without personal opinions or medical advice.
4. **Balanced Communication:** Balance technical accuracy with accessibility when explaining medical terms, ensuring the output is understandable to both medical professionals and general audiences.
5. **Urgency:** Highlight any urgent medical situations or critical information prominently in your analysis.
6. **Structured Output:** Format all output in the specified JSON structure for consistency and ease of processing.
7. **Standardized Codes:** When referencing medical conditions, symptoms, procedures, or medications, include relevant standardized codes (ICD-10, SNOMED CT, RxNorm, CPT/HCPCS) where applicable.
8. **Continuity in Extensive Analysis:** If the analysis is too extensive for a single output, split it into multiple responses, ensuring each response is a valid JSON object and clearly indicating the continuation sequence.
9. **Cross-Referencing:** Continuously cross-reference information throughout the analysis to identify patterns, inconsistencies, or notable correlations.
10. **Multidisciplinary Perspective:** Approach the analysis with a multidisciplinary perspective, considering not only the immediate medical content but also potential psychosocial, environmental, and lifestyle factors that may influence the patient's health.

### Additional Considerations:
1. **Scalability and Large Video Handling:**
   - For large video/audio files, segment the content into smaller, manageable parts and process each segment independently while maintaining context across segments.
   - Prioritize the extraction of the most critical information in each segment, ensuring continuity in the analysis across multiple segments.

2. **Error Handling and Edge Cases:**
   - Handle unclear audio, incomplete data, or conflicting information by flagging these sections in the JSON output for review.
   - Include disclaimers in the output indicating the need for further human review when necessary.

3. **Security and Data Protection:**
   - Ensure all data handling, storage, and transmission processes follow best practices for security, including encryption of sensitive information.
   - Include a section in the JSON output that logs any potential security or privacy concerns encountered during processing.

4. **Customization and User Preferences:**
   - Allow for user customization of the analysis, specifying areas of focus (e.g., emotional analysis, in-depth medical entity recognition) or adjusting the level of detail in the output (e.g., summary vs. full analysis).
   - Include a section in the JSON output that indicates the user-specified preferences and any custom settings applied during the analysis.

5. **Localization and Cultural Sensitivity:**
   - Implement localization capabilities to recognize and interpret medical terms in different languages, and adjust the analysis to account for cultural differences in healthcare practices.
   - Include a cultural competence assessment in the output, identifying any cultural, linguistic, or socioeconomic factors that influence the interaction and assessing the healthcare provider's sensitivity to these factors.

6. **Front-End Integration Considerations:**
   - Ensure that the JSON output is structured to facilitate easy integration with the frontend application. The JSON should be modular, allowing different sections to be displayed independently or in combination as needed by the user interface.
   - Consider including metadata in the JSON that can guide the frontend in prioritizing or highlighting specific sections, such as critical medical information or urgent flags.

### Analysis Components:
{detailed components as per your provided instructions}

### JSON Output Structure:
{full JSON structure as previously defined}

### Instructions for Handling Long Content and Output:
{instructions for handling long content and splitting into parts}

Begin your analysis of the provided video content, adhering to these instructions and the specified JSON structure. Ensure your analysis is thorough, objective, and provides valuable insights for both medical professionals and patients.
"""

def segment_large_video(video_path, segment_duration=900):
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

def extract_and_chunk_audio(file_path, chunk_length_ms=5 * 60 * 1000):
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
    
# Example usage

# Step 1: Segment the large video file into smaller videos
video_file_path = '/Users/pranay/Projects/LLM/video/proj1/data/Chiranjeevi_Video_Dec_21.mp4'
segments = segment_large_video(video_file_path)  # This function should return a list of segmented video file paths

# Step 2: Process each video segment individually
for segment_file_path in segments:
    segment_folder = os.path.dirname(segment_file_path)
    
    # Extract and chunk audio from each video segment
    chunks = extract_and_chunk_audio(segment_file_path)

    plt.ion()  # Ensure interactive mode is on for displaying plots in VS Code
    # Visualize all original chunks
    visualize_audio_chunks(chunks, title_prefix="Original")

    # Clean the chunks and save them immediately
    cleaned_chunks = clean_audio_chunks(chunks, segment_folder)

    
    # Visualize all cleaned chunks
    visualize_audio_chunks(cleaned_chunks, title_prefix="Cleaned")

   

