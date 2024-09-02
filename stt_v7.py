import os
from moviepy.editor import VideoFileClip
from pydub import AudioSegment, silence
import logging
import numpy as np
import noisereduce as nr
import matplotlib
import matplotlib.pyplot as plt


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

def extract_and_chunk_audio(file_path, chunk_length_ms=5 * 60 * 1000):
    """
    Extracts audio from a video file and chunks it into pieces of specified length.

    :param file_path: Path to the video file.
    :param chunk_length_ms: Length of each chunk in milliseconds. Default is 5 minutes.
    :return: List of AudioSegment objects, each representing a chunk of audio.
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        # Extract audio from video
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

        # Chunk the audio into 5-minute segments
        chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
        logger.info(f"Chunked audio into {len(chunks)} pieces, each {chunk_length_ms / 60000} minutes long.")
        
        return chunks

    except Exception as e:
        logger.error(f"Error during audio extraction and chunking: {str(e)}")
        raise e

def detect_silence_in_chunk(chunk, silence_thresh=-50.0, min_silence_len=1000):
    """
    Detects if a chunk is mostly silent.

    :param chunk: AudioSegment object
    :param silence_thresh: Silence threshold in dBFS
    :param min_silence_len: Minimum length of silence in milliseconds to consider it silent
    :return: Boolean indicating if the chunk is mostly silent
    """
    silence_chunks = silence.detect_silence(chunk, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    total_silence_duration = sum([end - start for start, end in silence_chunks])
    return total_silence_duration > (len(chunk) * 0.7)  # If more than 70% of the chunk is silent, consider it silent

def normalize_audio(chunk):
    """
    Normalize the audio chunk to a target dBFS level.

    :param chunk: AudioSegment object
    :return: Normalized AudioSegment object
    """
    target_dBFS = -20.0  # Target level
    change_in_dBFS = target_dBFS - chunk.dBFS
    return chunk.apply_gain(change_in_dBFS)

def reduce_noise(chunk):
    """
    Apply noise reduction to an audio chunk.

    :param chunk: AudioSegment object
    :return: Noise-reduced AudioSegment object
    """
    samples = np.array(chunk.get_array_of_samples())
    try:
        reduced_noise_samples = nr.reduce_noise(y=samples, sr=chunk.frame_rate)
    except ValueError as e:
        logger.warning(f"Noise reduction issue: {e}. Skipping chunk.")
        return chunk

    reduced_noise_chunk = AudioSegment(
        reduced_noise_samples.tobytes(),
        frame_rate=chunk.frame_rate,
        sample_width=chunk.sample_width,
        channels=chunk.channels
    )
    
    return reduced_noise_chunk

def save_cleaned_chunks(cleaned_chunks, indices=None, output_dir="cleaned_chunks"):
    """
    Save specified or all cleaned audio chunks to the specified directory.

    :param cleaned_chunks: List of cleaned AudioSegment objects
    :param indices: List of indices specifying which chunks to save (1-based). If None, all chunks will be saved.
    :param output_dir: Directory where the cleaned chunks will be saved. Default is "cleaned_chunks".
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # If no indices are specified, save all chunks
    if indices is None:
        indices = range(1, len(cleaned_chunks) + 1)

    for output_index, i in enumerate(indices):
        if 1 <= i <= len(cleaned_chunks):
            output_file = os.path.join(output_dir, f"cleaned_chunk_{output_index + 1}.wav")
            cleaned_chunks[i-1].export(output_file, format="wav")
            logger.info(f"Cleaned Chunk {output_index + 1} saved as {output_file}")
        else:
            logger.warning(f"Index {i} is out of range. No chunk saved for this index.")


def clean_audio_chunks(chunks):
    """
    Clean audio chunks by checking for silence, normalizing, and applying noise reduction.
    
    :param chunks: List of AudioSegment objects
    :return: List of cleaned AudioSegment objects
    """
    cleaned_chunks = []
    for i, chunk in enumerate(chunks):
        if detect_silence_in_chunk(chunk):
            logger.info(f"Chunk {i+1} is mostly silent. Skipping.")
            continue

        chunk = normalize_audio(chunk)
        chunk = reduce_noise(chunk)
        cleaned_chunks.append(chunk)
        logger.info(f"Chunk {i+1} cleaned and ready for processing.")

    return cleaned_chunks

def visualize_audio_chunks(cleaned_chunks, indices=None):
    """
    Visualize waveforms of the specified or all cleaned audio chunks.

    :param cleaned_chunks: List of cleaned AudioSegment objects
    :param indices: List of indices specifying which chunks to visualize (1-based). If None, all chunks will be visualized.
    """
    if indices is None:
        indices = range(1, len(cleaned_chunks) + 1)

    for i in indices:
        if 1 <= i <= len(cleaned_chunks):
            chunk = cleaned_chunks[i - 1]
            samples = np.array(chunk.get_array_of_samples())
            
            plt.figure(figsize=(10, 4))
            plt.plot(samples)
            plt.title(f"Waveform of Cleaned Chunk {i}")
            plt.xlabel("Sample Number")
            plt.ylabel("Amplitude")
            plt.show()
        else:
            logger.warning(f"Index {i} is out of range. No visualization available for this index.")

# Example usage
file_path = '/Users/pranay/Projects/LLM/video/proj1/data/Chiranjeevi_Video_Dec_21.mp4'
chunks = extract_and_chunk_audio(file_path)

# Clean the chunks
cleaned_chunks = clean_audio_chunks(chunks)

# Example: Save the cleaned chunks separately
save_cleaned_chunks(cleaned_chunks)

# Example: Visualize the cleaned chunks
visualize_audio_chunks(cleaned_chunks)

# Save specific chunks (e.g., chunks 1 and 3)
save_cleaned_chunks(cleaned_chunks, indices=[1, 3])

plt.ion()
# Visualize specific chunks (e.g., chunks 1 and 3)
visualize_audio_chunks(cleaned_chunks, indices=[1, 3])

# Save a single chunk (e.g., chunk 2)
save_cleaned_chunks(cleaned_chunks, indices=[2])

# Visualize a single chunk (e.g., chunk 2)
visualize_audio_chunks(cleaned_chunks, indices=[2])
