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
import re



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

# Embedding Models
EMBEDDING_MODEL_SMALL = "text-embedding-3-small"
EMBEDDING_MODEL_LARGE = "text-embedding-3-large"

# --- Function to Fetch Embeddings ---
def get_embedding_for_text(text, model=EMBEDDING_MODEL_SMALL):
    """Get embeddings for a given text using OpenAI."""
    try:
        text = text.replace("\n", " ")  # Ensure that newlines don't interfere with embedding
        response = client.embeddings.create(
            input=[text],  # Ensure input is a list of strings
            model=model,
            encoding_format="float"  # Optional, based on how you want the encoding format
        )
        
        # Correct access to embedding using new API structure
        embedding = response.data[0].embedding  # Access embedding through class attribute

        return embedding

    except Exception as e:
        logger.error(f"Error in fetching embeddings: {str(e)}")
        return None

    
def save_embeddings_to_file(embedding, aligned_embedding, filepath):
    """
    Save both transcription and aligned embeddings to a separate JSON file.
    """
    try:
        embeddings_data = {
            "transcription_embedding": embedding,
            "aligned_embedding": aligned_embedding
        }
        with open(filepath, "w") as f:
            json.dump(embeddings_data, f, indent=4)
        logger.info(f"Embeddings saved at: {filepath}")
    except Exception as e:
        logger.error(f"Error saving embeddings: {str(e)}")

transcript_schema= {
    # "transcription": {
    #     "full_text": "",
    #     "timestamps": [
    #         {
    #             "time": "",
    #             "text": "",
    #             "speaker": "",
    #             "non_verbal": ""
    #         }
    #     ]
    # },
    "content_summary": {
        "main_points": "",
        "interaction_purpose": ""
    },
    "participants": [
        {
            "name": "",
            "role": "",
            "summary": "",
            "details": {
                "symptoms": "",
                "diagnoses": "",
                "medications": "",
                "treatment_plans": ""
            }
        }
    ],
    "significant_timestamps": [
        {
            "time": "",
            "event_type": "",
            "description": ""
        }
    ],
    "emotional_psychological_analysis": [
        {
            "speaker": "",
            "emotion": "",
            "intensity": "",
            "timestamp": "",
            "context": ""
        }
    ],
    "patient_history": {
        "past_medical_conditions": "",
        "surgeries": "",
        "allergies": "",
        "family_history": "",
        "social_history": "",
        "noted_gaps_or_inconsistencies": ""
    },
    "test_results": [
        {
            "test_name": "",
            "result": "",
            "reference_range": "",
            "interpretation": "",
            "clinical_significance": "",
            "follow_up_recommended": ""
        }
    ],
    "keyword_topic_analysis": {
        "keywords": [""],
        "main_topics": [
            {
                "topic": "",
                "importance": "",
                "time_spent": ""
            }
        ],
        "topic_flow_assessment": ""
    },
    "quality_of_care_assessment": {
        "guideline_adherence": "",
        "comprehensiveness": "",
        "identified_gaps": "",
        "efficiency_assessment": ""
    },
    "overall_assessment": {
        "key_findings": "",
        "critical_points": "",
        "areas_for_improvement": "",
        "positive_aspects": "",
        "follow_up_recommendations": ""
    },
    "metadata": {
        "recording_date": "",
        "recording_time": "",
        "location": "",
        "setting": "",
        "duration": "",
        "visit_type": "",
        "technology_used": "",
        "healthcare_providers": ""
    },
    "doctor_patient_interaction": {
        "doctor_speaking_time": "",
        "patient_speaking_time": "",
        "interruptions": "",
        "empathy_expressions": "",
        "overall_patient_satisfaction": ""
    },
    "risk_stratification": {
        "risk_category": "",
        "underlying_conditions": "",
        "risk_factors": "",
        "recommendations": ""
    },
    "diagnostic_imaging_analysis": {
        "image_type": "",
        "findings": "",
        "follow_up_tests": "",
        "implications_for_treatment": ""
    },
    "treatment_follow_up": {
        "treatment_followed": "",
        "adherence_level": "",
        "outcomes_observed": "",
        "side_effects": "",
        "additional_treatments_required": ""
    },
    "unexpected_elements": {
        "conversation_segments": "",
        "additional_information": ""
    },
    "symptom_progression_analysis": {
        "symptom": "",
        "progression": "",
        "timeline": ""
    },
    "therapeutic_response_analysis": {
        "therapy_name": "",
        "response_status": "",
        "side_effects": "",
        "adherence_level": ""
    },
    "non_verbal_cues_analysis": {
        "cue_type": "",
        "timestamp": "",
        "emotional_context": ""
    },
    "behavioral_analysis": {
        "behavioral_marker": "",
        "frequency": "",
        "timestamp": "",
        "context": ""
    },
    "cross_patient_data_comparison": {
        "comparison_metric": "",
        "patient_set": "",
        "findings": ""
    },
    "language_translation_localization": {
        "original_language": "",
        "translated_text": "",
        "cultural_adaptations": "",
        "accuracy_score": ""
    },
    "medical_literature_suggestions": {
        "paper_title": "",
        "journal": "",
        "publication_date": "",
        "doi_link": ""
    },
    "longitudinal_health_trends_analysis": {
        "health_trend": "",
        "time_period": "",
        "observed_changes": ""
    },
    "clinical_trial_matching": {
        "trial_name": "",
        "eligibility_criteria": "",
        "trial_status": "",
        "contact_information": ""
    },
    "health_cues": {
        "cue": "",
        "frequency": "",
        "intensity": "",
        "timestamp": "",
        "context": ""
    },
    "doctor_mentions": {
        "doctor_name": "",
        "specialty": "",
        "context_of_mention": "",
        "timestamp": ""
    },
    "medicine_mentions": {
        "medicine_name": "",
        "dosage": "",
        "frequency": "",
        "context_of_use": "",
        "prescribed_by": "",
        "timestamp": ""
    },
    "lab_hospital_mentions": {
        "facility_name": "",
        "facility_type": "",
        "location": "",
        "services_provided": "",
        "timestamp": ""
    },
    "demographics_family": {
        "age": "",
        "gender": "",
        "location": "",
        "family_members": "",
        "family_history_of_conditions": ""
    },
    "region_specific_health_trends": {
        "region": "",
        "health_trend": "",
        "observed_increase_or_decrease": "",
        "commentary": ""
    },
    "ambient_sound_analysis": {
        "sound_type": "",
        "intensity": "",
        "duration": "",
        "timestamp": "",
        "context": ""
    },
    "voice_biomarkers": {
        "biomarker_type": "",
        "confidence_level": "",
        "timestamp": "",
        "related_symptoms": "",
        "context": ""
    },
    "additional_information": {
        "info_type": "",
        "details": ""
    },
    "call_technical_details": {
        "call_quality": "",
        "connection_issues": "",
        "background_noise": ""
    },
    "patient_communication_style": {
        "clarity_of_expression": "",
        "comprehension_level": "",
        "engagement_level": ""
    },
    "follow_up_actions": {
        "scheduled_appointments": "",
        "prescribed_tests": "",
        "referrals": ""
    },
    "patient_education": {
        "topics_covered": "",
        "patient_understanding": "",
        "resources_provided": ""
    },
    "compliance_and_regulatory": {
        "hipaa_adherence": "",
        "informed_consent": "",
        "documentation_completeness": ""
    },
    "decision_making_process": {
        "options_presented": "",
        "patient_involvement": "",
        "reasoning_explained": ""
    },
    "call_efficiency": {
        "total_duration": "",
        "time_to_key_points": "",
        "unresolved_issues": ""
    },
    "cultural_competence": {
        "language_considerations": "",
        "cultural_sensitivities_addressed": "",
        "use_of_interpreters": ""
    },
    "patient_satisfaction_indicators": {
        "verbal_cues": "",
        "expressed_concerns": "",
        "positive_feedback": ""
    },
    "telemedicine_specific_factors": {
        "visual_assessment_quality": "",
        "technical_difficulties_impact": "",
        "physical_exam_limitations": ""
    },
    "health_information_technology_usage": {
        "electronic_health_record_references": "",
        "clinical_decision_support_tools": "",
        "telemedicine_platform_functionality": ""
    },
    "research_participation_discussion": {
        "clinical_trials_mentioned": "",
        "patient_registries": "",
        "biobank_participation": ""
    },
    "end_of_life_care_planning": {
        "palliative_care_options": "",
        "hospice_discussion": "",
        "life-sustaining_treatment_preferences": ""
    },
    "health_equity_considerations": {
        "language_justice": "",
        "cultural_safety_practices": "",
        "accessibility_accommodations": ""
    },
    "patient_activation_measure": {
        "knowledge_level": "",
        "skill_level": "",
        "confidence_level": ""
    },
    "patient_self_management": {
        "self-monitoring_practices": "",
        "lifestyle_modification_efforts": "",
        "adherence_to_home_care_instructions": ""
    },
    "digital_health_integration": {
        "wearable_device_data": "",
        "patient_portal_usage": "",
        "mobile_health_app_recommendations": ""
    },
    "social_determinants_of_health": {
        "housing_situation": "",
        "food_security": "",
        "transportation_access": "",
        "employment_status": ""
    },
    "pain_assessment": {
        "pain_scale_used": "",
        "pain_description": "",
        "impact_on_daily_activities": "",
        "pain_management_strategies": ""
    },
    "medication_reconciliation": {
        "current_medications_reviewed": "",
        "medication_changes_made": "",
        "potential_drug_interactions_identified": ""
    },
    "genetic_considerations": {
        "family_genetic_history_discussed": "",
        "genetic_testing_recommendations": "",
        "personalized_medicine_approaches": ""
    },
    "environmental_health_factors": {
        "occupational_hazards": "",
        "home_environment_assessment": "",
        "exposure_to_pollutants": ""
    },
    "patient_reported_outcomes": {
        "quality_of_life_measures": "",
        "functional_status_assessment": "",
        "symptom_burden_evaluation": ""
    },
    "emergency_preparedness": {
        "emergency_contact_information": "",
        "disaster_readiness_plans": "",
        "medical_alert_systems": ""
    },
    "integrative_medicine_approaches": {
        "complementary_therapies_discussed": "",
        "mind-body_interventions": "",
        "nutritional_supplement_recommendations": ""
    },
    "medical_device_mentions": {
        "device_name": "",
        "purpose": "",
        "usage_instructions": "",
        "patient_familiarity": ""
    },
    "healthcare_cost_discussion": {
        "cost_estimates_provided": "",
        "insurance_coverage_discussed": "",
        "financial_concerns_addressed": ""
    },
    "care_coordination": {
        "interdisciplinary_team_involvement": "",
        "referral_coordination": "",
        "care_transition_plans": ""
    },
    "health_literacy_assessment": {
        "patient_knowledge_level": "",
        "use_of_medical_jargon": "",
        "clarification_requests": ""
    },
    "preventive_care_discussion": {
        "screenings_recommended": "",
        "lifestyle_modifications_suggested": "",
        "vaccination_status_reviewed": ""
    },
    "patient_support_system": {
        "family_involvement": "",
        "caregiver_presence": "",
        "community_resources_mentioned": ""
    },
    "legal_and_ethical_considerations": {
        "capacity_assessment": "",
        "advance_directive_discussion": "",
        "ethical_dilemmas_addressed": ""
    }
}


# system_prompt = """
# You are an advanced medical transcription analysis assistant. Your task is to analyze the provided medical transcription data, returning a structured output based on the predefined JSON schema. 
# You must utilize the semantic meaning from the text to produce the most accurate and insightful analysis.

# Ensure that the final output is a valid JSON and comprehensively covers all sections of the schema. 

# The output should strictly adhere to the following JSON schema:
# '{transcript_schema}'

# If any insights or findings fall outside the schema, include them in a dedicated 'additional_analysis' section.

# Your analysis must be:
# 1. Precise: Cover all key aspects derived from the text.
# 2. Thorough: Capture any subtle nuances, context, or patterns that the data presents.
# 3. Structured: Ensure the output is clean and ready for further processing.

# """

system_prompt="""
You are an advanced medical transcription analysis assistant. Your task is to analyze a provided medical transcription and return a structured output based on a predefined JSON schema. You must utilize the semantic meaning from the text to produce the most accurate and insightful analysis.

Here is the JSON schema you should use for your output:
<schema>
{{transcript_schema}}
</schema>

Now, here is the transcript you need to analyze:
<transcript>
{{TRANSCRIPT}}
</transcript>

Please follow these instructions to complete your task:

1. Carefully read and analyze the provided transcript.

2. Based on your analysis, fill out the JSON schema provided earlier. Ensure that your output is a valid JSON and comprehensively covers all sections of the schema.

3. For each section of the schema:
   a. Extract relevant information from the transcript.
   b. Summarize and structure the information according to the schema.
   c. If a particular field is not applicable or the information is not available in the transcript, use an empty string or array as appropriate.

4. Pay special attention to the following sections:
   a. transcription: Provide a full text transcription and detailed timestamps.
   b. content_summary: Summarize the main points and purpose of the interaction.
   c. participants: Identify all participants, their roles, and provide summaries of their contributions.
   d. significant_timestamps: Note any crucial moments in the conversation.
   e. emotional_psychological_analysis: Analyze the emotional state of participants throughout the conversation.
   f. patient_history: Extract any mentioned medical history.
   g. test_results: Note any test results discussed in the conversation.
   h. keyword_topic_analysis: Identify main topics and assess their importance.
   i. quality_of_care_assessment: Evaluate the care provided based on the conversation.
   j. overall_assessment: Provide a comprehensive assessment of the interaction.

5. If you encounter any insights or findings that fall outside the provided schema, include them in the "additional_information" section of the schema.

6. If you are unsure about any information or if the transcript is unclear, indicate this in the relevant field of the schema. Do not make assumptions or fill in information that is not present in the transcript.

7. Ensure your analysis is:
   a. Precise: Cover all key aspects derived from the text.
   b. Thorough: Capture any subtle nuances, context, or patterns that the data presents.
   c. Structured: Ensure the output is clean and ready for further processing.

8. After completing your analysis, review your output to ensure it adheres to the provided schema and contains all relevant information from the transcript.

9. Present your final output as a valid JSON object, structured according to the provided schema. Enclose your entire output within <analysis> tags.

Remember, your goal is to provide a comprehensive, accurate, and structured analysis of the medical transcription that can be easily processed and understood by other systems or healthcare professionals.
"""

# Load WhisperX Model
whisper_model = whisperx.load_model("large-v2", device, compute_type=compute_type)
# --- Schemas for Output Structure ---

# Utility function to save JSON data
def save_json_to_file(data, filepath):
    try:
        # Check if the data contains required fields like 'text' and 'embedding'
        if 'text' not in data and 'embedding' not in data:
            logger.error(f"Data missing 'text' or 'embedding'. Cannot save to {filepath}.")
            return

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
        
        # Ensure all segments contain text and check if segments exist
        if "segments" not in result or not result["segments"]:
            logger.error(f"No valid segments found in transcription result for {audio_file_path}.")
            return None
        
        # Ensure all segments contain text
        if not all("text" in segment for segment in result["segments"]):
            logger.error("Missing 'text' in one or more transcription segments for {audio_file_path}.")
            return None
        
        # Print the raw segments before alignment
        logger.info(f"Initial transcription result for {audio_file_path}: {result['segments']}")
        
         # Generate embeddings for the transcription result
        transcription_text = " ".join([segment["text"] for segment in result["segments"]])
        transcription_embedding = get_embedding_for_text(transcription_text, model=EMBEDDING_MODEL_SMALL)

        # Check if embedding generation was successful
        if transcription_embedding:
            # Store the embedding in the result
            result['embedding'] = transcription_embedding
        else:
            logger.error(f"Failed to generate embeddings for transcription result of {audio_file_path}.")
            return None
        
        return result
    
    except Exception as e:
        logger.error(f"Error during transcription for {audio_file_path}: {str(e)}")
        return None

# --- Step 8: WhisperX Alignment ---
def align_with_whisperx(result, audio_file_path, whisper_model, device="cpu"):
    logger.info(f"Aligning transcription for {audio_file_path}...")
    
    try:
        # Load alignment model
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        
        # Align results
        aligned_result = whisperx.align(result["segments"], model_a, metadata, audio_file_path, device)
        
        # Ensure all segments contain text and check if segments exist
        if "segments" not in aligned_result or not aligned_result["segments"]:
            logger.error(f"No valid segments found in alignment result for {audio_file_path}.")
            return None

        if not all("text" in segment for segment in aligned_result["segments"]):
            logger.error(f"Missing 'text' in one or more aligned segments for {audio_file_path}.")
            return None

        logger.info(f"Alignment completed for {audio_file_path}: {aligned_result}.")
        
        # Generate embeddings for the aligned result
        aligned_text = " ".join([segment["text"] for segment in aligned_result["segments"]])
        aligned_embedding = get_embedding_for_text(aligned_text, model=EMBEDDING_MODEL_LARGE)

        # Check if embedding generation was successful
        if aligned_embedding:
            # Store the embedding in the aligned result
            aligned_result['embedding'] = aligned_embedding
        else:
            logger.error(f"Failed to generate embeddings for aligned result of {audio_file_path}.")
            return None
        
        return aligned_result
    
    except Exception as e:
        logger.error(f"Error during alignment for {audio_file_path}: {str(e)}")
        return None

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


# def analyze_with_openai(transcription_chunk, aligned_result, seed=None):
#     """
#     This function sends the transcription data (text + embedding) and only the embedding from the aligned data 
#     to the OpenAI API for analysis. It expects a structured JSON output based on the schema provided in the prompt.
#     """
#     logger.info(f"Starting analysis with OpenAI for transcription chunk: {transcription_chunk}")
#     logger.info(f"Aligned result: {aligned_result}")
#     # Ensure both transcription and aligned data are available
#     # if not transcription_chunk or not aligned_result:
#     #     logger.error("Missing transcription or aligned result data. Cannot proceed with analysis.")
#     #     return None

#     # # Ensure transcription contains text and embeddings
#     # if "text" not in transcription_chunk or "embedding" not in transcription_chunk:
#     #     logger.error("Missing text or embeddings in the transcription chunk. Aborting analysis.")
#     #     return None

#     # # Ensure aligned result contains the embedding
#     # if "embedding" not in aligned_result:
#     #     logger.error("Missing embeddings in the aligned result. Aborting analysis.")
#     #     return None
#     # Step 1: Ensure transcription_chunk is present and extract text + embedding
#     try:
#         # Step 1: Ensure transcription_chunk is present and extract text + embedding
#         if not transcription_chunk:
#             logger.error("Transcription chunk is missing. Cannot proceed with analysis.")
#             return None

#         # Extract text and embedding from transcription_chunk
#         if "segments" in transcription_chunk and transcription_chunk["segments"]:
#             segment_text = " ".join([segment.get("text", "") for segment in transcription_chunk["segments"]])
#             embedding = transcription_chunk.get("embedding", None)

#             if not segment_text:
#                 logger.error("No text found in the transcription chunk segments.")
#                 return None
#             if not embedding:
#                 logger.error("No embedding found in the transcription chunk.")
#                 return None

#             logger.info(f"Extracted text from transcription chunk: {segment_text[:100]}...")  # Log first 100 characters of text
#             logger.info(f"Length of embedding from transcription chunk: {len(embedding)}")
#         else:
#             logger.error("No segments found in the transcription chunk.")
#             return None

#         # Step 2: Ensure aligned result is present and contains embedding
#         if not aligned_result:
#             logger.error("Aligned result is missing. Cannot proceed with analysis.")
#             return None

#         if "embedding" not in aligned_result:
#             logger.error(f"Missing embedding in the aligned result. Available keys: {aligned_result.keys()}")
#             return None

#         logger.info(f"Length of embedding from aligned result: {len(aligned_result['embedding'])}")

#     # Build the user prompt dynamically, inserting the transcription data (text + embedding) and only embedding from aligned data
#         user_prompt = f"""
#         Please analyze the following transcription data (text and embedding) and aligned data (embedding only), and return a **JSON** output based on the predefined schema. The analysis should consider both the semantic meaning of the transcription text and the embeddings from both datasets.
#         Transcription Data:
#         - Text: {segment_text[:300]}... (truncated for brevity)
#         - Embedding: {embedding[:10]}... (truncated for brevity)

#         Aligned Data (Embedding Only):
#         - Embedding: {aligned_result['embedding'][:10]}... (truncated for brevity)
#         """
#     # Transcription Data:
#     # - Text: {transcription_chunk['text']}
#     # - Embedding: {transcription_chunk['embedding']}

#     # Aligned Data (Embedding Only):
#     # - Embedding: {aligned_result['embedding']}
#     # """

#         logger.info(f"Built user prompt for OpenAI analysis: {user_prompt[:300]}...")  # Log first 300 characters of the prompt
    
    
#         # Call the OpenAI API for structured output
#         response = client.chat.completions.create(
#             model="gpt-4o-mini-2024-07-18",  # Use the latest available model
#             messages=[
#                 {"role": "system", "content": system_prompt},  # System prompt
#                 {"role": "user", "content": user_prompt}  # User prompt with dynamic data
#             ],
#             seed=SEED,
#             temperature=0.3,
#             top_p=1,
#             frequency_penalty=0,
#             presence_penalty=0,
#             # max_tokens=2000,  # Adjust based on the expected output size
#             stop=None  # This can be added if needed
#         )
#         # Extract and log the raw response
#         full_response = response.choices[0].message['content']
#         logger.info(f"Full raw response from OpenAI: {full_response}")  # Log the full raw response

#         # Optionally save raw response to a file
#         with open("raw_openai_response.json", "w") as f:
#             f.write(full_response)

#         # Attempt to parse the response as JSON
#         try:
#             structured_output_json = json.loads(full_response)
#             logger.info("Successfully parsed JSON response.")
#             return structured_output_json

#         except json.JSONDecodeError as e:
#             logger.error(f"Failed to decode JSON: {str(e)}")
#             logger.error(f"Raw response from OpenAI: {full_response}")  # Log raw response for debugging
#             return None

#     except Exception as e:
#         logger.error(f"Error during OpenAI API call: {str(e)}")
#         return None



# def analyze_with_openai(transcription_chunk, aligned_result, seed=None):
#     """
#     This function sends the transcription data (text + embedding) and only the embedding from the aligned data 
#     to the OpenAI API for analysis. It expects a structured JSON output based on the schema provided in the prompt.
#     """
#     logger.info(f"Starting analysis with OpenAI for transcription chunk: {transcription_chunk}")
#     logger.info(f"Aligned result: {aligned_result}")

#     try:
#         # Step 1: Ensure transcription_chunk is present and extract text + embedding
#         if not transcription_chunk:
#             logger.error("Transcription chunk is missing. Cannot proceed with analysis.")
#             return None

#         # Extract text and embedding from transcription_chunk
#         if "segments" in transcription_chunk and transcription_chunk["segments"]:
#             segment_text = " ".join([segment.get("text", "") for segment in transcription_chunk["segments"]])
#             embedding = transcription_chunk.get("embedding", None)

#             if not segment_text:
#                 logger.error("No text found in the transcription chunk segments.")
#                 return None
#             if not embedding:
#                 logger.error("No embedding found in the transcription chunk.")
#                 return None

#             logger.info(f"Extracted text from transcription chunk: {segment_text[:100]}...")  # Log first 100 characters of text
#             logger.info(f"Length of embedding from transcription chunk: {len(embedding)}")
#         else:
#             logger.error("No segments found in the transcription chunk.")
#             return None

#         # Step 2: Ensure aligned result is present and contains embedding
#         if not aligned_result:
#             logger.error("Aligned result is missing. Cannot proceed with analysis.")
#             return None

#         if "embedding" not in aligned_result:
#             logger.error(f"Missing embedding in the aligned result. Available keys: {aligned_result.keys()}")
#             return None

#         logger.info(f"Length of embedding from aligned result: {len(aligned_result['embedding'])}")

#         # Build the user prompt dynamically, inserting the transcription data (text + embedding) and only embedding from aligned data
#         # user_prompt = f"""
#         # Please analyze the following transcription data (text and embedding) and aligned data (embedding only), and return a **JSON** output based on the predefined schema. The analysis should consider both the semantic meaning of the transcription text and the embeddings from both datasets.
#         # Transcription Data:
#         # - Text: {segment_text[:300]}... (truncated for brevity)
#         # - Embedding: {embedding[:10]}... (truncated for brevity)

#         # Aligned Data (Embedding Only):
#         # - Embedding: {aligned_result['embedding'][:10]}... (truncated for brevity)
#         # """
#         user_prompt = f"""
#         Please analyze the following transcription data (text and embedding) and aligned data (embedding only), and return a **JSON** output based on the predefined schema. The analysis should consider both the semantic meaning of the transcription text and the embeddings from both datasets.
#         Transcription Data:
#         - Text: {segment_text[:300]}... (truncated for brevity)
#         - Embedding: {embedding[:10]}... (truncated for brevity)

#         Aligned Data (Embedding Only):
#         - Embedding: {aligned_result['embedding'][:10]}... (truncated for brevity)
#         """
        
#         logger.info(f"Built user prompt for OpenAI analysis: {user_prompt[:300]}...")  # Log first 300 characters of the prompt
        
#         # Call the OpenAI API for structured output
#         response = client.chat.completions.create(
#             model="gpt-4o-mini-2024-07-18",  # Use the latest available model
#             messages=[
#                 {"role": "system", "content": system_prompt},  # System prompt
#                 {"role": "user", "content": user_prompt}  # User prompt with dynamic data
#             ],
#             temperature=0.3,
#             top_p=1,
#             max_tokens=2000,  # Adjust based on expected output size
#             stop=None  # This can be added if needed
#         )

#         # Extract and log the raw response
#         full_response = response.choices[0].message.content
#         logger.info(f"Full raw response from OpenAI: {full_response}")  # Log the full raw response

#         # Optionally save raw response to a file
#         with open("raw_openai_response.json", "w") as f:
#             f.write(full_response)

#         # Attempt to parse the response as JSON
#         try:
#             # Remove markdown code block delimiters if present
#             json_string = re.sub(r'```json\s*|\s*```', '', full_response).strip()
            
#             structured_output_json = json.loads(json_string)
#             logger.info("Successfully parsed JSON response.")
#             return structured_output_json

#         except json.JSONDecodeError as e:
#             logger.error(f"Failed to decode JSON: {str(e)}")
#             logger.error(f"Raw response from OpenAI: {full_response}")  # Log raw response for debugging
#             return None

#     except Exception as e:
#         logger.error(f"Error during OpenAI API call: {str(e)}", exc_info=True)  # Log full traceback
#         return None

def analyze_with_openai(transcription_chunk, seed=None):
    """
    This function sends the transcription text (only) to the OpenAI API for analysis.
    The embeddings are extracted and saved separately.
    """
    logger.info(f"Starting analysis with OpenAI for transcription chunk: {transcription_chunk}")
    # logger.info(f"Aligned result: {aligned_result}")

    try:
        # Step 1: Ensure transcription_chunk is present and extract text + embedding
        if not transcription_chunk:
            logger.error("Transcription chunk is missing. Cannot proceed with analysis.")
            return None

        # Extract text and embedding from transcription_chunk
        if "segments" in transcription_chunk and transcription_chunk["segments"]:
            # Full text to be sent to OpenAI
            segment_text = " ".join([segment.get("text", "") for segment in transcription_chunk["segments"]])
            embedding = transcription_chunk.get("embedding", None)

            if not segment_text:
                logger.error("No text found in the transcription chunk segments.")
                return None
            if not embedding:
                logger.error("No embedding found in the transcription chunk.")
                return None

            # Log only the first 100 characters for brevity
            logger.info(f"Extracted text from transcription chunk (truncated): {segment_text[:100]}...")  
            logger.info(f"Length of embedding from transcription chunk: {len(embedding)}")
        else:
            logger.error("No segments found in the transcription chunk.")
            return None

        # # Step 2: Ensure aligned result is present and contains embedding
        # if not aligned_result:
        #     logger.error("Aligned result is missing. Cannot proceed with analysis.")
        #     return None

        # if "embedding" not in aligned_result:
        #     logger.error(f"Missing embedding in the aligned result. Available keys: {aligned_result.keys()}")
        #     return None

        # logger.info(f"Length of embedding from aligned result: {len(aligned_result['embedding'])}")

        # # Save embeddings separately
        # embedding_filepath = "embeddings_cleaned_chunk.json"
        # save_embeddings_to_file(embedding, aligned_result["embedding"], embedding_filepath)

        # Build the user prompt using the **full transcription text**
        user_prompt = f"""
        Please analyze the following transcription data, and return a **JSON** output based on the predefined schema. The analysis should consider the semantic meaning of the transcription text.
        Transcription Data:
        - Text: '{segment_text}'  # Full content, not truncated
        """

        logger.info(f"Built user prompt for OpenAI analysis: {user_prompt[:300]}...")  # Log only the first 300 characters of the prompt for brevity

        # Call the OpenAI API for structured output
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",  # Use the latest available model
            messages=[
                {"role": "system", "content": system_prompt},  # System prompt
                {"role": "user", "content": user_prompt}  # User prompt with the full transcription data
            ],
            seed=SEED,
            temperature=0.3,
            top_p=1,
            # max_tokens=2000,  # Adjust based on expected output size
            stop=None  # This can be added if needed
        )

        # Extract and log the raw response
        full_response = response.choices[0].message.content
        logger.info(f"Full raw response from OpenAI: {full_response}")  # Log the full raw response

        # Optionally save raw response to a file
        with open("raw_openai_response.json", "w") as f:
            f.write(full_response)

        # Attempt to parse the response as JSON
        try:
            # Remove markdown code block delimiters if present
            json_string = re.sub(r'```json\s*|\s*```', '', full_response).strip()
            
            structured_output_json = json.loads(json_string)
            logger.info("Successfully parsed JSON response.")
            return structured_output_json

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON: {str(e)}")
            logger.error(f"Raw response from OpenAI: {full_response}")  # Log raw response for debugging
            return None

    except Exception as e:
        logger.error(f"Error during OpenAI API call: {str(e)}", exc_info=True)  # Log full traceback
        return None

def process_transcription(transcription_data, schema_sections, aligned_keys, transcription_keys):
    """
    Process transcription with schema into aligned and transcription-related sections.
    """
    logger.info("Processing transcription with schema...")

    # Prepare to store both aligned and transcription-related outputs
    structured_output = {}

    # Analyze transcription sections based on the schema keys
    for key in transcription_keys:
        if key in transcription_data:
            structured_output[key] = transcription_data[key]

    for key in aligned_keys:
        if key in transcription_data:
            structured_output[key] = transcription_data[key]

    return structured_output

# def main():
#     # Path to your transcription file (you can modify it based on your file structure)
#     transcription_file_path = "/Users/pranay/Projects/LLM/video/proj1/data/Chiranjeevi_Video_Dec_21_segment_0_1401.583333/segment_0_1401.583333/transcription_chunk_cleaned_chunk_1.wav.json"

#     # Load the transcription data
#     transcription_data = load_transcription_file(transcription_file_path)

#     if transcription_data:
#         # Assuming the transcription chunk is part of the loaded JSON data
#         transcription_chunk = json.dumps(transcription_data)  # Convert dict to JSON string if necessary

#         # Analyze transcription and get the result
#         analysis_result = analyze_transcription_basic(transcription_chunk)

#         if analysis_result:
#             # Output the result
#             print("Analysis Result:")
#             print(analysis_result)
            
#             # Save result to a file (optional)
#             output_file_path = os.path.join(os.path.dirname(transcription_file_path), "final_analysis_cleaned_chunk_1.json")
#             with open(output_file_path, 'w') as outfile:
#                 json.dump({"analysis": analysis_result}, outfile)
#             logger.info(f"Results saved as {output_file_path}")

# if __name__ == "__main__":
#     main()

def main(video_file_path, hf_token, whisper_model):
    logger.info(f"Starting processing for video: {video_file_path}")

    try:
        # Step 1: Segment the video
        segments = segment_large_video(video_file_path)
        if not segments:
            logger.error(f"No segments created from video: {video_file_path}")
            return
    except Exception as e:
        logger.error(f"Error during video segmentation: {str(e)}")
        return

    for segment_file_path in segments:
        segment_folder = os.path.splitext(segment_file_path)[0]
        os.makedirs(segment_folder, exist_ok=True)

        try:
            # Step 2: Extract audio from the video segment
            audio_file_path = extract_audio_from_segment(segment_file_path)
            logger.info(f"Audio extracted from segment: {segment_file_path}")

            # Step 3: Remove silence from the audio file
            cleaned_audio_path = remove_silence(audio_file_path)
            logger.info(f"Silence removed from audio: {audio_file_path}")

            # Step 4: Chunk the cleaned audio into smaller parts
            chunks = chunk_audio(cleaned_audio_path)
            if not chunks:
                logger.error(f"No chunks created for cleaned audio: {cleaned_audio_path}")
                continue  # Skip this segment if chunking failed
            logger.info(f"Created {len(chunks)} chunks for cleaned audio: {cleaned_audio_path}")

            # Step 5: Normalize and reduce noise in the audio chunks
            cleaned_chunks = []
            for chunk in chunks:
                cleaned_chunk = normalize_and_reduce_noise(AudioSegment.from_file(chunk))
                cleaned_chunks.append(cleaned_chunk)
            logger.info(f"Normalized and noise-reduced chunks: {len(cleaned_chunks)}")

            # Step 6: Save the cleaned chunks
            cleaned_chunk_paths = save_cleaned_chunks(cleaned_chunks, segment_folder)

            # Step 7: Process each cleaned chunk for transcription and alignment
            for cleaned_chunk_path in tqdm(cleaned_chunk_paths, desc="Processing cleaned chunks"):
                logger.info(f"Processing chunk: {cleaned_chunk_path}")

                # Transcribe the cleaned chunk using WhisperX
                transcription_result = transcribe_with_whisperx(cleaned_chunk_path, whisper_model)

                # Skip the chunk if transcription fails
                if not transcription_result:
                    logger.error(f"Skipping chunk {cleaned_chunk_path} due to transcription failure.")
                    continue
                
                # Save the transcription result
                transcription_file = os.path.join(segment_folder, f"transcription_chunk_{os.path.basename(cleaned_chunk_path)}.json")
                save_json_to_file(transcription_result, transcription_file)
                logger.info(f"Transcription saved at: {transcription_file}")

                # # Align the transcription with WhisperX
                # aligned_result = align_with_whisperx(transcription_result, cleaned_chunk_path, whisper_model)

                # # Skip the chunk if alignment fails
                # if not aligned_result:
                #     logger.error(f"Skipping chunk {cleaned_chunk_path} due to alignment failure.")
                #     continue

                # # Save the aligned transcription result
                # aligned_file = os.path.join(segment_folder, f"aligned_transcription_chunk_{os.path.basename(cleaned_chunk_path)}.json")
                # save_json_to_file(aligned_result, aligned_file)
                # logger.info(f"Aligned transcription saved at: {aligned_file}")

                # Prepare transcription chunk and aligned result for analysis
                transcription_chunk = transcription_result  # Use full transcription result (text + embedding)
                # aligned_result_json = aligned_result  # Use full aligned result, but send only the embedding to OpenAI

                # Call OpenAI for final analysis using the transcription text + embedding and aligned embedding
                analysis_result = analyze_with_openai(transcription_chunk)

                if analysis_result:
                    # Save the final analysis result to a file
                    final_output_file = os.path.join(segment_folder, f"final_analysis_cleaned_chunk_{os.path.basename(cleaned_chunk_path)}.json")
                    save_json_to_file({"analysis": analysis_result}, final_output_file)
                    logger.info(f"Final analysis saved as {final_output_file}")
                else:
                    logger.error(f"Failed to generate analysis for chunk {os.path.basename(cleaned_chunk_path)}")

        except Exception as e:
            logger.error(f"Error processing segment {segment_file_path}: {str(e)}")
            continue  # Continue processing the next segment

if __name__ == "__main__":
    # Define the path to your video file
    video_file_path = '/Users/pranay/Projects/LLM/video/proj1/data/Chiranjeevi_Video_Dec_21.mp4'
    hf_token = os.getenv("HUGGINGFACE_TOKEN")  # Replace with your actual Hugging Face token if required

    # Call the main function with the video path and other parameters
    main(video_file_path, hf_token, whisper_model)
