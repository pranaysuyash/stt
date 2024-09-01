import os
from dotenv import load_dotenv
import google.generativeai as genai
from moviepy.editor import VideoFileClip
from pydub import AudioSegment, silence
import logging
import requests
import time
import json
import sys
import traceback

# Load environment variables
load_dotenv(dotenv_path='/Users/pranay/Projects/LLM/video/proj1/scripts/.env')

# Set up API Key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=api_key)

file_path = '/Users/pranay/Projects/LLM/video/proj1/data/Chiranjeevi_Video_Dec_21.mp4'  # Update with your file path

# --- Logging Setup --- 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def flush_log_handlers(logger):
    for handler in logger.handlers:
        handler.flush()

# --- System Prompt ---
system_prompt = """
You are an advanced AI assistant specialized in analyzing medical or health-related video/audio content. Your primary function is to extract, organize, and analyze various types of information from the provided content, focusing on medical details, patient interactions, and healthcare processes. You possess extensive knowledge of medical terminology, procedures, best practices in healthcare communication, and the latest advancements in medical science.

Your task is to analyze the given video/audio content and generate a comprehensive report in JSON format. This report will be used to power a frontend application, enabling users to explore summaries and conduct deep dives into detailed analyses based on specific individuals and key elements mentioned in the content.
"""

# --- User Prompt ---
user_prompt = """
Please analyze the following medical consultation recording. Ensure you capture all patient names, doctor names, and any other participant names. Organize the information so that users can easily explore summaries or deep-dive into each individual's details. Provide the output in JSON format, structured as per the guidelines.

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

Key Responsibilities:
1. Transcribe and analyze speech content with high accuracy, including speaker diarization.
2. Identify, categorize, and contextualize medical entities, conditions, and terminology.
3. Recognize and interpret medical diagnoses, treatments, and recommendations, including their certainty levels and potential implications.
4. Assess emotional content, interpersonal dynamics, and communication effectiveness in medical conversations.
5. Extract and organize relevant metadata and contextual information from the audio/video content.
6. Provide clear, concise, and accurate summaries of medical discussions while maintaining the nuances of the original content.
7. Analyze the quality and appropriateness of healthcare provider-patient interactions.
8. Identify potential cultural, socioeconomic, or linguistic factors that may influence the medical discussion or treatment plans.
9. Recognize and highlight any urgent medical situations, critical information, or potential red flags in the content.
10. Provide insights into the overall quality of care and adherence to medical best practices based on the analyzed content.

Guidelines:
1. Maintain strict confidentiality and adhere to HIPAA standards for all medical information. Include patient and doctor names as provided, but ensure security in data handling.
2. Use professional, clear, and grammatically correct language in all outputs.
3. Provide objective analysis without personal opinions or medical advice.
4. Balance technical accuracy with accessibility when explaining medical terms, ensuring the output is understandable to both medical professionals and general audiences.
5. Highlight any urgent medical situations or critical information prominently in your analysis.
6. Format all output in the specified JSON structure for consistency and ease of processing.
7. When referencing medical conditions, symptoms, procedures, or medications, include relevant standardized codes (ICD-10, SNOMED CT, RxNorm, CPT/HCPCS) where applicable.
8. If the analysis is too extensive for a single output, split it into multiple responses, ensuring each response is a valid JSON object and clearly indicating the continuation sequence.
9. Continuously cross-reference information throughout the analysis to identify patterns, inconsistencies, or notable correlations.
10. Approach the analysis with a multidisciplinary perspective, considering not only the immediate medical content but also potential psychosocial, environmental, and lifestyle factors that may influence the patient's health.

Additional Considerations:
1. Scalability and Large Video Handling:
   - For large video/audio files, segment the content into smaller, manageable parts and process each segment independently while maintaining context across segments.
   - Prioritize the extraction of the most critical information in each segment, ensuring continuity in the analysis across multiple segments.

2. Error Handling and Edge Cases:
   - Handle unclear audio, incomplete data, or conflicting information by flagging these sections in the JSON output for review.
   - Include disclaimers in the output indicating the need for further human review when necessary.

3. Security and Data Protection:
   - Ensure all data handling, storage, and transmission processes follow best practices for security, including encryption of sensitive information.
   - Include a section in the JSON output that logs any potential security or privacy concerns encountered during processing.

4. Customization and User Preferences:
   - Allow for user customization of the analysis, specifying areas of focus (e.g., emotional analysis, in-depth medical entity recognition) or adjusting the level of detail in the output (e.g., summary vs. full analysis).
   - Include a section in the JSON output that indicates the user-specified preferences and any custom settings applied during the analysis.

5. Localization and Cultural Sensitivity:
   - Implement localization capabilities to recognize and interpret medical terms in different languages, and adjust the analysis to account for cultural differences in healthcare practices.
   - Include a cultural competence assessment in the output, identifying any cultural, linguistic, or socioeconomic factors that influence the interaction and assessing the healthcare provider's sensitivity to these factors.

6. Front-End Integration Considerations:
   - Ensure that the JSON output is structured to facilitate easy integration with the frontend application. The JSON should be modular, allowing different sections to be displayed independently or in combination as needed by the user interface.
   - Consider including metadata in the JSON that can guide the frontend in prioritizing or highlighting specific sections, such as critical medical information or urgent flags.

Analysis Components:
{detailed components as per your provided instructions}

JSON Output Structure:
{full JSON structure as previously defined}

Instructions for Handling Long Content and Output:
{instructions for handling long content and splitting into parts}

Begin your analysis of the provided video content, adhering to these instructions and the specified JSON structure. Ensure your analysis is thorough, objective, and provides valuable insights for both medical professionals and patients.
"""

# --- Schemas for Output Structure ---
# Define the schema for transcription
transcription_schema = genai.protos.Schema(
    type=genai.protos.Type.OBJECT,
    properties={
        "full_text": genai.protos.Schema(type=genai.protos.Type.STRING),
        "timestamps": genai.protos.Schema(
            type=genai.protos.Type.ARRAY,
            items=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "time": genai.protos.Schema(type=genai.protos.Type.STRING),
                    "text": genai.protos.Schema(type=genai.protos.Type.STRING),
                    "speaker": genai.protos.Schema(type=genai.protos.Type.STRING),
                    "non_verbal": genai.protos.Schema(type=genai.protos.Type.STRING)
                }
            )
        )
    }
)

# Define the schema for content_summary
content_summary_schema = genai.protos.Schema(
    type=genai.protos.Type.OBJECT,
    properties={
        "main_points": genai.protos.Schema(type=genai.protos.Type.STRING),
        "interaction_purpose": genai.protos.Schema(type=genai.protos.Type.STRING)
    }
)

# Define the schema for participants
participants_schema = genai.protos.Schema(
    type=genai.protos.Type.OBJECT,
    properties={
        "patients": genai.protos.Schema(
            type=genai.protos.Type.ARRAY,
            items=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "name": genai.protos.Schema(type=genai.protos.Type.STRING),
                    "role": genai.protos.Schema(type=genai.protos.Type.STRING),
                    "summary": genai.protos.Schema(type=genai.protos.Type.STRING),
                    "details": genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            "symptoms": genai.protos.Schema(
                                type=genai.protos.Type.ARRAY,
                                items=genai.protos.Schema(
                                    type=genai.protos.Type.OBJECT,
                                    properties={
                                        "term": genai.protos.Schema(type=genai.protos.Type.STRING),
                                        "standardized_term": genai.protos.Schema(type=genai.protos.Type.STRING),
                                        "icd_10_code": genai.protos.Schema(type=genai.protos.Type.STRING),
                                        "snomed_ct_code": genai.protos.Schema(type=genai.protos.Type.STRING),
                                        "severity": genai.protos.Schema(type=genai.protos.Type.STRING),
                                        "duration": genai.protos.Schema(type=genai.protos.Type.STRING),
                                        "frequency": genai.protos.Schema(type=genai.protos.Type.STRING),
                                        "context": genai.protos.Schema(type=genai.protos.Type.STRING)
                                    }
                                )
                            ),
                            "diagnoses": genai.protos.Schema(
                                type=genai.protos.Type.ARRAY,
                                items=genai.protos.Schema(
                                    type=genai.protos.Type.OBJECT,
                                    properties={
                                        "diagnosis": genai.protos.Schema(type=genai.protos.Type.STRING),
                                        "icd_10_code": genai.protos.Schema(type=genai.protos.Type.STRING),
                                        "snomed_ct_code": genai.protos.Schema(type=genai.protos.Type.STRING),
                                        "status": genai.protos.Schema(type=genai.protos.Type.STRING),
                                        "certainty_level": genai.protos.Schema(type=genai.protos.Type.STRING),
                                        "basis": genai.protos.Schema(type=genai.protos.Type.STRING),
                                        "context": genai.protos.Schema(type=genai.protos.Type.STRING)
                                    }
                                )
                            ),
                            "medications": genai.protos.Schema(
                                type=genai.protos.Type.ARRAY,
                                items=genai.protos.Schema(
                                    type=genai.protos.Type.OBJECT,
                                    properties={
                                        "name": genai.protos.Schema(type=genai.protos.Type.STRING),
                                        "rxnorm_code": genai.protos.Schema(type=genai.protos.Type.STRING),
                                        "status": genai.protos.Schema(type=genai.protos.Type.STRING),
                                        "dosage": genai.protos.Schema(type=genai.protos.Type.STRING),
                                        "frequency": genai.protos.Schema(type=genai.protos.Type.STRING),
                                        "route": genai.protos.Schema(type=genai.protos.Type.STRING),
                                        "context": genai.protos.Schema(type=genai.protos.Type.STRING)
                                    }
                                )
                            ),
                            "treatment_plans": genai.protos.Schema(
                                type=genai.protos.Type.ARRAY,
                                items=genai.protos.Schema(
                                    type=genai.protos.Type.OBJECT,
                                    properties={
                                        "condition": genai.protos.Schema(type=genai.protos.Type.STRING),
                                        "plan": genai.protos.Schema(type=genai.protos.Type.STRING),
                                        "rationale": genai.protos.Schema(type=genai.protos.Type.STRING),
                                        "risks": genai.protos.Schema(type=genai.protos.Type.STRING),
                                        "benefits": genai.protos.Schema(type=genai.protos.Type.STRING),
                                        "alternatives_discussed": genai.protos.Schema(type=genai.protos.Type.STRING),
                                        "patient_involvement": genai.protos.Schema(type=genai.protos.Type.STRING)
                                    }
                                )
                            )
                        }
                    )
                }
            )
        ),
        "doctors": genai.protos.Schema(
            type=genai.protos.Type.ARRAY,
            items=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "name": genai.protos.Schema(type=genai.protos.Type.STRING),
                    "role": genai.protos.Schema(type=genai.protos.Type.STRING),
                    "summary": genai.protos.Schema(type=genai.protos.Type.STRING),
                    "details": genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            "speaking_patterns": genai.protos.Schema(
                                type=genai.protos.Type.OBJECT,
                                properties={
                                    "talk_time": genai.protos.Schema(type=genai.protos.Type.NUMBER),
                                    "interruptions": genai.protos.Schema(type=genai.protos.Type.NUMBER),
                                    "engagement_level": genai.protos.Schema(type=genai.protos.Type.STRING)
                                }
                            ),
                            "diagnoses": genai.protos.Schema(
                                type=genai.protos.Type.ARRAY,
                                items=genai.protos.Schema(
                                    type=genai.protos.Type.OBJECT,
                                    properties={
                                        "diagnosis": genai.protos.Schema(type=genai.protos.Type.STRING),
                                        "icd_10_code": genai.protos.Schema(type=genai.protos.Type.STRING),
                                        "snomed_ct_code": genai.protos.Schema(type=genai.protos.Type.STRING),
                                        "status": genai.protos.Schema(type=genai.protos.Type.STRING),
                                        "certainty_level": genai.protos.Schema(type=genai.protos.Type.STRING),
                                        "basis": genai.protos.Schema(type=genai.protos.Type.STRING),
                                        "context": genai.protos.Schema(type=genai.protos.Type.STRING)
                                    }
                                )
                            ),
                            "treatment_plans": genai.protos.Schema(
                                type=genai.protos.Type.ARRAY,
                                items=genai.protos.Schema(
                                    type=genai.protos.Type.OBJECT,
                                    properties={
                                        "condition": genai.protos.Schema(type=genai.protos.Type.STRING),
                                        "plan": genai.protos.Schema(type=genai.protos.Type.STRING),
                                        "rationale": genai.protos.Schema(type=genai.protos.Type.STRING),
                                        "risks": genai.protos.Schema(type=genai.protos.Type.STRING),
                                        "benefits": genai.protos.Schema(type=genai.protos.Type.STRING),
                                        "alternatives_discussed": genai.protos.Schema(type=genai.protos.Type.STRING),
                                        "patient_involvement": genai.protos.Schema(type=genai.protos.Type.STRING)
                                    }
                                )
                            )
                        }
                    )
                }
            )
        ),
        "others": genai.protos.Schema(
            type=genai.protos.Type.ARRAY,
            items=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "name": genai.protos.Schema(type=genai.protos.Type.STRING),
                    "role": genai.protos.Schema(type=genai.protos.Type.STRING),
                    "summary": genai.protos.Schema(type=genai.protos.Type.STRING),
                    "details": genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            "speaking_patterns": genai.protos.Schema(
                                type=genai.protos.Type.OBJECT,
                                properties={
                                    "talk_time": genai.protos.Schema(type=genai.protos.Type.NUMBER),
                                    "interruptions": genai.protos.Schema(type=genai.protos.Type.NUMBER),
                                    "engagement_level": genai.protos.Schema(type=genai.protos.Type.STRING)
                                }
                            )
                        }
                    )
                }
            )
        )
    }
)

# Define the schema for significant timestamps
significant_timestamps_schema = genai.protos.Schema(
    type=genai.protos.Type.ARRAY,
    items=genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "time": genai.protos.Schema(type=genai.protos.Type.STRING),
            "event_type": genai.protos.Schema(type=genai.protos.Type.STRING),
            "description": genai.protos.Schema(type=genai.protos.Type.STRING)
        }
    )
)

# Define the schema for emotional_psychological_analysis
emotional_psychological_analysis_schema = genai.protos.Schema(
    type=genai.protos.Type.ARRAY,
    items=genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "speaker": genai.protos.Schema(type=genai.protos.Type.STRING),
            "emotion": genai.protos.Schema(type=genai.protos.Type.STRING),
            "intensity": genai.protos.Schema(type=genai.protos.Type.STRING),
            "timestamp": genai.protos.Schema(type=genai.protos.Type.STRING),
            "context": genai.protos.Schema(type=genai.protos.Type.STRING)
        }
    )
)

# Define the schema for patient_history
patient_history_schema = genai.protos.Schema(
    type=genai.protos.Type.OBJECT,
    properties={
        "past_medical_conditions": genai.protos.Schema(
            type=genai.protos.Type.ARRAY,
            items=genai.protos.Schema(type=genai.protos.Type.STRING)
        ),
        "surgeries": genai.protos.Schema(
            type=genai.protos.Type.ARRAY,
            items=genai.protos.Schema(type=genai.protos.Type.STRING)
        ),
        "allergies": genai.protos.Schema(
            type=genai.protos.Type.ARRAY,
            items=genai.protos.Schema(type=genai.protos.Type.STRING)
        ),
        "family_history": genai.protos.Schema(
            type=genai.protos.Type.ARRAY,
            items=genai.protos.Schema(type=genai.protos.Type.STRING)
        ),
        "social_history": genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "occupation": genai.protos.Schema(type=genai.protos.Type.STRING),
                "living_situation": genai.protos.Schema(type=genai.protos.Type.STRING),
                "habits": genai.protos.Schema(
                    type=genai.protos.Type.ARRAY,
                    items=genai.protos.Schema(type=genai.protos.Type.STRING)
                )
            }
        ),
        "noted_gaps_or_inconsistencies": genai.protos.Schema(
            type=genai.protos.Type.ARRAY,
            items=genai.protos.Schema(type=genai.protos.Type.STRING)
        )
    }
)

# Define the schema for test_results
test_results_schema = genai.protos.Schema(
    type=genai.protos.Type.ARRAY,
    items=genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "test_name": genai.protos.Schema(type=genai.protos.Type.STRING),
            "result": genai.protos.Schema(type=genai.protos.Type.STRING),
            "reference_range": genai.protos.Schema(type=genai.protos.Type.STRING),
            "interpretation": genai.protos.Schema(type=genai.protos.Type.STRING),
            "clinical_significance": genai.protos.Schema(type=genai.protos.Type.STRING),
            "follow_up_recommended": genai.protos.Schema(type=genai.protos.Type.STRING)
        }
    )
)

# Define the schema for keyword_and_topic_analysis
keyword_and_topic_analysis_schema = genai.protos.Schema(
    type=genai.protos.Type.OBJECT,
    properties={
        "keywords": genai.protos.Schema(
            type=genai.protos.Type.ARRAY,
            items=genai.protos.Schema(type=genai.protos.Type.STRING)
        ),
        "main_topics": genai.protos.Schema(
            type=genai.protos.Type.ARRAY,
            items=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "topic": genai.protos.Schema(type=genai.protos.Type.STRING),
                    "importance": genai.protos.Schema(type=genai.protos.Type.STRING),
                    "time_spent": genai.protos.Schema(type=genai.protos.Type.STRING)
                }
            )
        ),
        "topic_flow_assessment": genai.protos.Schema(type=genai.protos.Type.STRING)
    }
)

# Define the schema for quality_of_care_assessment
quality_of_care_assessment_schema = genai.protos.Schema(
    type=genai.protos.Type.OBJECT,
    properties={
        "guideline_adherence": genai.protos.Schema(
            type=genai.protos.Type.ARRAY,
            items=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "guideline": genai.protos.Schema(type=genai.protos.Type.STRING),
                    "adherence_level": genai.protos.Schema(type=genai.protos.Type.STRING),
                    "notes": genai.protos.Schema(type=genai.protos.Type.STRING)
                }
            )
        ),
        "comprehensiveness": genai.protos.Schema(type=genai.protos.Type.STRING),
        "identified_gaps": genai.protos.Schema(
            type=genai.protos.Type.ARRAY,
            items=genai.protos.Schema(type=genai.protos.Type.STRING)
        ),
        "efficiency_assessment": genai.protos.Schema(type=genai.protos.Type.STRING)
    }
)

# Define the schema for overall_assessment
overall_assessment_schema = genai.protos.Schema(
    type=genai.protos.Type.OBJECT,
    properties={
        "key_findings": genai.protos.Schema(
            type=genai.protos.Type.ARRAY,
            items=genai.protos.Schema(type=genai.protos.Type.STRING)
        ),
        "critical_points": genai.protos.Schema(
            type=genai.protos.Type.ARRAY,
            items=genai.protos.Schema(type=genai.protos.Type.STRING)
        ),
        "areas_for_improvement": genai.protos.Schema(
            type=genai.protos.Type.ARRAY,
            items=genai.protos.Schema(type=genai.protos.Type.STRING)
        ),
        "positive_aspects": genai.protos.Schema(
            type=genai.protos.Type.ARRAY,
            items=genai.protos.Schema(type=genai.protos.Type.STRING)
        ),
        "follow_up_recommendations": genai.protos.Schema(
            type=genai.protos.Type.ARRAY,
            items=genai.protos.Schema(type=genai.protos.Type.STRING)
        )
    }
)

# Define the schema for metadata
metadata_schema = genai.protos.Schema(
    type=genai.protos.Type.OBJECT,
    properties={
        "recording_date": genai.protos.Schema(type=genai.protos.Type.STRING),
        "recording_time": genai.protos.Schema(type=genai.protos.Type.STRING),
        "location": genai.protos.Schema(type=genai.protos.Type.STRING),
        "setting": genai.protos.Schema(type=genai.protos.Type.STRING),
        "duration": genai.protos.Schema(type=genai.protos.Type.STRING),
        "visit_type": genai.protos.Schema(type=genai.protos.Type.STRING),
        "technology_used": genai.protos.Schema(
            type=genai.protos.Type.ARRAY,
            items=genai.protos.Schema(type=genai.protos.Type.STRING)
        ),
        "healthcare_providers": genai.protos.Schema(
            type=genai.protos.Type.ARRAY,
            items=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "role": genai.protos.Schema(type=genai.protos.Type.STRING),
                    "specialty": genai.protos.Schema(type=genai.protos.Type.STRING)
                }
            )
        )
    }
)


# Combine all schemas into one
final_output_schema = genai.protos.Schema(
    type=genai.protos.Type.OBJECT,
    properties={
        "transcription": transcription_schema,
        "content_summary": content_summary_schema,
        "participants": participants_schema,
        "significant_timestamps": significant_timestamps_schema,
        "emotional_psychological_analysis": emotional_psychological_analysis_schema,
        "patient_history": patient_history_schema,
        "test_results": test_results_schema,
        "keyword_and_topic_analysis": keyword_and_topic_analysis_schema,
        "quality_of_care_assessment": quality_of_care_assessment_schema,
        "overall_assessment": overall_assessment_schema,
        "metadata": metadata_schema
    }
)
# --- Function Declaration ---
# Define the function declaration
function_declaration = genai.protos.FunctionDeclaration(
    name="process_medical_data",
    description="Processes medical video/audio data and extracts comprehensive structured information.",
    parameters=final_output_schema
)


# --- User Preferences ---
user_preferences = {
  "focus_areas": [
    "patient_history",
    "medication_analysis",
    "emotional_psychological_analysis",
    "treatment_plan",
    "quality_of_care_assessment"
  ],
  "summary_level": "detailed",
  "language": "English",
  "time_range": {
    "start": 0,
    "end": None
  },
  "participant_focus": [
    "patient",
    "doctor"
  ],
  "medical_codes": [
    "ICD-10",
    "SNOMED CT"
  ],
  "sentiment_analysis": True,
  "keyword_extraction": {
    "enabled": True,
    "max_keywords": 10
  },
  "privacy": {
    "level": "high",
    "redact_personal_information": True,
    "hipaa_compliance": True
  },
  "output_formatting": {
    "include_timestamps": True,
    "speaker_diarization": True,
    "segment_by": "topics",
    "include_non_verbal_cues": True,
    "json_indent": 4
  },
  "special_instructions": [
    "Pay extra attention to any mentioned allergies.",
    "Highlight any urgent medical concerns.",
    "Note any discrepancies in patient history.",
    "Identify follow-up recommendations."
  ],
  "error_handling": {
    "log_errors": True,
    "retry_on_failure": True,
    "max_retries": 3
  },
  "translation": {
    "enabled": False,
    "target_language": None
  },
  "cultural_sensitivity": {
    "enabled": True,
    "localization": "US"
  },
  "advanced_settings": {
    "emotional_intensity_threshold": 0.5,
    "confidence_threshold": 0.8,
    "ignore_silence_below_seconds": 2
  }
}

# --- Full Prompt Template ---
full_prompt_template = """
{system_prompt}

{user_prompt}

{assistant_prompt}

Analyze the following chunk (chunk {{chunk_num}} of {{total_chunks}}) of the medical consultation recording.
This is a partial analysis. Focus on extracting information from this specific part of the conversation.
Provide the output in JSON format, structured as per the guidelines.

<chunk_content>
{{chunk_content}}
</chunk_content>
"""

# --- Model Initialization ---
model = genai.GenerativeModel(
    model_name='gemini-1.5-pro-exp-0827',
    tools=[function_declaration],
    system_instruction=system_prompt
)
chat = model.start_chat(enable_automatic_function_calling=True)

# --- Chunk Processing Functions ---

def process_chunk(chunk, chunk_num, total_chunks):
    audio_chunk_path = f"chunk_{chunk_num}.wav"
    try:
        chunk.export(audio_chunk_path, format="wav")
        
        # Check for silence
        if is_silence(chunk):
            logger.info(f"Chunk {chunk_num} is silent, skipping processing")
            return None

        your_file = genai.upload_file(audio_chunk_path)

        # Wait for file processing with exponential backoff
        retries = 0
        max_retries = 5
        while your_file.state.name == "PROCESSING":
            if retries > max_retries:
                raise TimeoutError(f"Processing of chunk {chunk_num} timed out.")
            delay = 5 * (2 ** retries)
            logger.info(f"Processing chunk {chunk_num} of {total_chunks}... retrying in {delay} seconds")
            time.sleep(delay)
            your_file = genai.get_file(your_file.name)
            retries += 1

        # --- EXTRACT TRANSCRIBED TEXT HERE ---
        # Example (you'll need to replace this with your actual transcription logic):
        # Assuming you have a function transcribe_audio(audio_file_path) that returns the text 
        video_content = f"This is the transcribed text from audio chunk {chunk_num} of {total_chunks}. " #transcribe_audio(audio_chunk_path) 
        user_prompt_filled = user_prompt.format(VIDEO_CONTENT=video_content, USER_PREFERENCES=json.dumps(user_preferences, indent=2))

        chunk_description = f"Processing audio chunk {chunk_num} of {total_chunks}"
        print(f"Chunk Description: {chunk_description}")
        chunk_prompt = full_prompt_template.format(
            system_prompt=system_prompt,
            user_prompt=user_prompt_filled,
            assistant_prompt=assistant_prompt,
            chunk_num=chunk_num,
            total_chunks=total_chunks,
            chunk_content=chunk_description
            # chunk_content=f"Processing audio chunk {chunk_num} of {total_chunks}"
        )

        response = chat.send_message([chunk_prompt, your_file])
        
        # After getting the raw API response
        # print(f"Raw API Response: {json.dumps(raw_api_response, indent=2)}")
        # # Before processing each key in the response
        # for key in raw_api_response:
        #     print(f"Processing key: {key}")
        #     print(f"Value: {raw_api_response[key]}")
        # Print the raw output for debugging
        # print(f"Raw API Response: {response}")
        # Extract data from the GenerateContentResponse object
        # chunk_data = response.text
        # Extract JSON data from the response
        chunk_data = extract_json_from_response(response)
        # Print the raw output for debugging
        print(json.dumps(chunk_data, indent=2))
        
        
        print(f"Raw API Response: {chunk_data}") # Print raw response
        
        # Assuming Gemini returns the JSON as text in the response:
        # chunk_data = json.loads(response.candidates[0].content.parts[0].text)
        
        # logger.info(f"Successfully processed chunk {chunk_num} of {total_chunks}")
        # return chunk_data
        if chunk_data:
            # Add a confidence check
            if check_confidence(chunk_data):
                logger.info(f"Successfully processed chunk {chunk_num} of {total_chunks}")
                return chunk_data
            else:
                logger.warning(f"Low confidence in chunk {chunk_num} output, discarding")
                return None
        else:
            logger.error(f"Failed to extract valid JSON data from response for chunk {chunk_num}")
            return None
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network-related error on chunk {chunk_num}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error on chunk {chunk_num}: {str(e)}")
        return None
    # finally:
    #     if os.path.exists(audio_chunk_path):
    #         os.remove(audio_chunk_path)

def extract_json_from_response(response):
    if response.candidates and len(response.candidates) > 0:
        candidate = response.candidates[0]
        if candidate.content and candidate.content.parts:
            json_text = candidate.content.parts[0].text
            # Remove the markdown code block indicators if present
            json_text = json_text.strip('`').replace('json\n', '', 1)
            try:
                return json.loads(json_text)
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from response")
                logger.error(f"Raw response text: {json_text}")
    logger.error("Invalid response structure")
    logger.error(f"Full response: {response}")
    return None

def is_silence(chunk, silence_threshold=-50.0, min_silence_duration=1000):
    """Check if an audio chunk is silent."""
    return chunk.dBFS < silence_threshold and len(chunk) >= min_silence_duration

def check_confidence(chunk_data, confidence_threshold=0.7):
    """Check if the chunk data meets a confidence threshold."""
    # This is a placeholder function. You'll need to implement your own confidence checking logic
    # based on the structure of your chunk_data and what you consider to be "confident" output.
    return True  # For now, always return True

def main(process_all=True, test_chunk_index=None, process_full_audio=False):
    try:
        # --- Load Audio From File ---
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.endswith(('.mp4', '.mkv', '.avi')):
            logger.info(f"Processing video file: {file_path}")
            clip = VideoFileClip(file_path)
            audio_clip = clip.audio
            audio_clip.write_audiofile("temp_audio.wav")
            audio = AudioSegment.from_wav("temp_audio.wav")
            logger.info(f"Extracted audio from video: {file_path}")
        else:
            logger.info(f"Processing audio file: {file_path}")
            audio = AudioSegment.from_file(file_path)

        # --- Split Audio into Chunks ---
        chunk_length_ms = 5 * 60 * 1000  # 5 minutes
        chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
        total_chunks = len(chunks)

        if process_full_audio:
            # Process the full audio file
            logger.info("Processing full audio file")
            chunk_data = process_chunk(audio, 1, 1)
            if chunk_data and validate_chunk_data(chunk_data):
                with open("full_audio_output.json", "w") as outfile:
                    json.dump(chunk_data, outfile, indent=2)
                logger.info("Full audio processed. Output saved to 'full_audio_output.json'.")
            else:
                logger.error("Failed to process full audio")
        elif process_all:
            # Process all chunks
            final_output = initialize_final_output()
            for i, chunk in enumerate(chunks, 1):
                chunk_data = process_chunk(chunk, i, total_chunks)
                if chunk_data and validate_chunk_data(chunk_data):
                    try:
                        combine_chunk_data_with_final_output(chunk_data, final_output)
                        logger.info(f"Processed chunk {i}/{total_chunks}")
                        # Save each chunk output after processing
                        with open(f"chunk_{i}_output.json", "w") as outfile:
                            json.dump(chunk_data, outfile, indent=2)
                    except TypeError as e:
                        logger.error(f"TypeError when processing chunk {i}: {str(e)}")
                        logger.error(f"Problematic chunk_data: {json.dumps(chunk_data, indent=2)}")
                else:
                    logger.warning(f"Invalid or empty data for chunk {i}/{total_chunks}")

            consolidated_output = consolidate_final_output(final_output)
            with open("final_consolidated_output.json", "w") as outfile:
                json.dump(consolidated_output, outfile, indent=2)
            logger.info("Processing complete. Final consolidated output saved to 'final_consolidated_output.json'.")

        elif test_chunk_index is not None:
            # Process a single chunk for testing
            if 0 <= test_chunk_index < total_chunks:
                chunk_data = process_chunk(chunks[test_chunk_index], test_chunk_index + 1, total_chunks)
                if chunk_data:
                    with open(f"chunk_{test_chunk_index + 1}_output.json", "w") as outfile:
                        json.dump(chunk_data, outfile, indent=2)
                    logger.info(f"Test chunk {test_chunk_index + 1} processed. Output saved to 'chunk_{test_chunk_index + 1}_output.json'.")
                else:
                    logger.error(f"Failed to process test chunk {test_chunk_index + 1}")
            else:
                logger.error(f"Invalid test chunk index. Must be between 0 and {total_chunks - 1}")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        logger.error(f"Error details: {traceback.format_exc()}")
        raise e

    # finally:
    #     # Clean up temporary files
    #     if os.path.exists("temp_audio.wav"):
    #         os.remove("temp_audio.wav")

def initialize_final_output():
    return {
        "transcription": {
            "full_text": "",
            "timestamps": []
        },
        "content_summary": {
            "main_points": "",
            "interaction_purpose": ""
        },
        "participants": {
            "patients": [],
            "doctors": [],
            "others": []
        },
        "significant_timestamps": [],
        "emotional_psychological_analysis": [],
        "patient_history": {
            "past_medical_conditions": [],
            "surgeries": [],
            "allergies": [],
            "family_history": [],
            "social_history": {
                "occupation": "",
                "living_situation": "",
                "habits": []
            },
            "noted_gaps_or_inconsistencies": []
        },
        "test_results": [],
        "keyword_and_topic_analysis": {
            "keywords": [],
            "main_topics": [],
            "topic_flow_assessment": ""
        },
        "quality_of_care_assessment": {
            "guideline_adherence": [],
            "comprehensiveness": "",
            "identified_gaps": [],
            "efficiency_assessment": ""
        },
        "overall_assessment": {
            "key_findings": [],
            "critical_points": [],
            "areas_for_improvement": [],
            "positive_aspects": [],
            "follow_up_recommendations": []
        },
        "metadata": {
            "recording_date": "",
            "recording_time": "",
            "location": "",
            "setting": "",
            "duration": "",
            "visit_type": "",
            "technology_used": [],
            "healthcare_providers": []
        }
    }

def validate_chunk_data(chunk_data):
    required_keys = ["transcription", "content_summary", "participants"]
    for key in required_keys:
        if key not in chunk_data or not chunk_data[key]:
            logger.warning(f"Missing or empty expected key '{key}' in chunk data")
            return False
    return True

def combine_chunk_data_with_final_output(chunk_data, final_output):
    for key in chunk_data:
        if key not in final_output:
            final_output[key] = chunk_data[key]
        elif isinstance(final_output[key], list):
            final_output[key].extend(chunk_data[key] if chunk_data[key] is not None else [])
        elif isinstance(final_output[key], dict):
            final_output[key].update(chunk_data[key] if chunk_data[key] is not None else {})
        elif isinstance(final_output[key], str):
            final_output[key] += str(chunk_data[key] if chunk_data[key] is not None else "")
        else:
            logger.warning(f"Unhandled data type for key {key}: {type(final_output[key])}")

    final_output["transcription"]["timestamps"].extend(chunk_data.get("transcription", {}).get("timestamps", []))
    final_output["transcription"]["full_text"] += chunk_data.get("transcription", {}).get("full_text", "")
    final_output["content_summary"]["main_points"] += chunk_data.get("content_summary", {}).get("main_points", "")
    final_output["content_summary"]["interaction_purpose"] += chunk_data.get("content_summary", {}).get("interaction_purpose", "")
    final_output["participants"]["patients"].extend(chunk_data.get("participants", {}).get("patients", []))
    final_output["participants"]["doctors"].extend(chunk_data.get("participants", {}).get("doctors", []))
    final_output["participants"]["others"].extend(chunk_data.get("participants", {}).get("others", []))
    final_output["significant_timestamps"].extend(chunk_data.get("significant_timestamps", []       ))
    final_output["emotional_psychological_analysis"].extend(chunk_data.get("emotional_psychological_analysis", []))
    final_output["patient_history"]["past_medical_conditions"].extend(chunk_data.get("patient_history", {}).get("past_medical_conditions", []))
    final_output["patient_history"]["surgeries"].extend(chunk_data.get("patient_history", {}).get("surgeries", []))
    # Initialize allergies as a list if it doesn't exist or is a string
    if "patient_history" not in final_output:
        final_output["patient_history"] = {}
    if "allergies" not in final_output["patient_history"] or isinstance(final_output["patient_history"]["allergies"], str):
        final_output["patient_history"]["allergies"] = []
    final_output["patient_history"]["allergies"].extend(chunk_data.get("patient_history", {}).get("allergies", []))
    final_output["patient_history"]["family_history"].extend(chunk_data.get("patient_history", {}).get("family_history", []))
    final_output["patient_history"]["social_history"]["habits"].extend(chunk_data.get("patient_history", {}).get("social_history", {}).get("habits", []))
    final_output["patient_history"]["social_history"]["occupation"] += chunk_data.get("patient_history", {}).get("social_history", {}).get("occupation", "")
    final_output["patient_history"]["social_history"]["living_situation"] += chunk_data.get("patient_history", {}).get("social_history", {}).get("living_situation", "")
    final_output["patient_history"]["noted_gaps_or_inconsistencies"].extend(chunk_data.get("patient_history", {}).get("noted_gaps_or_inconsistencies", []))
    final_output["test_results"].extend(chunk_data.get("test_results", []))
    final_output["keyword_and_topic_analysis"]["keywords"].extend(chunk_data.get("keyword_and_topic_analysis", {}).get("keywords", []))
    final_output["keyword_and_topic_analysis"]["main_topics"].extend(chunk_data.get("keyword_and_topic_analysis", {}).get("main_topics", []))
    final_output["keyword_and_topic_analysis"]["topic_flow_assessment"] += chunk_data.get("keyword_and_topic_analysis", {}).get("topic_flow_assessment", "")
    final_output["quality_of_care_assessment"]["guideline_adherence"].extend(chunk_data.get("quality_of_care_assessment", {}).get("guideline_adherence", []))
    final_output["quality_of_care_assessment"]["comprehensiveness"] += chunk_data.get("quality_of_care_assessment", {}).get("comprehensiveness", "")
    final_output["quality_of_care_assessment"]["identified_gaps"].extend(chunk_data.get("quality_of_care_assessment", {}).get("identified_gaps", []))
    final_output["quality_of_care_assessment"]["efficiency_assessment"] += chunk_data.get("quality_of_care_assessment", {}).get("efficiency_assessment", "")
    final_output["overall_assessment"]["key_findings"].extend(chunk_data.get("overall_assessment", {}).get("key_findings", []))
    final_output["overall_assessment"]["critical_points"].extend(chunk_data.get("overall_assessment", {}).get("critical_points", []))
    final_output["overall_assessment"]["areas_for_improvement"].extend(chunk_data.get("overall_assessment", {}).get("areas_for_improvement", []))
    final_output["overall_assessment"]["positive_aspects"].extend(chunk_data.get("overall_assessment", {}).get("positive_aspects", []))
    final_output["overall_assessment"]["follow_up_recommendations"].extend(chunk_data.get("overall_assessment", {}).get("follow_up_recommendations", []))
    final_output["metadata"]["technology_used"].extend(chunk_data.get("metadata", {}).get("technology_used", []))
    final_output["metadata"]["healthcare_providers"].extend(chunk_data.get("metadata", {}).get("healthcare_providers", []))


def consolidate_final_output(final_output):
    # Helper function to remove duplicates from a list while preserving order
    def deduplicate(seq):
        seen = set()
        result = []
        for item in seq:
            if item is None:
                continue
            if isinstance(item, dict):
                item_tuple = tuple(sorted(item.items()))
                if item_tuple not in seen:
                    seen.add(item_tuple)
                    result.append(item)
            elif item not in seen:
                seen.add(item)
                result.append(item)
        return result

    # Helper function to merge lists, handling different data types
    def merge_lists(list1, list2):
        return deduplicate(list1 + list2)

    # Helper function to merge dictionaries
    def merge_dicts(dict1, dict2):
        for key, value in dict2.items():
            if key in dict1:
                if isinstance(value, list):
                    dict1[key] = merge_lists(dict1[key], value)
                elif isinstance(value, dict):
                    dict1[key] = merge_dicts(dict1[key], value)
                elif isinstance(value, str):
                    dict1[key] = dict1[key].strip() + " " + value.strip()
                else:
                    dict1[key] = value
            else:
                dict1[key] = value
        return dict1

    # Consolidate transcription
    final_output["transcription"]["timestamps"] = deduplicate(final_output["transcription"]["timestamps"])
    final_output["transcription"]["full_text"] = " ".join(final_output["transcription"]["full_text"].split())

    # Consolidate content_summary
    final_output["content_summary"]["main_points"] = " ".join(final_output["content_summary"]["main_points"].split())
    final_output["content_summary"]["interaction_purpose"] = " ".join(final_output["content_summary"]["interaction_purpose"].split())

    # Consolidate participants
    for participant_type in ["patients", "doctors", "others"]:
        consolidated = {}
        for participant in final_output["participants"][participant_type]:
            name = participant["name"]
            if name not in consolidated:
                consolidated[name] = participant
            else:
                consolidated[name] = merge_dicts(consolidated[name], participant)
        final_output["participants"][participant_type] = list(consolidated.values())

    # Consolidate significant_timestamps
    final_output["significant_timestamps"] = deduplicate(final_output["significant_timestamps"])

    # Consolidate emotional_psychological_analysis
    final_output["emotional_psychological_analysis"] = deduplicate(final_output["emotional_psychological_analysis"])

    # Consolidate patient_history
    for key in ["past_medical_conditions", "surgeries", "allergies", "family_history"]:
        final_output["patient_history"][key] = deduplicate(final_output["patient_history"][key])
    final_output["patient_history"]["social_history"]["habits"] = deduplicate(final_output["patient_history"]["social_history"]["habits"])
    final_output["patient_history"]["social_history"]["occupation"] = " ".join(final_output["patient_history"]["social_history"]["occupation"].split())
    final_output["patient_history"]["social_history"]["living_situation"] = " ".join(final_output["patient_history"]["social_history"]["living_situation"].split())
    final_output["patient_history"]["noted_gaps_or_inconsistencies"] = deduplicate(final_output["patient_history"]["noted_gaps_or_inconsistencies"])

    # Consolidate test_results
    final_output["test_results"] = deduplicate(final_output["test_results"])

    # Consolidate keyword_and_topic_analysis
    final_output["keyword_and_topic_analysis"]["keywords"] = deduplicate(final_output["keyword_and_topic_analysis"]["keywords"])
    final_output["keyword_and_topic_analysis"]["main_topics"] = merge_lists(
        final_output["keyword_and_topic_analysis"].get("main_topics", []), [])
    final_output["keyword_and_topic_analysis"]["topic_flow_assessment"] = " ".join(final_output["keyword_and_topic_analysis"]["topic_flow_assessment"].split())

    # Consolidate quality_of_care_assessment
    final_output["quality_of_care_assessment"]["guideline_adherence"] = deduplicate(final_output["quality_of_care_assessment"]["guideline_adherence"])
    final_output["quality_of_care_assessment"]["comprehensiveness"] = " ".join(final_output["quality_of_care_assessment"]["comprehensiveness"].split())
    final_output["quality_of_care_assessment"]["identified_gaps"] = deduplicate(final_output["quality_of_care_assessment"]["identified_gaps"])
    final_output["quality_of_care_assessment"]["efficiency_assessment"] = " ".join(final_output["quality_of_care_assessment"]["efficiency_assessment"].split())

    # Consolidate overall_assessment
    for key in ["key_findings", "critical_points", "areas_for_improvement", "positive_aspects", "follow_up_recommendations"]:
        final_output["overall_assessment"][key] = deduplicate(final_output["overall_assessment"][key])

    # Consolidate metadata
    final_output["metadata"]["technology_used"] = deduplicate(final_output["metadata"]["technology_used"])
    final_output["metadata"]["healthcare_providers"] = merge_lists(
        final_output["metadata"].get("healthcare_providers", []), [])

    return final_output

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            main(process_all=False, test_chunk_index=int(sys.argv[2]))
        elif sys.argv[1] == "full":
            main(process_all=False, test_chunk_index=None, process_full_audio=True)
        else:
            print("Invalid argument. Use 'test <chunk_index>' or 'full' for processing options.")
    else:
        main(process_all=True)
