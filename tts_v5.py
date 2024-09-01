import os
import google.generativeai as genai
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import logging
import requests
import time
import json

# Set your API key here
os.environ["GEMINI_API_KEY"] = "AIzaSyCnEXqCUlsklCAvdVn5XUTBMjp4mBlSXHA"
genai.configure(api_key=os.environ["GEMINI_API_KEY"])


# # Specify the path to your video file in Google Drive
# file_path = '/content/drive/MyDrive/Chiranjeevi Video Dec 21.mp4'
file_path = '/Users/pranay/Projects/LLM/video/proj1/data/Chiranjeevi_Video_Dec_21.mp4'  # Update this path to your local file

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to flush all log handlers
def flush_log_handlers(logger):
    for handler in logger.handlers:
        handler.flush()


# Initialize the model with system instructions (system prompt)
system_prompt = """
You are an advanced AI assistant specialized in analyzing medical or health-related video/audio content. Your primary function is to extract, organize, and analyze various types of information from the provided content, focusing on medical details, patient interactions, and healthcare processes. You possess extensive knowledge of medical terminology, procedures, best practices in healthcare communication, and the latest advancements in medical science.

Your task is to analyze the given video/audio content and generate a comprehensive report in JSON format. This report will be used to power a frontend application, enabling users to explore summaries and conduct deep dives into detailed analyses based on specific individuals and key elements mentioned in the content.
"""


# Define the user prompt
user_prompt = """
Please analyze the following medical consultation recording. Ensure you capture all patient names, doctor names, and any other participant names. Organize the information so that users can easily explore summaries or deep-dive into each individual's details. Provide the output in JSON format, structured as per the guidelines.

<video_content>
{VIDEO_CONTENT}
</video_content>

<user_preferences>
{USER_PREFERENCES}
</user_preferences>
"""

# Define the assistant prompt
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

# Define the function declaration
function_declaration = genai.protos.FunctionDeclaration(
    name="process_medical_data",
    description="Processes medical video/audio data and extracts comprehensive structured information.",
    parameters=final_output_schema
)

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

final_output = {
    "transcription": {
        "full_text": "",
        "timestamps": [
            {
                "time": "",
                "text": "",
                "speaker": "",
                "non_verbal": ""
            }
        ]
    },
    "content_summary": {
        "main_points": "",
        "interaction_purpose": ""
    },
    "participants": {
        "patients": [
            {
                "name": "",
                "role": "Patient",
                "summary": "",
                "details": {
                    "symptoms": [
                        {
                            "term": "",
                            "standardized_term": "",
                            "icd_10_code": "",
                            "snomed_ct_code": "",
                            "severity": "",
                            "duration": "",
                            "frequency": "",
                            "context": ""
                        }
                    ],
                    "diagnoses": [
                        {
                            "diagnosis": "",
                            "icd_10_code": "",
                            "snomed_ct_code": "",
                            "status": "",
                            "certainty_level": "",
                            "basis": "",
                            "context": ""
                        }
                    ],
                    "medications": [
                        {
                            "name": "",
                            "rxnorm_code": "",
                            "status": "",
                            "dosage": "",
                            "frequency": "",
                            "route": "",
                            "context": ""
                        }
                    ],
                    "treatment_plans": [
                        {
                            "condition": "",
                            "plan": "",
                            "rationale": "",
                            "risks": "",
                            "benefits": "",
                            "alternatives_discussed": "",
                            "patient_involvement": ""
                        }
                    ]
                }
            }
        ],
        "doctors": [
            {
                "name": "",
                "role": "Doctor",
                "summary": "",
                "details": {
                    "speaking_patterns": {
                        "talk_time": 0.0,
                        "interruptions": 0,
                        "engagement_level": ""
                    },
                    "diagnoses": [
                        {
                            "diagnosis": "",
                            "icd_10_code": "",
                            "snomed_ct_code": "",
                            "status": "",
                            "certainty_level": "",
                            "basis": "",
                            "context": ""
                        }
                    ],
                    "treatment_plans": [
                        {
                            "condition": "",
                            "plan": "",
                            "rationale": "",
                            "risks": "",
                            "benefits": "",
                            "alternatives_discussed": "",
                            "patient_involvement": ""
                        }
                    ]
                }
            }
        ],
        "others": [
            {
                "name": "",
                "role": "",
                "summary": "",
                "details": {
                    "speaking_patterns": {
                        "talk_time": 0.0,
                        "interruptions": 0,
                        "engagement_level": ""
                    }
                }
            }
        ]
    },
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
        "past_medical_conditions": [""],
        "surgeries": [""],
        "allergies": [""],
        "family_history": [""],
        "social_history": {
            "occupation": "",
            "living_situation": "",
            "habits": [""]
        },
        "noted_gaps_or_inconsistencies": [""]
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
    "keyword_and_topic_analysis": {
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
        "guideline_adherence": [
            {
                "guideline": "",
                "adherence_level": "",
                "notes": ""
            }
        ],
        "comprehensiveness": "",
        "identified_gaps": [""],
        "efficiency_assessment": ""
    },
    "overall_assessment": {
        "key_findings": [""],
        "critical_points": [""],
        "areas_for_improvement": [""],
        "positive_aspects": [""],
        "follow_up_recommendations": [""]
    },
    "metadata": {
        "recording_date": "",
        "recording_time": "",
        "location": "",
        "setting": "",
        "duration": "",
        "visit_type": "",
        "technology_used": [""],
        "healthcare_providers": [
            {
                "role": "",
                "specialty": ""
            }
        ]
    }
}



# full_prompt = system_prompt + user_prompt + assistant_prompt
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

# Initialize the model with the function declaration
model = genai.GenerativeModel(
    model_name='gemini-1.5-pro-exp-0827',
    tools=[function_declaration],
    system_instruction=system_prompt
)


# Begin a chat session for function calling
chat = model.start_chat(enable_automatic_function_calling=True)
# Function to call Gemini API with a single chunk
# Function to call Gemini API with a single chunk
def process_single_chunk(chunk, chunk_num):
    audio_chunk_path = f"chunk_{chunk_num}.wav"
    try:
        # Save the chunk to a file and send it to the Gemini API
        chunk.export(audio_chunk_path, format="wav")
        your_file = genai.upload_file(audio_chunk_path)

        # Wait for the file to be processed with exponential backoff
        retries = 0
        max_retries = 5
        while your_file.state.name == "PROCESSING":
            if retries > max_retries:
                raise TimeoutError(f"Processing of chunk {chunk_num} timed out.")
            delay = 5 * (2 ** retries)
            logger.info(f"Processing chunk {chunk_num}... retrying in {delay} seconds")
            time.sleep(delay)
            your_file = genai.get_file(your_file.name)
            retries += 1

        # Replace VIDEO_CONTENT with actual content
        video_content = "Your transcribed video content here"  # Replace with actual content extraction logic
        user_prompt_filled = user_prompt.format(VIDEO_CONTENT=video_content, USER_PREFERENCES=json.dumps(user_preferences, indent=2))

        # Generate the chunk-specific prompt with a meaningful description
        chunk_description = f"Processing audio chunk {chunk_num}"  # Modify as needed
        chunk_prompt = full_prompt_template.format(
            system_prompt=system_prompt,
            user_prompt=user_prompt_filled,
            assistant_prompt=assistant_prompt,
            chunk_num=chunk_num,
            total_chunks=1,
            chunk_content=chunk_description  # Meaningful content description
        )

        # Generate content by calling the chat session
        response = chat.send_message([chunk_prompt, your_file])

        # Extract data from the GenerateContentResponse object
        chunk_data = response.to_dict()  # Convert response to a dictionary

        # Print the raw output for debugging
        print(json.dumps(chunk_data, indent=2))

    except requests.exceptions.RequestException as e:
        logger.error(f"Network-related error on chunk {chunk_num}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error on chunk {chunk_num}: {str(e)}")
    finally:
        if os.path.exists(audio_chunk_path):
            os.remove(audio_chunk_path)


# # Example: Assuming file_path is already defined
# file_path = '/content/drive/MyDrive/Chiranjeevi Video Dec 21.mp4'  # Ensure this is correctly set

# This code block is needed to handle file processing and audio extraction
try:
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        flush_log_handlers(logger)
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.lower().endswith(('.mp4', '.mkv', '.avi')):
        logger.info(f"Processing video file: {file_path}")
        flush_log_handlers(logger)
        clip = VideoFileClip(file_path)
        audio_clip = clip.audio
        audio_clip.write_audiofile("temp_audio.wav")
        audio = AudioSegment.from_wav("temp_audio.wav")
        logger.info(f"Extracted audio from video: {file_path}")
        flush_log_handlers(logger)
    else:
        logger.info(f"Processing audio file: {file_path}")
        flush_log_handlers(logger)
        audio = AudioSegment.from_file(file_path)

except Exception as e:
    logger.error(f"Error processing file {file_path}: {str(e)}")
    flush_log_handlers(logger)
    raise

# Step 3: Split the audio into chunks (5 minutes each)
chunk_length_ms = 5 * 60 * 1000  # 5 minutes
chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

print(chunks)

# Process a single chunk for debugging
process_single_chunk(chunks[2], 1)

# Verify chunk durations
for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1} duration: {len(chunk) / 1000} seconds")



# Invoke the model with the prompts
response = chat.send_message(user_prompt + assistant_prompt)


# Function to validate the chunk data before processing
def validate_chunk_data(chunk_data):
    required_keys = ["transcription", "content_summary", "participants"]
    for key in required_keys:
        if key not in chunk_data or not chunk_data[key]:
            logger.warning(f"Missing or empty expected key '{key}' in chunk data")
            return False
    return True

# Function to combine chunk data with the final output
def combine_chunk_data_with_final_output(chunk_data, final_output):
    final_output["transcription"]["timestamps"].extend(chunk_data.get("transcription", {}).get("timestamps", []))
    final_output["transcription"]["full_text"] += chunk_data.get("transcription", {}).get("full_text", "")
    final_output["content_summary"]["main_points"] += chunk_data.get("content_summary", {}).get("main_points", "")
    final_output["content_summary"]["interaction_purpose"] += chunk_data.get("content_summary", {}).get("interaction_purpose", "")
    final_output["participants"]["patients"].extend(chunk_data.get("participants", {}).get("patients", []))
    final_output["participants"]["doctors"].extend(chunk_data.get("participants", {}).get("doctors", []))
    final_output["participants"]["others"].extend(chunk_data.get("participants", {}).get("others", []))
    final_output["significant_timestamps"].extend(chunk_data.get("significant_timestamps", []))
    final_output["emotional_psychological_analysis"].extend(chunk_data.get("emotional_psychological_analysis", []))
    final_output["patient_history"]["past_medical_conditions"].extend(chunk_data.get("patient_history", {}).get("past_medical_conditions", []))
    final_output["patient_history"]["surgeries"].extend(chunk_data.get("patient_history", {}).get("surgeries", []))
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


logger.info(f"Total duration of audio: {len(audio)} ms")
logger.info(f"Expected number of chunks: {len(chunks)}")
for idx, chunk in enumerate(chunks):
    logger.info(f"Chunk {idx + 1} duration: {len(chunk)} ms")

# Step 4: Check that the combined duration of chunks matches the original audio length
combined_duration = sum(len(chunk) for chunk in chunks)
if combined_duration != len(audio):
    logger.error(f"Combined chunk duration ({combined_duration} ms) does not match original audio duration ({len(audio)} ms)")
else:
    logger.info("Combined chunk duration matches original audio duration.")


def process_chunk(chunk, chunk_num, total_chunks, final_output):
    audio_chunk_path = f"chunk_{chunk_num}.wav"
    try:
        # Save the chunk to a file and send it to the Gemini API
        chunk.export(audio_chunk_path, format="wav")
        your_file = genai.upload_file(audio_chunk_path)

        # Wait for the file to be processed with exponential backoff
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

        # Replace VIDEO_CONTENT with actual content
        video_content = "Your transcribed video content here"  # Replace with actual content extraction logic
        user_prompt_filled = user_prompt.format(VIDEO_CONTENT=video_content, USER_PREFERENCES=json.dumps(user_preferences, indent=2))

        # Generate the chunk-specific prompt with a meaningful description
        chunk_description = f"Processing audio chunk {chunk_num} of {total_chunks}"  # Modify as needed
        chunk_prompt = full_prompt_template.format(
            system_prompt=system_prompt,
            user_prompt=user_prompt_filled,
            assistant_prompt=assistant_prompt,
            chunk_num=chunk_num,
            total_chunks=total_chunks,
            chunk_content=chunk_description  # Meaningful content description
        )

        # Generate content by calling the chat session
        response = chat.send_message([chunk_prompt, your_file])

        # Extract data from the GenerateContentResponse object
        chunk_data = response.to_dict()  # Convert response to a dictionary

        if validate_chunk_data(chunk_data):
            combine_chunk_data_with_final_output(chunk_data, final_output)
        else:
            logger.warning(f"Skipping invalid chunk data for chunk {chunk_num}")

        logger.info(f"Successfully processed chunk {chunk_num} of {total_chunks}")

    except requests.exceptions.RequestException as e:
        logger.error(f"Network-related error on chunk {chunk_num}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error on chunk {chunk_num}: {str(e)}")
    finally:
        if os.path.exists(audio_chunk_path):
            os.remove(audio_chunk_path)


# Process all chunks
total_chunks = len(chunks)
final_output = {
    "transcription": {"timestamps": [], "full_text": ""},
    "content_summary": {"main_points": "", "interaction_purpose": ""},
    "participants": {"patients": [], "doctors": [], "others": []},
    "significant_timestamps": [],
    "emotional_psychological_analysis": [],
    "patient_history": {
        "past_medical_conditions": [],
        "surgeries": [],
        "allergies": [],
        "family_history": [],
        "social_history": {"habits": [], "occupation": "", "living_situation": ""},
        "noted_gaps_or_inconsistencies": []
    },
    "test_results": [],
    "keyword_and_topic_analysis": {"keywords": [], "main_topics": [], "topic_flow_assessment": ""},
    "quality_of_care_assessment": {"guideline_adherence": [], "comprehensiveness": "", "identified_gaps": [], "efficiency_assessment": ""},
    "overall_assessment": {
        "key_findings": [],
        "critical_points": [],
        "areas_for_improvement": [],
        "positive_aspects": [],
        "follow_up_recommendations": []
    },
    "metadata": {"technology_used": [], "healthcare_providers": []}
}

def consolidate_final_output(final_output):
    # Helper function to remove duplicates from a list while preserving order
    def deduplicate(seq):
        seen = set()
        return [x for x in seq if not (x in seen or seen.add(x))]

    # Helper function to merge lists, handling different data types
    def merge_lists(list1, list2):
        if all(isinstance(i, dict) for i in list1 + list2):
            combined = {frozenset(item.items()): item for item in list1}
            for item in list2:
                combined[frozenset(item.items())] = item
            return list(combined.values())
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

# Save the final combined output
final_output = consolidate_final_output(final_output)

# Then save the consolidated output
with open("final_consolidated_output.json", "w") as outfile:
    json.dump(final_output, outfile, indent=2)

logger.info("Processing complete. Final consolidated output saved to 'final_consolidated_output.json'.")

