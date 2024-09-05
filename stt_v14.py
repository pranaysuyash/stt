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
import google.generativeai as genai
import tempfile
from google.api_core import retry
import matplotlib.pyplot as plt
from scipy.io import wavfile

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
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=api_key)

# Detect the available device for WhisperX
device = "cpu"  # For macOS and MPS, we will run everything on the CPU
compute_type = "int8"  # Explicitly set to int8 for compatibility
logger.info(f"Running on {device} with {compute_type} compute type.")


# Load WhisperX Model
whisper_model = whisperx.load_model("large-v2", device, compute_type=compute_type)
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

# Define the schema for content summary
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

# Define the schema for emotional/psychological analysis
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

# Define the schema for patient history
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

# Define the schema for test results
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

# Define the schema for keyword and topic analysis
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
# Define the schema for quality of care assessment
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
# Define the schema for overall assessment
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
# Define the schema for additional analysis
additional_analysis_schema = genai.protos.Schema(
    type=genai.protos.Type.OBJECT,
    properties={
        "additional_insights": genai.protos.Schema(
            type=genai.protos.Type.STRING,
            description="Any other important findings or insights not covered by the existing schema."
        )
    }
)

doctor_patient_interaction_schema = genai.protos.Schema(
    type=genai.protos.Type.OBJECT,
    properties={
        "doctor_speaking_time": genai.protos.Schema(type=genai.protos.Type.NUMBER),
        "patient_speaking_time": genai.protos.Schema(type=genai.protos.Type.NUMBER),
        "interruptions": genai.protos.Schema(type=genai.protos.Type.NUMBER),
        "empathy_expressions": genai.protos.Schema(type=genai.protos.Type.STRING),
        "overall_patient_satisfaction": genai.protos.Schema(type=genai.protos.Type.STRING)
    }
)
risk_stratification_schema = genai.protos.Schema(
    type=genai.protos.Type.OBJECT,
    properties={
        "risk_category": genai.protos.Schema(type=genai.protos.Type.STRING),  # e.g., Low, Moderate, High
        "underlying_conditions": genai.protos.Schema(type=genai.protos.Type.ARRAY, items=genai.protos.Schema(type=genai.protos.Type.STRING)),
        "risk_factors": genai.protos.Schema(type=genai.protos.Type.ARRAY, items=genai.protos.Schema(type=genai.protos.Type.STRING)),
        "recommendations": genai.protos.Schema(type=genai.protos.Type.STRING)
    }
)
diagnostic_imaging_analysis_schema = genai.protos.Schema(
    type=genai.protos.Type.OBJECT,
    properties={
        "image_type": genai.protos.Schema(type=genai.protos.Type.STRING),  # X-ray, MRI, etc.
        "findings": genai.protos.Schema(type=genai.protos.Type.ARRAY, items=genai.protos.Schema(type=genai.protos.Type.STRING)),
        "follow_up_tests": genai.protos.Schema(type=genai.protos.Type.STRING),
        "implications_for_treatment": genai.protos.Schema(type=genai.protos.Type.STRING)
    }
)
treatment_follow_up_schema = genai.protos.Schema(
    type=genai.protos.Type.OBJECT,
    properties={
        "treatment_followed": genai.protos.Schema(type=genai.protos.Type.BOOLEAN),
        "adherence_level": genai.protos.Schema(type=genai.protos.Type.STRING),  # e.g., Fully, Partially, Not at all
        "outcomes_observed": genai.protos.Schema(type=genai.protos.Type.STRING),
        "side_effects": genai.protos.Schema(type=genai.protos.Type.ARRAY, items=genai.protos.Schema(type=genai.protos.Type.STRING)),
        "additional_treatments_required": genai.protos.Schema(type=genai.protos.Type.STRING)
    }
)
unexpected_elements_schema = genai.protos.Schema(
    type=genai.protos.Type.OBJECT,
    properties={
        "conversation_segments": genai.protos.Schema(
            type=genai.protos.Type.ARRAY,
            items=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "text": genai.protos.Schema(type=genai.protos.Type.STRING),
                    "additional_information": genai.protos.Schema(type=genai.protos.Type.STRING)
                }
            )
        )
    }
)
symptom_progression_analysis_schema = genai.protos.Schema(
    type=genai.protos.Type.OBJECT,
    properties={
        "symptom": genai.protos.Schema(type=genai.protos.Type.STRING),
        "progression": genai.protos.Schema(type=genai.protos.Type.STRING),  # e.g., Worsening, Improving, Stable
        "timeline": genai.protos.Schema(
            type=genai.protos.Type.ARRAY,
            items=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "date": genai.protos.Schema(type=genai.protos.Type.STRING),
                    "status": genai.protos.Schema(type=genai.protos.Type.STRING)  # Symptom status on that date
                }
            )
        )
    }
)
therapeutic_response_analysis_schema = genai.protos.Schema(
    type=genai.protos.Type.OBJECT,
    properties={
        "therapy_name": genai.protos.Schema(type=genai.protos.Type.STRING),
        "response_status": genai.protos.Schema(type=genai.protos.Type.STRING),  # e.g., Positive, Negative, No change
        "side_effects": genai.protos.Schema(type=genai.protos.Type.ARRAY, items=genai.protos.Schema(type=genai.protos.Type.STRING)),
        "adherence_level": genai.protos.Schema(type=genai.protos.Type.STRING)  # e.g., Full, Partial, None
    }
)
non_verbal_cues_analysis_schema = genai.protos.Schema(
    type=genai.protos.Type.ARRAY,
    items=genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "cue_type": genai.protos.Schema(type=genai.protos.Type.STRING),  # e.g., Pause, Sigh, Tone shift
            "timestamp": genai.protos.Schema(type=genai.protos.Type.STRING),
            "emotional_context": genai.protos.Schema(type=genai.protos.Type.STRING)  # e.g., Frustration, Relief
        }
    )
)

behavioral_analysis_schema = genai.protos.Schema(
    type=genai.protos.Type.ARRAY,
    items=genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "behavioral_marker": genai.protos.Schema(type=genai.protos.Type.STRING),  # e.g., Hesitation, Repetition
            "frequency": genai.protos.Schema(type=genai.protos.Type.NUMBER),
            "timestamp": genai.protos.Schema(type=genai.protos.Type.STRING),
            "context": genai.protos.Schema(type=genai.protos.Type.STRING)
        }
    )
)
cross_patient_data_comparison_schema = genai.protos.Schema(
    type=genai.protos.Type.OBJECT,
    properties={
        "comparison_metric": genai.protos.Schema(type=genai.protos.Type.STRING),  # e.g., Treatment Effectiveness
        "patient_set": genai.protos.Schema(
            type=genai.protos.Type.ARRAY,
            items=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "patient_id": genai.protos.Schema(type=genai.protos.Type.STRING),
                    "data_points": genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            "metric_name": genai.protos.Schema(type=genai.protos.Type.STRING),  # Define relevant properties
                            "value": genai.protos.Schema(type=genai.protos.Type.STRING)
                        }
                    )
                }
            )
        ),
        "findings": genai.protos.Schema(type=genai.protos.Type.STRING)
    }
)

language_translation_localization_schema = genai.protos.Schema(
    type=genai.protos.Type.OBJECT,
    properties={
        "original_language": genai.protos.Schema(type=genai.protos.Type.STRING),
        "translated_text": genai.protos.Schema(type=genai.protos.Type.STRING),
        "cultural_adaptations": genai.protos.Schema(type=genai.protos.Type.STRING),
        "accuracy_score": genai.protos.Schema(type=genai.protos.Type.NUMBER)
    }
)
medical_literature_suggestions_schema = genai.protos.Schema(
    type=genai.protos.Type.ARRAY,
    items=genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "paper_title": genai.protos.Schema(type=genai.protos.Type.STRING),
            "journal": genai.protos.Schema(type=genai.protos.Type.STRING),
            "publication_date": genai.protos.Schema(type=genai.protos.Type.STRING),
            "doi_link": genai.protos.Schema(type=genai.protos.Type.STRING)
        }
    )
)
longitudinal_health_trends_analysis_schema = genai.protos.Schema(
    type=genai.protos.Type.OBJECT,
    properties={
        "health_trend": genai.protos.Schema(type=genai.protos.Type.STRING),  # e.g., Blood Pressure, Mood
        "time_period": genai.protos.Schema(type=genai.protos.Type.STRING),  # e.g., 6 months, 1 year
        "observed_changes": genai.protos.Schema(
            type=genai.protos.Type.ARRAY,
            items=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "date": genai.protos.Schema(type=genai.protos.Type.STRING),
                    "metric": genai.protos.Schema(type=genai.protos.Type.STRING),  # e.g., Increase, Decrease
                    "commentary": genai.protos.Schema(type=genai.protos.Type.STRING)
                }
            )
        )
    }
)
clinical_trial_matching_schema = genai.protos.Schema(
    type=genai.protos.Type.OBJECT,
    properties={
        "trial_name": genai.protos.Schema(type=genai.protos.Type.STRING),
        "eligibility_criteria": genai.protos.Schema(type=genai.protos.Type.STRING),
        "trial_status": genai.protos.Schema(type=genai.protos.Type.STRING),  # e.g., Recruiting, Completed
        "contact_information": genai.protos.Schema(type=genai.protos.Type.STRING)
    }
)
health_cues_schema = genai.protos.Schema(
    type=genai.protos.Type.ARRAY,
    items=genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "cue": genai.protos.Schema(type=genai.protos.Type.STRING),  # e.g., Cough, Sneeze
            "frequency": genai.protos.Schema(type=genai.protos.Type.NUMBER),  # Number of occurrences
            "intensity": genai.protos.Schema(type=genai.protos.Type.STRING),  # e.g., Mild, Severe
            "timestamp": genai.protos.Schema(type=genai.protos.Type.STRING),  # When the cue occurred
            "context": genai.protos.Schema(type=genai.protos.Type.STRING)  # Additional context for the cue
        }
    )
)
doctor_mentions_schema = genai.protos.Schema(
    type=genai.protos.Type.ARRAY,
    items=genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "doctor_name": genai.protos.Schema(type=genai.protos.Type.STRING),  # Name of the doctor mentioned
            "specialty": genai.protos.Schema(type=genai.protos.Type.STRING),  # e.g., Cardiologist, Oncologist
            "context_of_mention": genai.protos.Schema(type=genai.protos.Type.STRING),  # e.g., Referral, Checkup
            "timestamp": genai.protos.Schema(type=genai.protos.Type.STRING)
        }
    )
)
medicine_mentions_schema = genai.protos.Schema(
    type=genai.protos.Type.ARRAY,
    items=genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "medicine_name": genai.protos.Schema(type=genai.protos.Type.STRING),  # e.g., Ibuprofen, Paracetamol
            "dosage": genai.protos.Schema(type=genai.protos.Type.STRING),  # e.g., 500 mg
            "frequency": genai.protos.Schema(type=genai.protos.Type.STRING),  # e.g., Twice daily
            "context_of_use": genai.protos.Schema(type=genai.protos.Type.STRING),  # e.g., Pain relief, Inflammation
            "prescribed_by": genai.protos.Schema(type=genai.protos.Type.STRING),  # Doctor or authority prescribing it
            "timestamp": genai.protos.Schema(type=genai.protos.Type.STRING)
        }
    )
)
lab_hospital_mentions_schema = genai.protos.Schema(
    type=genai.protos.Type.ARRAY,
    items=genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "facility_name": genai.protos.Schema(type=genai.protos.Type.STRING),  # e.g., XYZ Hospital, ABC Labs
            "facility_type": genai.protos.Schema(type=genai.protos.Type.STRING),  # e.g., Hospital, Lab
            "location": genai.protos.Schema(type=genai.protos.Type.STRING),  # City, Region, Country
            "services_provided": genai.protos.Schema(type=genai.protos.Type.STRING),  # e.g., Blood Test, MRI Scan
            "timestamp": genai.protos.Schema(type=genai.protos.Type.STRING)
        }
    )
)
demographics_family_schema = genai.protos.Schema(
    type=genai.protos.Type.OBJECT,
    properties={
        "age": genai.protos.Schema(type=genai.protos.Type.STRING),  # Age of the patient
        "gender": genai.protos.Schema(type=genai.protos.Type.STRING),  # Gender of the patient
        "location": genai.protos.Schema(type=genai.protos.Type.STRING),  # City, Region, Country
        "family_members": genai.protos.Schema(
            type=genai.protos.Type.ARRAY,
            items=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "name": genai.protos.Schema(type=genai.protos.Type.STRING),
                    "relation": genai.protos.Schema(type=genai.protos.Type.STRING),  # e.g., Father, Mother, Sibling
                    "health_condition": genai.protos.Schema(type=genai.protos.Type.STRING)  # Family history conditions
                }
            )
        ),
        "family_history_of_conditions": genai.protos.Schema(
            type=genai.protos.Type.ARRAY,
            items=genai.protos.Schema(type=genai.protos.Type.STRING)  # Conditions like diabetes, hypertension
        )
    }
)
region_specific_health_trends_schema = genai.protos.Schema(
    type=genai.protos.Type.OBJECT,
    properties={
        "region": genai.protos.Schema(type=genai.protos.Type.STRING),  # e.g., New York, Delhi
        "health_trend": genai.protos.Schema(type=genai.protos.Type.STRING),  # e.g., Diabetes, Respiratory issues
        "observed_increase_or_decrease": genai.protos.Schema(type=genai.protos.Type.STRING),  # Increase/Decrease
        "commentary": genai.protos.Schema(type=genai.protos.Type.STRING)  # Additional insights on the trend
    }
)
ambient_sound_analysis_schema = genai.protos.Schema(
    type=genai.protos.Type.ARRAY,
    items=genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "sound_type": genai.protos.Schema(type=genai.protos.Type.STRING),  # e.g., Cough, Sneeze, Background Noise
            "intensity": genai.protos.Schema(type=genai.protos.Type.STRING),  # Mild, Moderate, Severe
            "duration": genai.protos.Schema(type=genai.protos.Type.STRING),  # How long the sound lasted
            "timestamp": genai.protos.Schema(type=genai.protos.Type.STRING),  # When the sound occurred
            "context": genai.protos.Schema(type=genai.protos.Type.STRING)  # Additional context for sound
        }
    )
)
voice_biomarkers_schema = genai.protos.Schema(
    type=genai.protos.Type.OBJECT,
    properties={
        "biomarker_type": genai.protos.Schema(type=genai.protos.Type.STRING),  # Stress, Fatigue, Asthma
        "confidence_level": genai.protos.Schema(type=genai.protos.Type.STRING),  # How certain the system is
        "timestamp": genai.protos.Schema(type=genai.protos.Type.STRING),  # Time the voice marker was detected
        "related_symptoms": genai.protos.Schema(
            type=genai.protos.Type.ARRAY,
            items=genai.protos.Schema(type=genai.protos.Type.STRING)
        ),  # Related symptoms like breathlessness, coughing, etc.
        "context": genai.protos.Schema(type=genai.protos.Type.STRING)  # Additional context for biomarkers
    }
)
additional_information_schema = genai.protos.Schema(
    type=genai.protos.Type.OBJECT,
    properties={
        "info_type": genai.protos.Schema(type=genai.protos.Type.STRING),  # Example property
        "details": genai.protos.Schema(type=genai.protos.Type.STRING)
    }
)


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
        "metadata": metadata_schema,
        "doctor_patient_interaction": doctor_patient_interaction_schema,
        "risk_stratification": risk_stratification_schema,
        "diagnostic_imaging_analysis": diagnostic_imaging_analysis_schema,
        "treatment_follow_up": treatment_follow_up_schema,
        "unexpected_elements": unexpected_elements_schema,
        "symptom_progression_analysis": symptom_progression_analysis_schema,
        "therapeutic_response_analysis": therapeutic_response_analysis_schema,
        "non_verbal_cues_analysis": non_verbal_cues_analysis_schema,
        "behavioral_analysis": behavioral_analysis_schema,
        "cross_patient_data_comparison": cross_patient_data_comparison_schema,
        "language_translation_localization": language_translation_localization_schema,
        "medical_literature_suggestions": medical_literature_suggestions_schema,
        "longitudinal_health_trends_analysis": longitudinal_health_trends_analysis_schema,
        "clinical_trial_matching": clinical_trial_matching_schema,
        "health_cues": health_cues_schema,
        "doctor_mentions": doctor_mentions_schema,
        "medicine_mentions": medicine_mentions_schema,
        "lab_hospital_mentions": lab_hospital_mentions_schema,
        "demographics_family": demographics_family_schema,
        "region_specific_health_trends": region_specific_health_trends_schema,
        "ambient_sound_analysis": ambient_sound_analysis_schema,
        "voice_biomarkers": voice_biomarkers_schema,
        "additional_analysis": additional_analysis_schema,  # For unexpected properties or additional insights
        "additional_information": additional_information_schema,  # For unexpected properties or additional insights
    }
)
# --- Function Declaration ---
# Define the function declaration
function_declaration = genai.protos.FunctionDeclaration(
    name="process_medical_data",
    description="Processes medical video/audio data and extracts comprehensive structured information including transcription, summary, emotional analysis, and other health-related data.",
    parameters=final_output_schema  # The final schema incorporating all the components
)

process_medical_data = genai.protos.Tool(
    function_declarations=[function_declaration]
)

# Prepare the system and user prompt
system_prompt = """
You are an advanced medical transcription analysis assistant. Your task is to analyze the provided medical transcription and return a structured output based on the predefined schema.
    
The analysis should be thorough and comprehensive, taking into account all nuances in the conversation, including non-verbal cues. If there is any important information or insights that fall outside the schema, dynamically add a section in the analysis labeled 'additional_analysis' with relevant details.
    
Ensure the final output adheres to the predefined schema but allows flexibility for additional insights and findings as needed.
"""
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
    
    # Load audio
    audio = whisperx.load_audio(audio_file_path)
    
    # Transcribe with WhisperX
    result = whisper_model.transcribe(audio, batch_size=batch_size)
    
    # Print the raw segments before alignment
    logger.info(f"Initial transcription result for {audio_file_path}: {result['segments']}")
    
    return result

# --- Step 8: WhisperX Alignment ---
def align_with_whisperx(result, audio_file_path, whisper_model):
    logger.info(f"Aligning transcription for {audio_file_path}...")
    
    # Load alignment model
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    
    # Align results
    aligned_result = whisperx.align(result["segments"], model_a, metadata, audio_file_path, device)
    
    logger.info(f"Alignment completed for {audio_file_path}.")
    return aligned_result

# --- Step 9: WhisperX Diarization ---
def diarize_with_whisperx(audio_file_path, hf_token):
    logger.info(f"Performing speaker diarization for {audio_file_path}...")
    try:
        diarization_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
        
        # Perform diarization
        diarization_result = diarization_model(audio_file_path)
        
        logger.info(f"Diarization completed for {audio_file_path}.")
        logger.info(f"Type of diarization_result: {type(diarization_result)}")
        
        if isinstance(diarization_result, pd.DataFrame):
            # Convert DataFrame to list of dicts
            diarization_segments = diarization_result.to_dict('records')
            logger.info(f"Converted DataFrame to list of dicts. First segment: {diarization_segments[0] if diarization_segments else 'No segments'}")
        else:
            logger.warning(f"Unexpected diarization result type: {type(diarization_result)}")
            diarization_segments = []

        return diarization_segments

    except Exception as e:
        logger.error(f"Error during diarization: {str(e)}")
        logger.exception("Traceback:")
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
        
        # Use the assign_word_speakers function from WhisperX
        final_result = whisperx.assign_word_speakers(diarization_df, aligned_transcription)
        
        logger.info(f"Speaker labels assigned.")
        return final_result
    except Exception as e:
        logger.error(f"Error during speaker assignment: {str(e)}")
        return aligned_transcription

# --- Step 11: Gemini AI Integration for Analysis ---
# Analyze transcription with Gemini AI
def analyze_transcription_with_gemini(chunk_result, model, function_declaration):
    logger.info("Starting analysis of transcription chunk with Gemini AI...")
    transcription_text = " ".join([segment.get('text', '') for segment in chunk_result.get('segments', [])])

    user_prompt = f"""
    Here is a transcription chunk:
    {transcription_text}
    
    Please analyze this transcription thoroughly and return a structured output based on the predefined schema.
    If there are additional insights that do not fit into the schema, add them under the 'additional_analysis' section.
    """

    try:
    #     response = model.generate_content(
    #         contents=[user_prompt],
    #         generation_config=genai.types.GenerationConfig(
    #             temperature=0.7,
    #             top_p=0.9,
    #             top_k=40,
    #         ),
    #         tools=process_medical_data,
    #         tool_config={'function_calling_config':'ANY'}
    #     )

    #     if hasattr(response.candidates[0].content.parts[0], 'function_call'):
    #         fc = response.candidates[0].content.parts[0].function_call
    #         return fc.args

    #     return response.text

    # except Exception as e:
    #     logger.error(f"Error during Gemini model analysis: {str(e)}")
    #     return None

        # --- Generate content and analyze the transcription ---
        response = model.generate_content(
            contents=[user_prompt],
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                top_p=0.9,
                top_k=40,
            ),
            tools=[process_medical_data],  # Pass the tool again here as a list
            tool_config={'function_calling_config': 'ANY'}  # Ensure function calling is enabled
        )

        # --- Check if the function call was triggered ---
        if hasattr(response.candidates[0].content.parts[0], 'function_call'):
            fc = response.candidates[0].content.parts[0].function_call
            return fc.args  # Return the arguments from the function call
        else:
            return response.text  # Fallback in case no function call was made

    except Exception as e:
        logger.error(f"Error during Gemini model analysis: {str(e)}")
        return None
# --- Main Processing Flow ---
# --- Main Processing Flow ---
def main(video_file_path, hf_token):
    logger.info(f"Starting processing for video: {video_file_path}")
    
    # Step 1: Segment the video
    try:
        segments = segment_large_video(video_file_path)
    except Exception as e:
        logger.error(f"Error during video segmentation: {str(e)}")
        return

    for segment_file_path in segments:
        segment_folder = os.path.splitext(segment_file_path)[0]
        os.makedirs(segment_folder, exist_ok=True)

        # Step 2: Extract audio
        try:
            audio_file_path = extract_audio_from_segment(segment_file_path)
        except Exception as e:
            logger.error(f"Error extracting audio: {str(e)}")
            continue

        # Visualize the audio waveform
        waveform_image_path = os.path.join(segment_folder, f"waveform_{os.path.basename(audio_file_path)}.png")
        try:
            visualize_audio_waveform(audio_file_path, waveform_image_path)
        except Exception as e:
            logger.error(f"Error visualizing waveform: {str(e)}")

        # Step 3: Remove silence
        try:
            cleaned_audio_path = remove_silence(audio_file_path)
        except Exception as e:
            logger.error(f"Error removing silence: {str(e)}")
            continue

        # Step 4: Chunk audio
        try:
            chunks = chunk_audio(cleaned_audio_path)
        except Exception as e:
            logger.error(f"Error chunking audio: {str(e)}")
            continue

        # Step 5: Normalize and reduce noise in chunks
        cleaned_chunks = []
        for chunk in chunks:
            try:
                cleaned_chunk = normalize_and_reduce_noise(AudioSegment.from_file(chunk))
                cleaned_chunks.append(cleaned_chunk)
            except Exception as e:
                logger.error(f"Error normalizing and reducing noise: {str(e)}")
                continue

        # Step 6: Save cleaned chunks and get their paths
        try:
            cleaned_chunk_paths = save_cleaned_chunks(cleaned_chunks, segment_folder)
        except Exception as e:
            logger.error(f"Error saving cleaned chunks: {str(e)}")
            continue

        # Step 7: Transcribe using WhisperX and align transcription
        for cleaned_chunk_path in tqdm(cleaned_chunk_paths, desc="Processing cleaned chunks"):
            try:
                transcription_result = transcribe_with_whisperx(cleaned_chunk_path, whisper_model)
                
                # Save the transcription result
                transcription_file = os.path.join(segment_folder, f"transcription_chunk_{os.path.basename(cleaned_chunk_path)}.json")
                save_json_to_file(transcription_result, transcription_file, "Transcription result")
                
            except Exception as e:
                logger.error(f"Error during transcription: {str(e)}")
                continue

            # Step 8: Align transcription with WhisperX
            try:
                aligned_result = align_with_whisperx(transcription_result, cleaned_chunk_path, whisper_model)
                transcription_file = os.path.join(segment_folder, f"aligned_transcription_chunk_{os.path.basename(cleaned_chunk_path)}.json")
                save_json_to_file(aligned_result, transcription_file, "Aligned transcription result")
            except Exception as e:
                logger.error(f"Error aligning transcription: {str(e)}")
                continue

            # Step 9: Analyze transcription with Gemini AI (always using the aligned transcription)
            try:
                analysis_result = analyze_transcription_with_gemini(transcription_result, model, function_declaration)
                if analysis_result:
                    analysis_file = os.path.join(segment_folder, f"analysis_chunk_{os.path.basename(cleaned_chunk_path)}.json")
                    save_json_to_file(analysis_result, analysis_file, "Gemini AI analysis result")
                else:
                    logger.error("Gemini AI analysis failed.")
            except Exception as e:
                logger.error(f"Error during Gemini AI analysis: {str(e)}")
                continue

if __name__ == "__main__":
    video_file_path = '/Users/pranay/Projects/LLM/video/proj1/data/Chiranjeevi_Video_Dec_21.mp4'
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise ValueError("Hugging Face token not found. Please set it in the environment.")
    
    # --- Initialize the Gemini model ---
    model = genai.GenerativeModel(
    model_name='models/gemini-1.5-flash',
    tools=[process_medical_data],  # Pass the tool correctly as a list
    system_instruction=system_prompt
)

    logger.info("Using Hugging Face token: {}".format(hf_token))
    main(video_file_path, hf_token)