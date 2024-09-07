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
if torch.backends.mps.is_available():
    device = "mps"  # Try to use MPS
    compute_type = "float32"  # MPS default
    logger.info(f"Running on MPS with {compute_type} compute type.")
else:
    device = "cpu"  # Fallback to CPU if MPS is not available
    compute_type = "int8"  # Use int8 for CPU efficiency
    logger.info(f"Running on {device} with {compute_type} compute type.")

# Check if the 'mps' device is supported by ctranslate2, otherwise fallback to 'cpu'
if device == "mps":
    try:
        whisper_model = whisperx.load_model("large-v2", device="mps", compute_type=compute_type)
    except ValueError as e:
        logger.warning(f"MPS device is not supported by ctranslate2. Falling back to CPU. Error: {str(e)}")
        device = "cpu"  # Force CPU usage
        whisper_model = whisperx.load_model("large-v2", device=device, compute_type="int8")
else:
    whisper_model = whisperx.load_model("large-v2", device=device, compute_type=compute_type)

# Embedding Models
EMBEDDING_MODEL_SMALL = "text-embedding-3-small"
EMBEDDING_MODEL_LARGE = "text-embedding-3-large"

# --- Function to Fetch Embeddings ---
def get_embedding_for_text(text, model=EMBEDDING_MODEL_SMALL):
    """
    Get embeddings for a given text using OpenAI's API.
    """
    try:
        if not text.strip():
            logger.error("Text input is empty or only contains whitespace.")
            return None

        # Clean up newlines and ensure embedding API receives proper input
        text = text.replace("\n", " ")
        response = client.embeddings.create(
            input=[text],
            model=model
        )
        
        embedding = response.data[0].embedding
        return embedding

    except Exception as e:
        logger.error(f"Error fetching embeddings: {str(e)}")
        return None
    
def save_embeddings_to_file(embedding, aligned_embedding, filepath):
    """
    Save both transcription and aligned embeddings to a JSON file.
    """
    try:
        if not embedding or not aligned_embedding:
            logger.error("One or both embeddings are missing. Cannot save to file.")
            return
        
        embeddings_data = {
            "transcription_embedding": embedding,
            "aligned_embedding": aligned_embedding
        }

        with open(filepath, "w") as f:
            json.dump(embeddings_data, f, indent=4)
        
        logger.info(f"Embeddings successfully saved to {filepath}")
    
    except Exception as e:
        logger.error(f"Error saving embeddings to file: {str(e)}")
        
transcript_schema= {
  "metadata": {
    "location": "",
    "setting": "",
    "duration": "",
    "visit_type": "",
    "technology_used": "",
    "healthcare_providers": [
      {
        "name": "",
        "role": "",
        "qualifications": "",
        "specializations": "",
        "years_of_experience": ""
      }
    ],
    "call_quality": "",
    "connection_issues": "",
    "background_noise": "",
    "language_of_consultation": ""
  },
  "transcription": {
    "segments": [
      {
        "speaker": "",
        "non_verbal_cues": ""
      }
    ]
  },
  "content_summary": {
    "main_points": "",
    "interaction_purpose": "",
    "key_decisions_made": ""
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
  "patient_information": {
    "demographics": {
      "name": "",
      "age": "",
      "gender": "",
      "location": "",
      "ethnicity": "",
      "preferred_language": "",
      "occupation": ""
    },
    "history": {
      "past_medical_conditions": "",
      "surgeries": "",
      "allergies": "",
      "family_history": "",
      "social_history": "",
      "noted_gaps_or_inconsistencies": "",
      "previous_hospitalizations": "",
      "childhood_illnesses": ""
    },
    "current_status": {
      "symptoms": [
        {
          "symptom": "",
          "progression": "",
          "timeline": "",
          "severity": "",
          "triggers": "",
          "alleviating_factors": ""
        }
      ],
      "risk_stratification": {
        "risk_category": "",
        "underlying_conditions": "",
        "risk_factors": "",
        "recommendations": ""
      },
      "vital_signs": {
        "blood_pressure": "",
        "heart_rate": "",
        "respiratory_rate": "",
        "temperature": "",
        "oxygen_saturation": ""
      }
    },
    "health_behaviors": {
      "smoking_status": "",
      "alcohol_consumption": "",
      "physical_activity_level": "",
      "diet_habits": "",
      "sleep_patterns": ""
    },
    "insurance_information": {
      "insurance_provider": "",
      "policy_number": "",
      "coverage_details": ""
    }
  },
  "clinical_data": {
    "test_results": [
      {
        "test_name": "",
        "result": "",
        "reference_range": "",
        "interpretation": "",
        "clinical_significance": "",
        "follow_up_recommended": "",
        "date_performed": ""
      }
    ],
    "diagnostic_imaging": {
      "image_type": "",
      "findings": "",
      "follow_up_tests": "",
      "implications_for_treatment": "",
      "radiologist_notes": ""
    },
    "medications": [
      {
        "name": "",
        "dosage": "",
        "frequency": "",
        "context_of_use": "",
        "prescribed_by": "",
        "timestamp": "",
        "side_effects_experienced": "",
        "effectiveness": ""
      }
    ],
    "medical_devices": [
      {
        "name": "",
        "purpose": "",
        "usage_instructions": "",
        "patient_familiarity": "",
        "maintenance_requirements": ""
      }
    ],
    "physical_examination": {
      "general_appearance": "",
      "cardiovascular": "",
      "respiratory": "",
      "gastrointestinal": "",
      "musculoskeletal": "",
      "neurological": "",
      "skin": ""
    },
    "laboratory_tests": [
      {
        "test_name": "",
        "test_date": "",
        "result": "",
        "reference_range": "",
        "interpretation": ""
      }
    ],
    "allergies_and_intolerances": [
      {
        "allergen": "",
        "reaction_type": "",
        "severity": "",
        "date_identified": ""
      }
    ],
    "immunizations": [
      {
        "vaccine_name": "",
        "date_administered": "",
        "lot_number": "",
        "next_due_date": ""
      }
    ]
  },
  "interaction_analysis": {
    "significant_events": [
      {
        "time": "",
        "event_type": "",
        "description": ""
      }
    ],
    "emotional_psychological": [
      {
        "speaker": "",
        "emotion": "",
        "intensity": "",
        "timestamp": "",
        "context": ""
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
    "doctor_patient_dynamics": {
      "doctor_speaking_time": "",
      "patient_speaking_time": "",
      "interruptions": "",
      "empathy_expressions": "",
      "overall_patient_satisfaction": "",
      "rapport_building_efforts": ""
    },
    "non_verbal_cues": [
      {
        "cue_type": "",
        "timestamp": "",
        "emotional_context": ""
      }
    ],
    "behavioral_markers": [
      {
        "marker": "",
        "frequency": "",
        "timestamp": "",
        "context": ""
      }
    ],
    "voice_biomarkers": [
      {
        "type": "",
        "confidence_level": "",
        "timestamp": "",
        "related_symptoms": "",
        "context": ""
      }
    ],
    "ambient_sounds": [
      {
        "type": "",
        "intensity": "",
        "duration": "",
        "timestamp": "",
        "context": ""
      }
    ]
  },
  "care_quality_assessment": {
    "guideline_adherence": "",
    "comprehensiveness": "",
    "identified_gaps": "",
    "efficiency_assessment": "",
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
    "health_equity_considerations": {
      "language_justice": "",
      "cultural_safety_practices": "",
      "accessibility_accommodations": ""
    },
    "care_continuity": "",
    "patient_safety_measures": ""
  },
  "treatment_and_follow_up": {
    "current_treatment": {
      "treatment_followed": "",
      "adherence_level": "",
      "outcomes_observed": "",
      "side_effects": "",
      "additional_treatments_required": ""
    },
    "therapeutic_response": {
      "therapy_name": "",
      "response_status": "",
      "side_effects": "",
      "adherence_level": ""
    },
    "follow_up_actions": {
      "scheduled_appointments": "",
      "prescribed_tests": "",
      "referrals": "",
      "recommended_lifestyle_changes": "",
      "self_monitoring_instructions": ""
    },
    "care_plan": {
      "short_term_goals": "",
      "long_term_goals": "",
      "anticipated_challenges": "",
      "support_resources": ""
    },
    "medication_adjustments": [
      {
        "medication_name": "",
        "previous_dosage": "",
        "new_dosage": "",
        "reason_for_change": ""
      }
    ],
    "prescription_details": [
      {
        "medication_name": "",
        "dosage": "",
        "frequency": "",
        "duration": "",
        "special_instructions": ""
      }
    ]
  },
  "patient_engagement": {
    "communication_style": {
      "clarity_of_expression": "",
      "comprehension_level": "",
      "engagement_level": ""
    },
    "health_literacy": {
      "patient_knowledge_level": "",
      "use_of_medical_jargon": "",
      "clarification_requests": ""
    },
    "patient_activation": {
      "knowledge_level": "",
      "skill_level": "",
      "confidence_level": ""
    },
    "self_management": {
      "self_monitoring_practices": "",
      "lifestyle_modification_efforts": "",
      "adherence_to_home_care_instructions": ""
    },
    "patient_education": {
      "topics_covered": "",
      "patient_understanding": "",
      "resources_provided": "",
      "teach_back_method_used": ""
    },
    "support_system": {
      "family_involvement": "",
      "caregiver_presence": "",
      "community_resources_mentioned": "",
      "support_group_recommendations": ""
    },
    "patient_preferences": {
      "treatment_preferences": "",
      "communication_preferences": "",
      "cultural_considerations": ""
    },
    "motivational_interviewing": {
      "stage_of_change": "",
      "patient_motivation_level": "",
      "identified_barriers": "",
      "agreed_upon_goals": ""
    }
  },
  "decision_making": {
    "options_presented": "",
    "patient_involvement": "",
    "reasoning_explained": "",
    "shared_decision_making_approach": "",
    "patient_values_considered": ""
  },
  "specialized_considerations": {
    "pain_assessment": {
      "pain_scale_used": "",
      "pain_description": "",
      "impact_on_daily_activities": "",
      "pain_management_strategies": ""
    },
    "end_of_life_care": {
      "palliative_care_options": "",
      "hospice_discussion": "",
      "life_sustaining_treatment_preferences": "",
      "advance_care_planning": ""
    },
    "genetic_considerations": {
      "family_genetic_history_discussed": "",
      "genetic_testing_recommendations": "",
      "personalized_medicine_approaches": "",
      "genetic_counseling_referral": ""
    },
    "environmental_health": {
      "occupational_hazards": "",
      "home_environment_assessment": "",
      "exposure_to_pollutants": "",
      "environmental_risk_mitigation_strategies": ""
    },
    "emergency_preparedness": {
      "emergency_contact_information": "",
      "disaster_readiness_plans": "",
      "medical_alert_systems": "",
      "emergency_medication_supply": ""
    },
    "integrative_medicine": {
      "complementary_therapies_discussed": "",
      "mind_body_interventions": "",
      "nutritional_supplement_recommendations": "",
      "integration_with_conventional_treatments": ""
    },
    "reproductive_health": {
      "fertility_concerns": "",
      "contraception_discussion": "",
      "pregnancy_planning": "",
      "menopause_management": ""
    },
    "mental_health": {
      "mood_assessment": "",
      "anxiety_screening": "",
      "substance_use_evaluation": "",
      "mental_health_referrals": ""
    },
    "nutritional_assessment": {
      "dietary_restrictions": "",
      "nutritional_deficiencies": "",
      "recommended_dietary_changes": ""
    },
    "rehabilitation_needs": {
      "physical_therapy": "",
      "occupational_therapy": "",
      "speech_therapy": ""
    }
  },
  "healthcare_system_interactions": {
    "facilities_mentioned": [
      {
        "facility_name": "",
        "facility_type": "",
        "location": "",
        "services_provided": "",
        "timestamp": ""
      }
    ],
    "healthcare_costs": {
      "cost_estimates_provided": "",
      "insurance_coverage_discussed": "",
      "financial_concerns_addressed": "",
      "payment_plans_offered": ""
    },
    "care_coordination": {
      "interdisciplinary_team_involvement": "",
      "referral_coordination": "",
      "care_transition_plans": "",
      "communication_between_providers": ""
    },
    "health_information_technology": {
      "electronic_health_record_references": "",
      "clinical_decision_support_tools": "",
      "telemedicine_platform_functionality": "",
      "patient_portal_instructions": ""
    },
    "digital_health_integration": {
      "wearable_device_data": "",
      "patient_portal_usage": "",
      "mobile_health_app_recommendations": "",
      "remote_monitoring_setup": ""
    },
    "prior_authorizations": [
      {
        "service_or_medication": "",
        "insurance_response": "",
        "status": "",
        "next_steps": ""
      }
    ]
  },
  "compliance_and_legal": {
    "hipaa_adherence": "",
    "informed_consent": "",
    "documentation_completeness": "",
    "capacity_assessment": "",
    "advance_directive_discussion": "",
    "ethical_dilemmas_addressed": "",
    "malpractice_risk_assessment": ""
  },
  "research_and_data": {
    "research_participation": {
      "clinical_trials_mentioned": "",
      "patient_registries": "",
      "biobank_participation": "",
      "research_consent_process": ""
    },
    "cross_patient_comparison": {
      "comparison_metric": "",
      "patient_set": "",
      "findings": ""
    },
    "longitudinal_health_trends": {
      "health_trend": "",
      "time_period": "",
      "observed_changes": ""
    },
    "region_specific_trends": {
      "region": "",
      "health_trend": "",
      "observed_increase_or_decrease": "",
      "commentary": ""
    },
    "patient_reported_outcomes": {
      "quality_of_life_measures": "",
      "functional_status_assessment": "",
      "symptom_burden_evaluation": "",
      "patient_satisfaction_scores": ""
    }
  },
  "social_determinants_of_health": {
    "housing_situation": "",
    "food_security": "",
    "transportation_access": "",
    "employment_status": "",
    "education_level": "",
    "social_support_network": "",
    "access_to_healthcare": "",
    "neighborhood_safety": ""
  },
  "preventive_care": {
    "screenings_recommended": "",
    "lifestyle_modifications_suggested": "",
    "vaccination_status_reviewed": "",
    "health_risk_assessments": "",
    "preventive_medications_discussed": ""
  },
  "medication_management": {
    "current_medications_reviewed": "",
    "medication_changes_made": "",
    "potential_drug_interactions_identified": "",
    "medication_reconciliation_performed": "",
    "adherence_strategies_discussed": ""
  },
  "overall_assessment": {
    "key_findings": "",
    "critical_points": "",
    "areas_for_improvement": "",
    "positive_aspects": "",
    "follow_up_recommendations": "",
    "prognosis": "",
    "care_plan_summary": ""
  },
  "unexpected_elements": {
    "conversation_segments": "",
    "additional_information": "",
    "unusual_patient_concerns": "",
    "unforeseen_complications": ""
  },
  "language_translation": {
    "original_language": "",
    "translated_text": "",
    "cultural_adaptations": "",
    "accuracy_score": "",
    "medical_terminology_translation_challenges": ""
  },
  "medical_literature_suggestions": [
    {
      "paper_title": "",
      "journal": "",
      "publication_date": "",
      "doi_link": "",
      "relevance_to_case": ""
    }
  ],
  "clinical_trial_matching": [
    {
      "trial_name": "",
      "eligibility_criteria": "",
      "trial_status": "",
      "contact_information": "",
      "potential_benefits_risks": ""
    }
  ],
  "telemedicine_specific_factors": {
    "visual_assessment_quality": "",
    "technical_difficulties_impact": "",
    "physical_exam_limitations": "",
    "patient_comfort_with_technology": "",
    "remote_monitoring_equipment_used": ""
  },
  "follow_up_communication": {
    "scheduled_follow_up_method": "",
    "patient_instructions_provided": "",
    "care_summary_sent": "",
    "post_visit_survey_planned": "",
    "patient_portal_messages": [
      {
        "date": "",
        "subject": "",
        "content_summary": "",
        "response_needed": ""
      }
    ]
  },
  "care_team_collaboration": {
    "consulted_specialists": [
      {
        "specialty": "",
        "reason_for_consult": "",
        "recommendations": ""
      }
    ],
    "multidisciplinary_team_discussions": "",
    "care_coordination_efforts": ""
  },
  "patient_reported_measures": {
    "pain_scales": "",
    "functional_status_assessments": "",
    "quality_of_life_questionnaires": "",
    "symptom_severity_scales": ""
  },
  "clinical_decision_support": {
    "alerts_triggered": [
      {
        "alert_type": "",
        "reason": "",
        "action_taken": ""
      }
    ],
    "risk_scores_calculated": [
      {
        "score_name": "",
        "result": "",
        "interpretation": ""
      }
    ],
    "clinical_pathways_followed": ""
  },
  "quality_metrics": {
    "visit_duration": "",
    "wait_time": "",
    "patient_satisfaction_score": "",
    "clinical_outcome_measures": "",
    "adherence_to_clinical_guidelines": ""
  },
  "communication_assessment": {
    "use_of_teach_back_method": "",
    "clarity_of_explanations": "",
    "addressing_patient_questions": "",
    "use_of_visual_aids": "",
    "empathy_and_compassion_demonstrated": ""
  },
  "health_cues": {
    "cue": "",
    "frequency": "",
    "intensity": "",
    "timestamp": "",
    "context": ""
  },
  "doctor_mentions": [
    {
      "doctor_name": "",
      "specialty": "",
      "context_of_mention": "",
      "timestamp": ""
    }
  ],
  "medicine_mentions": [
    {
      "medicine_name": "",
      "dosage": "",
      "frequency": "",
      "context_of_use": "",
      "prescribed_by": "",
      "timestamp": ""
    }
  ],
  "lab_hospital_mentions": [
    {
      "facility_name": "",
      "facility_type": "",
      "location": "",
      "services_provided": "",
      "timestamp": ""
    }
  ],
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
  "ambient_sound_analysis": [
    {
      "sound_type": "",
      "intensity": "",
      "duration": "",
      "timestamp": "",
      "context": ""
    }
  ],
  "voice_biomarkers": [
    {
      "biomarker_type": "",
      "confidence_level": "",
      "timestamp": "",
      "related_symptoms": "",
      "context": ""
    }
  ],
  "additional_information": {
    "info_type": "",
    "details": ""
  },
  "call_technical_details": {
    "call_quality": "",
    "connection_issues": "",
    "background_noise": ""
  },
"healthcare_costs": {
    "consultation_fee": "",
    "medication_costs": "",
    "procedure_costs": "",
    "laboratory_test_costs": "",
    "imaging_costs": "",
    "insurance_coverage_details": "",
    "out_of_pocket_expenses": "",
    "payment_plans_discussed": "",
    "financial_assistance_programs_mentioned": "",
    "cost_comparison_with_alternatives": ""
  },

  "medical_travel": {
    "travel_required": "",
    "destination": "",
    "purpose": "",
    "duration": "",
    "mode_of_transport": "",
    "accommodation_details": "",
    "travel_costs": "",
    "travel_insurance": "",
    "travel_related_health_concerns": "",
    "post_travel_follow_up_plans": ""
  },

  "telemedicine_specifics": {
    "platform_used": "",
    "technical_issues_encountered": [
      {
        "issue_type": "",
        "impact_on_consultation": "",
        "resolution": ""
      }
    ],
    "patient_location_during_consultation": "",
    "privacy_measures_taken": "",
    "digital_consent_process": "",
    "remote_physical_exam_techniques_used": "",
    "virtual_waiting_room_experience": "",
    "screen_sharing_used": "",
    "chat_features_used": "",
    "video_quality": "",
    "audio_quality": "",
    "patient_tech_literacy_assessment": "",
    "caregiver_involvement_in_telconsult": "",
    "comparison_to_in_person_visit": ""
  },

  "prescription_and_pharmacy": {
    "e_prescription_sent": "",
    "pharmacy_details": {
      "name": "",
      "location": "",
      "contact_information": ""
    },
    "medication_availability_checked": "",
    "pharmacy_benefit_manager_interactions": "",
    "drug_substitution_discussions": ""
  },

  "patient_generated_health_data": {
    "wearable_device_data": {
      "device_type": "",
      "data_type": "",
      "data_summary": "",
      "integration_with_ehr": ""
    },
    "home_monitoring_data": {
      "device_type": "",
      "measurements": "",
      "frequency_of_measurements": "",
      "data_trends": ""
    },
    "patient_reported_outcomes": {
      "questionnaire_type": "",
      "scores": "",
      "interpretation": ""
    },
    "mobile_health_app_data": {
      "app_name": "",
      "data_type": "",
      "data_summary": "",
      "clinician_assessment_of_data": ""
    }
  },

  "care_transitions": {
    "transition_type": "",
    "reason_for_transition": "",
    "receiving_facility_or_provider": "",
    "handoff_communication_details": "",
    "medication_reconciliation_performed": "",
    "follow_up_appointments_scheduled": "",
    "patient_education_provided_for_transition": ""
  },

  "cultural_and_linguistic_considerations": {
    "interpreter_used": "",
    "interpreter_type": "",
    "cultural_beliefs_affecting_care": "",
    "health_literacy_assessment": "",
    "culturally_tailored_education_materials_provided": ""
  },

  "patient_safety": {
    "fall_risk_assessment": "",
    "medication_safety_review": "",
    "allergy_alerts_reviewed": "",
    "patient_identification_protocol_followed": "",
    "safety_concerns_raised_during_consultation": ""
  },

  "patient_engagement_technology": {
    "patient_portal_usage_discussed": "",
    "mobile_app_recommendations": "",
    "telehealth_platform_training_provided": "",
    "digital_health_literacy_assessment": ""
  },

  "care_coordination": {
    "primary_care_provider_notified": "",
    "specialist_referrals_made": [
      {
        "specialty": "",
        "reason_for_referral": "",
        "urgency": ""
      }
    ],
    "care_team_communication_method": "",
    "shared_decision_making_tools_used": ""
  },

  "environmental_health_assessment": {
    "home_environment_discussed": "",
    "occupational_exposures_evaluated": "",
    "environmental_allergies_considered": "",
    "recommendations_for_environmental_modifications": ""
  },

  "health_equity_considerations": {
    "social_determinants_of_health_addressed": "",
    "access_to_care_challenges_identified": "",
    "health_disparities_noted": "",
    "culturally_competent_care_provided": ""
  },

  "patient_education_materials": {
    "format_of_materials": "",
    "topics_covered": "",
    "language_of_materials": "",
    "accessibility_considerations": ""
  },

  "follow_up_and_continuity_of_care": {
    "follow_up_appointment_scheduled": "",
    "care_plan_shared_with_patient": "",
    "ongoing_monitoring_plan": "",
    "communication_preferences_for_future_contact": ""
  }
}

system_prompt="""
You are an advanced medical transcription analysis assistant. Your task is to analyze a provided medical transcription and return a structured output based on a predefined JSON schema. 
You must utilize the semantic meaning from the text to produce the most accurate and insightful analysis.

Here is the JSON schema you should use for your output:
{transcript_schema_category}

Please follow these instructions to complete your task:

1. Carefully read and analyze the provided transcript.

2. Based on your analysis, fill out the JSON schema provided earlier. Ensure that your output is a valid JSON and comprehensively covers all sections of the schema.

3. For each section of the schema:
   a. Extract relevant information from the transcript.
   b. Summarize and structure the information according to the schema.
   c. If a particular field is not applicable or the information is not available in the transcript, denote it as a "N/A".
   d. If you encounter multiple relevant pieces of information for a single field, create an array to contain all the information, even if the schema shows the field as a single value.

4. If you encounter any insights or findings that fall outside the provided schema, include them in the "additional_information" section of the schema.

5. If you are unsure about any information or if the transcript is unclear, indicate this in the relevant field of the schema. Do not make assumptions or fill in information that is not present in the transcript.

6. Ensure your analysis is:
   a. Precise: Cover all key aspects derived from the text.
   b. Thorough: Capture any subtle nuances, context, or patterns that the data presents.
   c. Structured: Ensure the output is clean and ready for further processing.

7. After completing your analysis, review your output to ensure it adheres to the provided schema and contains all relevant information from the transcript.

8. Present your final output as a valid JSON object, structured according to the provided schema. Enclose your entire output within the JSON Schema.

9. For any field where multiple values are appropriate (such as symptoms, medications, or procedures), use an array even if the schema shows a single value. This allows for more comprehensive data capture without modifying the base schema.

Remember, your goal is to provide a comprehensive, accurate, and structured analysis of the medical transcription that can be easily processed and understood by other systems or healthcare professionals. Prioritize capturing all relevant information, even if it means adapting single-value fields into arrays when necessary.
"""

# Utility function to save JSON data
def save_json_to_file(data, filepath):
    """
    Save data as a JSON file to the specified filepath.

    Args:
    - data: The data to save, expected to be a dictionary or JSON-serializable object.
    - filepath: The location where the file will be saved.
    """
    try:
        # Check if data is JSON-serializable
        json.dumps(data)  # This will raise an error if not serializable
        
        # Write the JSON data to the file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        
        logger.info(f"Results successfully saved at: {filepath}")
    
    except (TypeError, ValueError) as e:
        logger.error(f"Failed to serialize data as JSON: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error saving JSON to file: {str(e)}")
        
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
            logger.error(f"Missing 'text' in one or more transcription segments for {audio_file_path}.")
            return None

        logger.info(f"Initial transcription result for {audio_file_path}: {result['segments']}")
        
        # Return transcription result without embedding
        return result

    except Exception as e:
        logger.error(f"Error during transcription for {audio_file_path}: {str(e)}")
        return None

# --- Step 8: WhisperX Alignment without embeddings ---
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

        # Return aligned result without embedding
        return aligned_result

    except Exception as e:
        logger.error(f"Error during alignment for {audio_file_path}: {str(e)}")
        return None
# --- Step 9: WhisperX Diarization ---
def diarize_with_whisperx(audio_file_path, hf_token, retries=3, min_speakers=None, max_speakers=None):
    """
    Perform speaker diarization using WhisperX, with optional retries and speaker count limits.
    """
    logger.info(f"Performing speaker diarization for {audio_file_path}...")
    try:
        # Initialize the diarization pipeline
        diarization_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)

        # Retry mechanism in case of failure
        for attempt in range(retries):
            try:
                # Perform diarization with optional min/max speaker counts
                if min_speakers and max_speakers:
                    diarization_result = diarization_model(audio_file_path, min_speakers=min_speakers, max_speakers=max_speakers)
                else:
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
        diarization_df = pd.DataFrame(diarization_segments)

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
    
def analyze_with_openai(client, aligned_data, category, schema, system_prompt, seed=None):
    """
    This function sends the aligned transcription data to the OpenAI API for analysis of a specific category.
    """
    logger.info(f"Starting analysis with OpenAI for category: {category}")

    try:
        # Format the system prompt with the current category's schema
        category_system_prompt = system_prompt.format(transcript_schema_category=json.dumps({category: schema[category]}, indent=2))
        
        # Improved user prompt
        user_prompt = f"""
        Analyze the following aligned transcription data for the '{category}' category.

        Guidelines:
        1. Extract relevant information based on the provided schema.
        2. Consider semantic meaning, alignment info, and pertinent metadata.
        3. Include important information not fitting the schema in 'additional_information'.
        4. Mark unavailable or unclear info as 'N/A' or explain uncertainty.
        5. Ensure output is a valid JSON object adhering to the given schema.

        Aligned Transcription Data:
        {json.dumps(aligned_data, indent=2)}

        Provide your analysis as a structured JSON output.
        """
        
        response = client.chat.completions.create(
                model="gpt-4o-mini",  # Ensure you're using the correct model name
                messages=[
                    {"role": "system", "content": category_system_prompt},  # System prompt with schema instructions
                    {"role": "user", "content": user_prompt}  # User prompt with the aligned transcription data
                ],
                seed=SEED,
                temperature=0.3,
                top_p=1,
                # max_tokens=2000,  # Adjust based on expected output size
                stop=None  # This can be added if needed
            )
        # Extract and log the raw response
        full_response = response.choices[0].message.content
        logger.info(f"Full raw response from OpenAI for {category}: {full_response}")

        # Attempt to parse the response as JSON
        try:
            # Remove markdown code block delimiters if present
            json_string = re.sub(r'```json\s*|\s*```', '', full_response).strip()
            category_result = json.loads(json_string)
            if isinstance(category_result, dict) and category in category_result:
                return category_result[category]
            else:
                logger.warning(f"Unexpected format for {category}. Using raw result.")
                return category_result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON for {category}: {str(e)}")
            logger.error(f"Raw response from OpenAI: {full_response}")
            return None

    except Exception as e:
        logger.error(f"Error during OpenAI API call for {category}: {str(e)}", exc_info=True)
        return None

def main(video_file_path, hf_token, whisper_model):
    logger.info(f"Starting processing for video: {video_file_path}")

    try:
        
        # Step 1: Segment the video
        segments = segment_large_video(video_file_path)
        if not segments:
            logger.error(f"No segments created from video: {video_file_path}")
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

                    # Step 8: Align the transcription with WhisperX
                    aligned_result = align_with_whisperx(transcription_result, cleaned_chunk_path, whisper_model)

                    # Skip the chunk if alignment fails
                    if not aligned_result:
                        logger.error(f"Skipping chunk {cleaned_chunk_path} due to alignment failure.")
                        continue

                    # Save the aligned transcription result
                    aligned_file = os.path.join(segment_folder, f"aligned_transcription_chunk_{os.path.basename(cleaned_chunk_path)}.json")
                    save_json_to_file(aligned_result, aligned_file)
                    logger.info(f"Aligned transcription saved at: {aligned_file}")

                    # Analyze with OpenAI
                    analysis_results = {}
                    for category in transcript_schema.keys():
                        category_result = analyze_with_openai(client, aligned_result, category, transcript_schema, system_prompt)
                        if category_result:
                            analysis_results[category] = category_result
                            print(f"\n{category.upper()}:")
                            print(json.dumps(category_result, indent=2))
                        else:
                            logger.error(f"Failed to generate analysis for category {category} in chunk {os.path.basename(cleaned_chunk_path)}")

                    # Save the final analysis result
                    if analysis_results:
                        final_output_file = os.path.join(segment_folder, f"final_analysis_cleaned_chunk_{os.path.basename(cleaned_chunk_path)}.json")
                        save_json_to_file(analysis_results, final_output_file)
                        logger.info(f"Final analysis saved as {final_output_file}")
                    else:
                        logger.error(f"No analysis results generated for chunk {os.path.basename(cleaned_chunk_path)}")

            except Exception as e:
                logger.error(f"Error processing segment {segment_file_path}: {str(e)}")
                continue  # Continue processing the next segment

    except Exception as e:
        logger.error(f"Error during video processing: {str(e)}")
        
if __name__ == "__main__":
    # Define the path to your video file
    video_file_path = '/Users/pranay/Projects/LLM/video/proj1/data/Chiranjeevi_Video_Dec_21.mp4'
    hf_token = os.getenv("HUGGINGFACE_TOKEN")  # Replace with your actual Hugging Face token if required

    # Call the main function with the video path and other parameters
    main(video_file_path, hf_token, whisper_model)