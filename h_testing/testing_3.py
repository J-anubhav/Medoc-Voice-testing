import os
import re
import string
import numpy as np
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

# --- Setup: Load API Key ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        exit()
else:
    print("Error: GOOGLE_API_KEY not found in environment variables or .env file.")
    print("Please create a .env file in the same directory as this script and add the line: GOOGLE_API_KEY='your_api_key'")
    exit()

def normalize_text_for_wer(text):
    """
    FIXED: Proper text normalization that handles medical terms correctly
    """
    text = text.lower()
    
    # Remove speaker labels first
    text = re.sub(r'^(doctor|patient|dr\.?|pt\.?|physician|doc):\s*', '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove special markers and brackets
    text = re.sub(r'\[.*?\]', '', text)  
    text = re.sub(r'\(.*?\)', '', text)  
    
    # Remove bullet points and special characters but keep hyphens for compound words
    text = re.sub(r'[•●▪▫◦‣⁃]', '', text)  # Remove bullet points
    
    # Handle medical compound words BEFORE removing punctuation
    # This ensures "5-ASA" stays together and "Anti-TNF" stays together
    medical_compounds = {
        r'5-asa\b': '5asa',
        r'anti-tnf\b': 'antitnf', 
        r'ht-?1b': 'ht1b',
        r'beta-?blocker': 'betablocker',
        r'steroid-?sparing': 'steroidsparing',
        r'treatment-?resistant': 'treatmentresistant',
        r'first-?line': 'firstline',
        r'non-?steroidal': 'nonsteroidal',
        r'anti-?inflammatory': 'antiinflammatory',
        r'migraine-?associated': 'migraineassociated'
    }
    
    for pattern, replacement in medical_compounds.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Now remove remaining punctuation EXCEPT spaces
    text = re.sub(r'[^\w\s]', '', text)
    
    # Handle medical abbreviations and normalizations
    medical_normalizations = {
        r'\bdegrees?\b': 'degree',
        r'°': 'degree',
        r'\bfahrenheit\b': 'f',
        r'\bcelsius\b': 'c',
        r'\bdr\b': 'doctor',
        r'\bpt\b': 'patient',
        r'\btemp\b': 'temperature',
        r'\bbp\b': 'blood pressure',
        r'\bhr\b': 'heart rate',
        r'\bwbc\b': 'white blood cells',
        r'\brbc\b': 'red blood cells',
        r'\bmg\b': 'milligram',
        r'\bml\b': 'milliliter',
        r'\bcc\b': 'milliliter',
        r'\blbs?\b': 'pounds',
        r'\boz\b': 'ounce',
        r'\bft\b': 'foot',
        r'\bin\b(?!\s+(?:depression|treatment|pain))': 'inch'  # Don't replace "in" when it's a preposition
    }
    
    for pattern, replacement in medical_normalizations.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Handle contractions
    contractions = {
        r"\bdon'?t\b": "do not",
        r"\bwon'?t\b": "will not", 
        r"\bcan'?t\b": "cannot",
        r"\bisn'?t\b": "is not",
        r"\baren'?t\b": "are not",
        r"\bwasn'?t\b": "was not",
        r"\bweren'?t\b": "were not",
        r"\bhaven'?t\b": "have not",
        r"\bhasn'?t\b": "has not",
        r"\bhadn'?t\b": "had not",
        r"\bwouldn'?t\b": "would not",
        r"\bshouldn'?t\b": "should not",
        r"\bcouldn'?t\b": "could not"
    }
    
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Clean up extra spaces and normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def calculate_wer(reference, hypothesis):
    """
    WER calculation with improved normalization
    """
    # Normalize both texts
    ref_normalized = normalize_text_for_wer(reference)
    hyp_normalized = normalize_text_for_wer(hypothesis)
    
    # Debug output to see what's being compared
    print(f"    DEBUG - Reference normalized: {ref_normalized}")
    print(f"    DEBUG - Hypothesis normalized: {hyp_normalized}")
    
    ref_words = ref_normalized.split()
    hyp_words = hyp_normalized.split()
    
    # Handle empty cases
    if not ref_words and not hyp_words:
        return 0.0, []
    if not ref_words:
        return 1.0 if hyp_words else 0.0, [("All Insertion", "---", " ".join(hyp_words))]
    if not hyp_words:
        return 1.0, [("All Deletion", " ".join(ref_words), "---")]

    # Dynamic programming matrix for edit distance
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=int)
    
    # Initialize first row and column
    for i in range(len(ref_words) + 1):
        d[i, 0] = i
    for j in range(len(hyp_words) + 1):
        d[0, j] = j

    # Fill the matrix
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i, j] = min(
                d[i - 1, j] + 1,        # Deletion
                d[i, j - 1] + 1,        # Insertion
                d[i - 1, j - 1] + cost  # Substitution
            )

    # Backtrack to find errors
    incorrect_words = []
    i, j = len(ref_words), len(hyp_words)
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
            # Match - no error
            i -= 1
            j -= 1
        else:
            if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + 1:
                # Substitution
                incorrect_words.append(("Substitution", ref_words[i - 1], hyp_words[j - 1]))
                i -= 1
                j -= 1
            elif j > 0 and d[i, j] == d[i, j - 1] + 1:
                # Insertion
                incorrect_words.append(("Insertion", "---", hyp_words[j - 1]))
                j -= 1
            elif i > 0 and d[i, j] == d[i - 1, j] + 1:
                # Deletion
                incorrect_words.append(("Deletion", ref_words[i - 1], "---"))
                i -= 1
            else:
                break

    incorrect_words.reverse()
    
    # Calculate WER
    total_errors = d[len(ref_words), len(hyp_words)]
    wer = total_errors / len(ref_words)
    
    return wer, incorrect_words

def calculate_additional_metrics(reference, hypothesis):
    """
    Calculate additional evaluation metrics
    """
    ref_normalized = normalize_text_for_wer(reference)
    hyp_normalized = normalize_text_for_wer(hypothesis)
    
    ref_words = set(ref_normalized.split())
    hyp_words = set(hyp_normalized.split())
    
    # Word-level precision, recall, F1
    if not hyp_words:
        precision = 0.0
    else:
        precision = len(ref_words.intersection(hyp_words)) / len(hyp_words)
    
    if not ref_words:
        recall = 0.0
    else:
        recall = len(ref_words.intersection(hyp_words)) / len(ref_words)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'ref_unique_words': len(ref_words),
        'hyp_unique_words': len(hyp_words),
        'common_words': len(ref_words.intersection(hyp_words))
    }

def transcribe_audio_only(audio_path):
    """
    Using Gemini to extract transcription
    """
    if not os.path.exists(audio_path):
        return {"error": f"Audio file not found at {audio_path}"}
    try:
        print(f"INFO: Uploading '{os.path.basename(audio_path)}'...")
        audio_file = genai.upload_file(path=audio_path)
        
        print(f"INFO: Transcribing '{os.path.basename(audio_path)}'...")
        model = genai.GenerativeModel('models/gemini-2.0-flash-lite')
        prompt = """
        You are a medical transcription specialist. Transcribe this audio with MAXIMUM ACCURACY.

        RULES:
        1. ONLY transcribe what is clearly audible
        2. Use DOCTOR: and PATIENT: labels for speakers
        3. Do NOT add extra words or interpret unclear audio
        4. Preserve exact medical terminology
        5. Use [UNCLEAR] for unintelligible parts
        6. Keep transcription proportional to actual speech length
        
        Format: 
        DOCTOR: [speech]
        PATIENT: [speech]
        
        Transcribe now:
        """
        
        response = model.generate_content([prompt, audio_file], request_options={"timeout": 600})
        return response.text
    except Exception as e:
        return {"error": f"An error occurred during transcription: {e}"}

# Test the normalization function
def test_normalization():
    """
    Test function to check normalization behavior
    """
    print("=" * 50)
    print("TESTING NORMALIZATION FUNCTION")
    print("=" * 50)
    
    test_cases = [
        "Aminosalicylates or 5-ASA drugs",
        "amino salicylates or 5-ASA drugs", 
        "Anti-TNF biologics",
        "anti-TNF biologics",
        "Tricyclic Antidepressants",
        "tricyclic antidepressant"
    ]
    
    for test in test_cases:
        normalized = normalize_text_for_wer(test)
        print(f"Original: {test}")
        print(f"Normalized: {normalized}")
        print(f"Words: {normalized.split()}")
        print("-" * 30)

def run_testing_pipeline(audio_dir, groundtruth_dir):
    """
    Medical transcription model evaluation testing pipeline with fixed normalization
    """
    results = []
    total_wer = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    file_count = 0

    # Output folders 
    TRANSCRIPTION_FOLDER = "transcription_3"
    DETAILED_ERRORS_FOLDER = "detailed_errors_3"
    
    for folder in [TRANSCRIPTION_FOLDER, DETAILED_ERRORS_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Find audio files
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(('.wav', '.mp3', '.flac', '.ogg'))]

    if not audio_files:
        print(f"No audio files found in '{audio_dir}'. Please check the directory and file extensions.")
        return

    print("=" * 70)
    print("           FIXED MEDICAL TRANSCRIPTION TESTING PIPELINE")
    print("=" * 70)
    print(f"Found {len(audio_files)} audio files")
    
    # Test normalization first
    test_normalization()
    
    print("=" * 70)

    for audio_file in audio_files:
        file_count += 1
        base_name = os.path.splitext(audio_file)[0].replace('_with_effects', '')
        
        audio_path = os.path.join(audio_dir, audio_file)
        groundtruth_path = os.path.join(groundtruth_dir, base_name + '.txt')

        print(f"\nProcessing {file_count}/{len(audio_files)}: {audio_file}")

        # Check ground truth file exists
        if not os.path.exists(groundtruth_path):
            print(f"Warning: Ground truth for '{audio_file}' not found. Skipping.")
            continue

        try:
            with open(groundtruth_path, 'r', encoding='utf-8') as f:
                ground_truth_text = f.read().strip()
        except Exception as e:
            print(f"❌ Error reading ground truth file: {e}")
            continue

        # Transcription
        model_transcription_raw = transcribe_audio_only(audio_path)
        
        if isinstance(model_transcription_raw, dict) and 'error' in model_transcription_raw:
            print(f"❌ Error processing '{audio_file}': {model_transcription_raw['error']}")
            continue

        # Save transcription
        try:
            script_number = re.search(r'\d+', base_name)
            if script_number:
                script_number = script_number.group()
                transcription_filename = f"transcription_{script_number}.txt"
            else:
                transcription_filename = f"transcription_{base_name}.txt"
                
            transcription_filepath = os.path.join(TRANSCRIPTION_FOLDER, transcription_filename)
            with open(transcription_filepath, 'w', encoding='utf-8') as f:
                f.write(model_transcription_raw)
            print(f"✅ Transcription saved: {transcription_filename}")
        except Exception as e:
            print(f"⚠️  Could not save transcription: {e}")

        # WER calculation with normalization
        wer, incorrect_words = calculate_wer(ground_truth_text, model_transcription_raw)
        additional_metrics = calculate_additional_metrics(ground_truth_text, model_transcription_raw)
        
        total_wer += wer
        total_precision += additional_metrics['precision']
        total_recall += additional_metrics['recall']
        total_f1 += additional_metrics['f1']

        print(f"✅ WER: {wer:.2%}")
        print(f"   Precision: {additional_metrics['precision']:.2%}")
        print(f"   Recall: {additional_metrics['recall']:.2%}")
        print(f"   F1-Score: {additional_metrics['f1']:.2%}")
        
        if incorrect_words:
            substitutions = sum(1 for e in incorrect_words if e[0]=='Substitution')
            insertions = sum(1 for e in incorrect_words if e[0]=='Insertion')
            deletions = sum(1 for e in incorrect_words if e[0]=='Deletion')
            
            print(f"   Errors: {len(incorrect_words)} (S:{substitutions} | I:{insertions} | D:{deletions})")
            
            # Save detailed error analysis
            try:
                if script_number:
                    error_filename = f"errors_{script_number}.txt"
                else:
                    error_filename = f"errors_{base_name}.txt"
                    
                error_filepath = os.path.join(DETAILED_ERRORS_FOLDER, error_filename)
                with open(error_filepath, 'w', encoding='utf-8') as f:
                    f.write(f"File: {audio_file}\n")
                    f.write(f"WER: {wer:.2%}\n")
                    f.write(f"Total Errors: {len(incorrect_words)}\n")
                    f.write(f"Error Breakdown - Substitutions: {substitutions}, Insertions: {insertions}, Deletions: {deletions}\n\n")
                    f.write("Original Ground Truth:\n")
                    f.write(f"{ground_truth_text}\n\n")
                    f.write("Normalized Ground Truth:\n")
                    f.write(f"{normalize_text_for_wer(ground_truth_text)}\n\n")
                    f.write("Raw Transcription:\n")
                    f.write(f"{model_transcription_raw}\n\n")
                    f.write("Normalized Transcription:\n")
                    f.write(f"{normalize_text_for_wer(model_transcription_raw)}\n\n")
                    f.write("Detailed Errors:\n")
                    for i, (error_type, ref_word, hyp_word) in enumerate(incorrect_words, 1):
                        f.write(f"{i}. {error_type}: '{ref_word}' -> '{hyp_word}'\n")
                print(f"✅ Error analysis saved: {error_filename}")
            except Exception as e:
                print(f"⚠️  Could not save detailed error analysis: {e}")

        # Store result
        result = {
            'Audio File': audio_file,
            'WER': f"{wer:.2%}",
            'Precision': f"{additional_metrics['precision']:.2%}",
            'Recall': f"{additional_metrics['recall']:.2%}",
            'F1-Score': f"{additional_metrics['f1']:.2%}",
            'Total Errors': len(incorrect_words),
            'Substitutions': sum(1 for e in incorrect_words if e[0] == 'Substitution'),
            'Insertions': sum(1 for e in incorrect_words if e[0] == 'Insertion'),
            'Deletions': sum(1 for e in incorrect_words if e[0] == 'Deletion'),
            'Ref Words': len(normalize_text_for_wer(ground_truth_text).split()),
            'Hyp Words': len(normalize_text_for_wer(model_transcription_raw).split()),
            'Common Words': additional_metrics['common_words']
        }
        
        results.append(result)

    # Final Results
    print("\n" + "=" * 70)
    print("                      OVERALL RESULTS")
    print("=" * 70)
    
    if file_count > 0:
        processed_count = len(results)
        if processed_count > 0:
            average_wer = total_wer / processed_count
            average_precision = total_precision / processed_count
            average_recall = total_recall / processed_count
            average_f1 = total_f1 / processed_count
            
            print(f"Average Metrics on {processed_count} file(s):")
            print(f"  • Word Error Rate (WER): {average_wer:.2%}")
            print(f"  • Precision: {average_precision:.2%}")
            print(f"  • Recall: {average_recall:.2%}")
            print(f"  • F1-Score: {average_f1:.2%}")
            print()
            
            # Detailed results table
            df = pd.DataFrame(results)
            pd.set_option('display.max_rows', 500)
            pd.set_option('display.max_columns', 500)
            pd.set_option('display.width', 150)
            pd.set_option('display.max_colwidth', 80)
            print(df)

            # Save main results
            output_filename = 'fixed_evaluation_report.csv'
            df.to_csv(output_filename, index=False, encoding='utf-8-sig')
            
            print(f"\n✅ Individual transcriptions saved in '{TRANSCRIPTION_FOLDER}/' folder.")
            print(f"✅ Detailed error analysis saved in '{DETAILED_ERRORS_FOLDER}/' folder.")
            print(f"✅ Fixed evaluation report saved to '{output_filename}'")
        else:
            print("No files were successfully processed.\n")
    else:
        print("No audio files were found to process.")

if __name__ == "__main__":
    AUDIO_FOLDER = "../h_audio"
    GROUNDTRUTH_FOLDER = "../h_Groundtruth"

    if not os.path.isdir(AUDIO_FOLDER):
        print(f"Error: Audio directory not found at '{AUDIO_FOLDER}'")
    elif not os.path.isdir(GROUNDTRUTH_FOLDER):
        print(f"Error: Groundtruth directory not found at '{GROUNDTRUTH_FOLDER}'")
    else:
        run_testing_pipeline(AUDIO_FOLDER, GROUNDTRUTH_FOLDER)