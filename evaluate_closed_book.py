import os
import re
import time
import json 
from collections import Counter
from langchain_community.llms import Ollama 

print("Libraries imported for evaluate_closed_book.")

def normalize_answer(s):
    s = str(s).lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s]', '', s)
    s = ' '.join(s.split())
    return s

def calculate_f1(prediction_tokens, ground_truth_tokens): 
    if not prediction_tokens or not ground_truth_tokens: 
        return 0.0
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def get_best_f1_for_prediction(prediction_norm, reference_answers_norm_list):
    if not prediction_norm: return 0.0
    
    pred_tokens = prediction_norm.split()
    max_f1 = 0.0
    for ref_norm in reference_answers_norm_list:
        if not ref_norm: continue # Skip empty reference answers
        ref_tokens = ref_norm.split()
        max_f1 = max(max_f1, calculate_f1(pred_tokens, ref_tokens))
    return max_f1

def calculate_exact_match(prediction_norm, reference_answers_norm_list):
    return any(prediction_norm == ref_norm for ref_norm in reference_answers_norm_list)

# Function to Load Test Data
def load_test_data(questions_filepath, answers_filepath):
    questions = []
    with open(questions_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            questions.append(line.strip())
    
    reference_answers_list_of_lists = []
    with open(answers_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            reference_answers_list_of_lists.append([ans.strip() for ans in line.strip().split(';')])
    
    if len(questions) != len(reference_answers_list_of_lists):
        raise ValueError("Mismatch between the number of questions and reference answer sets.")
    return questions, reference_answers_list_of_lists

# Main Evaluation Logic
if __name__ == "__main__":
    print("--- Starting Closed-Book LLM Evaluation ---")

    # Initialize LLM Only (Closed-Book)
    OLLAMA_MODEL = "gemma3:4b" 
    print(f"\nInitializing LLM '{OLLAMA_MODEL}' for Closed-Book evaluation...")
    print(f"Ensure Ollama application is running with the '{OLLAMA_MODEL}' model pulled.")
    start_llm_init_time = time.time()
    llm = None 
    try:
        llm = Ollama(model=OLLAMA_MODEL)
        print(f"LLM '{OLLAMA_MODEL}' connected/initialized in {time.time() - start_llm_init_time:.2f} seconds.")
    except Exception as e:
        print(f"Error initializing Ollama LLM. Is Ollama running and '{OLLAMA_MODEL}' pulled? Error: {e}")
        exit()

    TEST_QUESTIONS_FILE = "data/test/questions.txt"
    TEST_ANSWERS_FILE = "data/test/reference_answers.txt"
    RESULTS_JSON_FILE = "evaluation_results_closed_book_gemma.json" 

    # 1. Load Test Data
    print(f"\nLoading test data from {TEST_QUESTIONS_FILE} and {TEST_ANSWERS_FILE}...")
    try:
        questions, reference_answers_raw_lists = load_test_data(TEST_QUESTIONS_FILE, TEST_ANSWERS_FILE)
        print(f"Loaded {len(questions)} test questions and their reference answers.")
    except Exception as e:
        print(f"Error loading test data: {e}. Exiting.")
        exit()

    # 2. Perform Evaluation
    all_results_data = [] 
    total_em_score = 0
    total_f1_score = 0
    total_processing_time = 0

    print("\n--- Processing Test Questions (Closed-Book Mode) ---")
    for i, question_text in enumerate(questions):
        print(f"\nProcessing Question {i+1}/{len(questions)}: \"{question_text}\"")
        current_question_start_time = time.time()
        
        predicted_answer_raw_str = ""
        # In closed-book, sources are not applicable in the same way as RAG
        source_docs_info = ["CLOSED_BOOK_MODE"] 
        
        try:
            closed_book_prompt = f"""Answer the following question based on your general knowledge. Provide a concise answer.

Question: {question_text}

Answer:"""
            
            predicted_answer_raw_str = llm.invoke(closed_book_prompt)
        
        except Exception as e:
            print(f"  ERROR invoking LLM for this question: {e}")
            predicted_answer_raw_str = "[LLM_INVOCATION_ERROR]" 
            source_docs_info = ["CLOSED_BOOK_ERROR"]

        current_question_processing_time = time.time() - current_question_start_time
        total_processing_time += current_question_processing_time

        normalized_predicted_answer = normalize_answer(predicted_answer_raw_str)
        current_raw_references = reference_answers_raw_lists[i]
        normalized_reference_answers = [normalize_answer(ref) for ref in current_raw_references]

        em_score_for_q = 1 if calculate_exact_match(normalized_predicted_answer, normalized_reference_answers) else 0
        f1_score_for_q = get_best_f1_for_prediction(normalized_predicted_answer, normalized_reference_answers)
        
        total_em_score += em_score_for_q
        total_f1_score += f1_score_for_q

        print(f"  Raw Predicted: \"{predicted_answer_raw_str[:250]}{'...' if len(predicted_answer_raw_str) > 250 else ''}\"")
        print(f"  EM: {em_score_for_q}, F1: {f1_score_for_q:.4f}, Time: {current_question_processing_time:.2f}s")

        all_results_data.append({
            "question_id": i + 1,
            "question": question_text,
            "predicted_answer_raw": predicted_answer_raw_str,
            "normalized_prediction": normalized_predicted_answer,
            "reference_answers_raw": current_raw_references,
            "normalized_references": normalized_reference_answers,
            "exact_match": em_score_for_q,
            "f1_score": f1_score_for_q,
            "processing_time_seconds": current_question_processing_time,
            "retrieved_sources": source_docs_info 
        })

    # 3. Calculate and Print Final Averages
    num_questions = len(questions)
    average_em = (total_em_score / num_questions) * 100 if num_questions > 0 else 0
    average_f1 = (total_f1_score / num_questions) * 100 if num_questions > 0 else 0
    average_time_per_q = total_processing_time / num_questions if num_questions > 0 else 0

    print("\n\n--- Overall Closed-Book Evaluation Summary ---")
    print(f"Total Questions Evaluated: {num_questions}")
    print(f"Average Exact Match (EM): {average_em:.2f}%")
    print(f"Average F1 Score: {average_f1:.2f}%")
    print(f"Average Processing Time per Question: {average_time_per_q:.2f} seconds")
    print(f"Total Evaluation Time: {total_processing_time:.2f} seconds")

    # 4. Save detailed results to JSON
    try:
        with open(RESULTS_JSON_FILE, "w", encoding="utf-8") as f_out:
            json.dump(all_results_data, f_out, indent=2, ensure_ascii=False)
        print(f"\nDetailed closed-book evaluation results saved to: {RESULTS_JSON_FILE}")
    except Exception as e:
        print(f"\nError saving detailed closed-book results to JSON: {e}")

    print("\n--- Closed-Book Evaluation Script Finished ---")