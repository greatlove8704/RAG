import os
import time
import sys
import csv 
from rag_system_gemma import get_gemma_rag_chain 

if __name__ == "__main__":
    print("--- Generating Official System Output ---")
    
    if len(sys.argv) < 2:
        print("ERROR: Missing path to the official questions CSV file.")
        print("Usage: python generate_official_output.py <path_to_official_questions.csv>")
        print("Example: python generate_official_output.py test_questions.csv")
        sys.exit(1) 
            
    official_questions_file_path = sys.argv[1]
    
    output_file_path = "system_output_1.txt" 

    print(f"Input questions CSV file: {official_questions_file_path}")
    print(f"Output answers will be saved to: {output_file_path}")

    # Initialize RAG Pipeline V3
    print("\nInitializing RAG pipeline (Your Best System - Variation 3)...")
    qa_chain = get_gemma_rag_chain() 
    if qa_chain is None:
        print("CRITICAL ERROR: Failed to initialize RAG chain. Cannot generate outputs. Exiting.")
        sys.exit(1)
    print("RAG pipeline initialized successfully and ready.")

    # Load Questions from the test_question.csv
    questions = []
    print(f"\nLoading questions from '{official_questions_file_path}'...")
    try:
        with open(official_questions_file_path, mode='r', encoding='utf-8', newline='') as csvfile:
            
            csv_reader = csv.reader(csvfile)
            header = next(csv_reader, None)

            if header:
                print(f"  CSV Header: {header}")
                question_column_index = 0 
            else:
                print("  Warning: CSV file has no header row. Assuming questions are in the first column.")
                question_column_index = 0

            line_number = 1
            for row in csv_reader:
                line_number += 1
                if row: 
                    try:
                        question_text = row[question_column_index].strip()
                        if question_text: 
                            questions.append(question_text)
                    except IndexError:
                        print(f"  Warning: Row {line_number} in CSV does not have column index {question_column_index}. Row content: {row}. Skipping.")
        
        print(f"Loaded {len(questions)} questions from CSV file.")

    except FileNotFoundError:
        print(f"CRITICAL ERROR: The official questions CSV file was not found at '{official_questions_file_path}'. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load official questions CSV file: {e}. Exiting.")
        sys.exit(1)

    if not questions:
        print("No questions were loaded from the CSV input file. Output file will be empty. Exiting.")
        sys.exit(1)

    # Process Questions and Collect Raw Predicted Answers
    predicted_answers_raw_list = [] 
    total_processing_time = 0

    print("\n--- Processing Official Questions ---")
    for i, question_text in enumerate(questions):
        question_snippet = question_text[:70] + "..." if len(question_text) > 70 else question_text
        print(f"Processing Question {i+1}/{len(questions)}: \"{question_snippet}\"")
        
        individual_question_start_time = time.time()
        predicted_answer_for_this_question = "[RAG_SYSTEM_ERROR_DEFAULT_ANSWER]" 
        
        try:
            rag_output = qa_chain.invoke({"query": question_text})
            predicted_answer_for_this_question = rag_output.get('result', "[ERROR_NO_RESULT_FIELD_IN_RAG_OUTPUT]")
        except Exception as e:
            print(f"  WARNING: Error invoking RAG chain for question {i+1}: {e}")
            
        individual_question_processing_time = time.time() - individual_question_start_time
        total_processing_time += individual_question_processing_time
        print(f"  Answered in {individual_question_processing_time:.2f}s. Predicted: \"{str(predicted_answer_for_this_question)[:60]}...\"")
        
        predicted_answers_raw_list.append(str(predicted_answer_for_this_question).strip())

    print(f"\n--- Finished Processing All Questions ---")
    print(f"Total processing time for {len(questions)} questions: {total_processing_time:.2f} seconds.")
    average_time = total_processing_time / len(questions) if questions else 0
    print(f"Average time per question: {average_time:.2f} seconds.")

    # Write Raw Answers to the Output File
    try:
        with open(output_file_path, "w", encoding="utf-8") as f_out:
            for ans_text in predicted_answers_raw_list:
                f_out.write(ans_text + "\n") 
        print(f"\nOfficial system output successfully saved to: {output_file_path}")
    except Exception as e:
        print(f"\nCRITICAL ERROR: Failed to save official system output to '{output_file_path}': {e}")

    print("\n--- Official Output Generation Script Finished ---")