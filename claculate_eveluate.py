import json

results_file = "evaluation_results_gemma_v1.json"
with open(results_file, 'r', encoding='utf-8') as f: 
    data = json.load(f)

total_em = sum(item['exact_match'] for item in data)
total_f1 = sum(item['f1_score'] for item in data)
num_questions = len(data)

avg_em = (total_em / num_questions) * 100 if num_questions > 0 else 0
avg_f1 = (total_f1 / num_questions) * 100 if num_questions > 0 else 0
avg_time = sum(item['processing_time_seconds'] for item in data) / num_questions if num_questions > 0 else 0

print(f"Total Questions: {num_questions}")
print(f"Average Exact Match (EM): {avg_em:.2f}%")
print(f"Average F1 Score: {avg_f1:.2f}%")
print(f"Average Processing Time per Question: {avg_time:.2f} seconds")