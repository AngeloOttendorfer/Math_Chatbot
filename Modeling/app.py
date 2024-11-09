import re

from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel

# Load GPT-2 model and tokenizer
gpt2_model_path = 'C:\\Users\\angel\\PycharmProjects\\pythonProject\\Modeling\\trained_models\\GPT2\\Model_32000_8_8\\checkpoint-8000'
gpt_neo_model_path = 'C:\\Users\\angel\\PycharmProjects\\pythonProject\\Modeling\\trained_models\\GPT-NEO\Model_32000_8_8\\checkpoint-8000'

model = GPT2LMHeadModel.from_pretrained(gpt2_model_path)
# model = AutoModelForCausalLM.from_pretrained(gpt_neo_model_path)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125m')

app = Flask(__name__)


def preprocess_input(question, tokenizer):
    question = f"QUESTION: {question}\nFINAL SOLUTION:\n"
    input_ids = tokenizer.encode(question, return_tensors='pt')
    return input_ids


def generate_solution(model, input_ids):
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=300,
                                    num_return_sequences=1,
                                    temperature=0.7,
                                    do_sample=True,
                                    pad_token_id=tokenizer.eos_token_id)
    return output_ids


def postprocess_output(output_ids, tokenizer):
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    final_solution_index = output_text.find("FINAL SOLUTION:")
    if final_solution_index != -1:
        output_text = output_text[final_solution_index + len("FINAL SOLUTION:"):].strip()
        # Use a regular expression to find the first occurrence of content ending with a single `$`
        match = re.search(r'(\$.*?\$)', output_text)
        if match:
            output_text = match.group(0)
    return output_text


def generate_self_consistency(model, input_ids, num_generations=5):
    answers = []
    for _ in range(num_generations):
        output_text = generate_solution(model, input_ids)
        processed_text = postprocess_output(output_text, tokenizer)
        answers.append(processed_text)
    # Find the most frequent answer
    most_common_answer = max(set(answers), key=answers.count)
    return most_common_answer


def generate_active_prompt(model, input_ids, correct_answer, num_generations=5):
    output_text = generate_solution(model, input_ids)
    processed_text = postprocess_output(output_text, tokenizer)
    for _ in range(num_generations):
        is_correct = processed_text.strip() == correct_answer.strip()
        if is_correct:
            return processed_text
        else:
            error_message = f"You have made a calculation error. The correct answer should be {correct_answer}. Please try again, ensuring that all steps lead to the correct solution."
            revised_input = preprocess_input(f"{error_message} {input_ids}", tokenizer)
            revised_text = generate_solution(model, revised_input)
            processed_text = postprocess_output(revised_text, tokenizer)

    return processed_text


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data['question']
    prompting_strategy = data.get('prompting_strategy', 'problem-answer')
    correct_answer = "$\\left\\{\\left\\{x\\to -\\frac{6}{17}\\right\\}\\right\\}$"  # Placeholder correct answer

    input_ids = preprocess_input(question, tokenizer)

    if prompting_strategy == 'problem-answer':
        output_text = generate_solution(model, input_ids)
        answer = postprocess_output(output_text, tokenizer)
        is_correct = answer.strip() == correct_answer.strip()

    elif prompting_strategy == 'self-consistency':
        answer = generate_self_consistency(model, input_ids)
        is_correct = answer.strip() == correct_answer.strip()

    elif prompting_strategy == 'active-prompt':
        answer = generate_active_prompt(model, input_ids, correct_answer)
        is_correct = answer.strip() == correct_answer.strip()

    return jsonify({'question': question, 'solution': answer, 'correct': is_correct})


@app.route('/switch_model', methods=['POST'])
def switch_model():
    global model, tokenizer
    selected_model = request.json['model']

    if selected_model == 'gpt2':
        model = GPT2LMHeadModel.from_pretrained(gpt2_model_path)
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    elif selected_model == 'gpt_neo':
        model = AutoModelForCausalLM.from_pretrained(gpt_neo_model_path)
        tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125m')

    return jsonify({'message': f"Switched to {selected_model} model."})


if __name__ == '__main__':
    app.run(debug=True)
