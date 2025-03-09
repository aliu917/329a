import json

MATH_COT_PROMPT = """Problem:
Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}

Solution:
The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\nFinal Answer: The final answer is $[2,5)$

Problem:
If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$

Solution:
We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\nFinal Answer: The final answer is $24$

Problem:
Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?

Solution:
If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:\n\\begin{align*}\n30n&=480\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\nFinal Answer: The final answer is $16$

Problem:
If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero.

Solution:
If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\nFinal Answer: The final answer is $-\\frac{2}{3}$"""

# From HW1
def math500_prompt(add_prompt=None) -> str:
    prompt = (
        "You are a language model that solves math problems. Think step by step. Use the below format when responding."
        + "\n"
        + MATH_COT_PROMPT
    )
    if add_prompt is not None:
        prompt += "\n" + add_prompt
    return prompt

def teacher_iteration_prompt(question, attempt, solution, history):
    history_str = "\n".join([
        f"""========== ROUND {result['round']} ==========
Your Feedback: {result['prev_feedback']}
Attempt: {result['pred_steps']}
Correct: {result['correct']}"""
        for result in history
    ])
    return f"""You have been trying to help a student correctly answer the following math question:
{question}
This is the correct solution:
{solution}
You are refining your feedback to the student over the course of several rounds. In each round, the student remembers nothing about the previous rounds. You provide the student with feedback/hints on the question, the student provides you with their solution attempt, and a verifier tells you whether the solution is correct or incorrect. Here are the rounds that have happened so far:
{history_str}
Compare the reasoning step process of the most recent attempt with the steps of the correct solution repeated below:
{solution}
Based on any errors you notice in the attempt and what feedback worked well in the past, provide some feedback for improving the reasoning steps. The format of the response should be "Feedback: <how to fix the reasoning>"."""

def feedback_from_baseline(num_samples=100): 
    with open("runs/baseline/base_results.json", "r") as file:
        feedback_data = json.load(file)
    with  open("runs/baseline/results.json", "r") as file:
        steps_data = json.load(file)
    

    example_sections = []

    for feedback, step in zip(feedback_data[:num_samples], steps_data[:num_samples]):


        question = feedback["question"]
        label = feedback["answer"]
        teacher_feedback = feedback["feedback"]
        student_steps = step["pred_steps"]

        example_section = f"""Example:
            Question: {question}
            Attempt: {student_steps}
            Correct Solution: {label}
            Feedback:
            {teacher_feedback}
        """

        example_sections.append(example_section)
    

    all_examples_text = "\n".join(example_sections)

    return all_examples_text

def teacher_best_feedback_prompt(question, student, label, num_feedback_samples=100): 
    all_examples = feedback_from_baseline(num_samples=num_feedback_samples)


    return f"""You are given multiple reasoning attempts along with correct solutions and feedback. Use these examples to determine the best way to evaluate reasoning steps.

### Learn from these examples:
{all_examples}


### Now, apply what you have learned:
Given what you have learned from the examples above, evaluate the following reasoning attempt for a new question: {question}

Attempt: {student}

Correct solution: {label}

Compare the reasoning attempt step process with the correct solution and determine if the solution is correct; if not, provide some feedback for improving the reasoning steps.
The format of the response should be "Correct: <yes/no> Feedback: <how to fix the reasoning>". The feedback should be a general axiom or statement, and should not reference a person, the reasoning attempt, or the correct solution specifically.

Correct: """

def feedback_from_dataset(dataset_path, max_samples=None): 
    with open(dataset_path, "r") as file:
        dataset = json.load(file)
    if max_samples is None:
        max_samples = len(dataset)
    
    example_sections = []

    for result in dataset[:max_samples]:
        question = result["question"]
        label = result["answer"]
        teacher_feedback = result["prev_feedback"]
        student_steps = result["pred_steps"]

        example_section = f"""Example:
            Question: {question}
            Attempt: {student_steps}
            Correct Solution: {label}
            Feedback:
            {teacher_feedback}
        """

        example_sections.append(example_section)
    all_examples_text = "\n".join(example_sections)
    return all_examples_text

def teacher_best_from_dataset_prompt(dataset_path, question, student, label, history=None): 
    all_examples = feedback_from_dataset(dataset_path, max_samples=100)


    return f"""You are given multiple reasoning attempts along with correct solutions and feedback. Use these examples to determine the best way to evaluate reasoning steps.

### Learn from these examples:
{all_examples}


### Now, apply what you have learned:
Given what you have learned from the examples above, evaluate the following reasoning attempt for a new question: {question}

Attempt: {student}

Correct solution: {label}

Compare the reasoning attempt step process with the correct solution and determine if the solution is correct; if not, provide some feedback for improving the reasoning steps.
The format of the response should be "Correct: <yes/no> Feedback: <how to fix the reasoning>". The feedback should be a general axiom or statement, and should not reference a person, the reasoning attempt, or the correct solution specifically.

Correct: """


##### Eval time student prompts

def build_next_student_example(result_dict, keys):
    all_keys = ["question"] + keys
    all_elems = []
    for k in all_keys:
        if k not in result_dict:
            print(f"Error: requested key {k} not in results for question {result_dict['question']}")
            continue
        all_elems.append(k.replace('_', ' ').title() + ": " + result_dict[k])
    return "\n\n".join(all_elems)


def get_eval_student_context(result_path, keys, num_samples=10):
    with open(result_path, "r") as file:
        train_result_data = json.load(file)[:num_samples]

    all_examples = [build_next_student_example(d, keys) for d in train_result_data]
    all_examples_str = "\n\n".join(all_examples)

    keys_str = ", ".join(keys[:-1])
    if len(keys) > 1:
        keys_str += ", and " + keys[-1]
    else:
        keys_str = keys[-1]

    return f"""You are given multiple example questions along with their corresponding {keys_str}. Use these examples to determine the best way to approach, reason, and correctly answer the question.

### Learn from these examples:
{all_examples_str}


### Now, apply what you have learned:
Given what you have learned from the examples above, answer the following question: """
