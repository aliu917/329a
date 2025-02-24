import json


def default_prompt(add_prompt=""):
    prompt = """Think step by step and then provide the final answer boxed in the format: '\\boxed{final answer}'"""
    if add_prompt:
        prompt += "\n" + add_prompt
    return prompt

def feedback_prompt(feedback):
    if not feedback:
        return ""
    if isinstance(feedback, str):
        text = feedback
    elif isinstance(feedback, list):
        text = "\n- ".join(feedback)
    elif isinstance(feedback, dict):
        text = "\n".join([f"- {key}: {val}" for key, val in feedback.items()])
    return f"""Consider the following hints and feedback when reasoning about the response:\n{text}"""


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



def teacher_default_prompt(question, student, label):
        return f"""You are given a reasoning attempt for answering the following question: {question}

    Attempt: {student}

    Correct solution: {label}

    Compare the reasoning attempt step process with the correct solution and determine if the solution is correct; if not, provide some feedback for improving the reasoning steps.
    The format of the response should be "Correct: <yes/no> Feedback: <how to fix the reasoning>". The feedback should be a general axiom or statement, and should not reference a person, the reasoning attempt, or the correct solution specifically.

    Correct: """


if __name__ == '__main__':
    out = default_prompt(["focus on the geometric relationships more clearly and ensure they are correctly applying similarity and distance formulas. Specifically, they should re-evaluate the triangle relationships and correctly analyze the tangents from the circle to the sides of the trapezoid. They should also confirm that the calculations for \(m^2\) match the derived expressions for \(x^2\)."])
    print(out)