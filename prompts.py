def default_prompt():
    return """Think step by step and then provide the final answer boxed in the format: '\\boxed{final answer}'"""

def teacher_default_prompt(question, student, label):
    return f"""You are given a student approach for answering the following question: {question}

Student: {student}

Correct solution: {label}

Compare the student's reasoning step process with the correct solution and determine if there solution is correct, and if not, provide some feedback for improving the reasoning steps.
The format of the response should be "Correct: <yes/no> Feedback: <how to fix the reasoning>".

Correct: """