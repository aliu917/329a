from project.utils import QuestionHistory

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

def default_prompt(add_prompt=""):
    prompt = """Think step by step and then provide the final answer boxed in the format: '\\boxed{final answer}'"""
    if add_prompt:
        prompt += "\n" + add_prompt
    return prompt

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

def teacher_default_prompt(question, student, label):
    return f"""You are given a reasoning attempt for answering the following question: {question}

Attempt: {student}

Correct solution: {label}

Compare the reasoning attempt step process with the correct solution and determine if the solution is correct; if not, provide some feedback for improving the reasoning steps.
The format of the response should be "Correct: <yes/no> Feedback: <how to fix the reasoning>". The feedback should be a general axiom or statement, and should not reference a person, the reasoning attempt, or the correct solution specifically."""

def teacher_iteration_prompt(h: QuestionHistory):
    num_rounds = len(h.feedback_history)
    history_str = "\n".join([
        f"""========== ROUND {round + 1} ==========
Attempt: {attempt}
Correct: {is_correct}
Your Feedback: {feedback}"""
        for round, attempt, is_correct, feedback in zip(range(len(num_rounds)), h.prediction_steps_history, h.is_correct_history, h.feedback_history)
    ])
    return f"""You have been trying to help a student correctly answer the following math question:
{h.question}
This is the correct solution:
{h.ground_truth_steps}
You are refining your feedback to the student over the course of several rounds. In each round, you provide the student with hints on the question, the student provides you with their solution attempt, and you det
"""


if __name__ == '__main__':
    out = default_prompt(["focus on the geometric relationships more clearly and ensure they are correctly applying similarity and distance formulas. Specifically, they should re-evaluate the triangle relationships and correctly analyze the tangents from the circle to the sides of the trapezoid. They should also confirm that the calculations for \(m^2\) match the derived expressions for \(x^2\)."])
    print(out)