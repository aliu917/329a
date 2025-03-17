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
Based on any errors you notice in the attempt and what feedback worked well in the past, provide some feedback for improving the reasoning steps without giving away the final answer. The format of the response should be "Feedback: <how to fix the reasoning>"."""

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

def cot_to_steps_prompt(question, sol_cot):
    return """You are given a solution chain of thought reasoning process for a question. Break down the reasoning process into individual discrete steps. Use step 0 to refer to any setup steps for defining variables that are unrelated to the solution reasoning progress. Finally, use "Final answer" to denote the final answer of the given solution process.

### Learn from these examples:

Question: A  piece of paper measures 4 units by 5 units. Several s are drawn  to the edges of the paper. A rectangle determined by the s of some of these lines is called ''basic'' if
:(i) all four sides of the rectangle are segments of drawn line segments, and
:(ii) no s of drawn lines lie inside the rectangle.
Given that the total length of all lines drawn is exactly 2007 units, let \$N\$ be the maximum possible number of basic rectangles determined. Find the  when \$N\$ is divided by 1000.

Solution chain-of-thought reasoning: =
Denote the number of horizontal lines drawn as \$x\$, and the number of vertical lines drawn as \$y\$. The number of basic rectangles is \$(x - 1)(y - 1)\$. \$5x + 4y = 2007 \Longrightarrow y = \\frac{2007 - 5x}{4}\$. Substituting, we find that \$(x - 1)\left(-\\frac 54x + \\frac{2003}4\right)\$.
this to get a quadratic, \$-\\frac 54x^2 + 502x - \\frac\{2003}4\$. Use \$\\frac{-b}{2a}\$ to find the maximum possible value of the quadratic: \$x = \\frac{-502}{-2 \cdot \\frac 54} = \\frac{1004}5 \approx 201\$. However, this gives a non-integral answer for \$y\$. The closest two values that work are \$(199,253)\$ and \$(203,248)\$.
We see that \$252 \cdot 198 = 49896 > 202 \cdot 247 = 49894\$. The solution is \$\\boxed{896}\$.

Solution steps: 
Step 0: Denote the number of horizontal lines drawn as \$x\$, and the number of vertical lines drawn as \$y\$.
Step 1: The number of basic rectangles is \$(x - 1)(y - 1)\$.
Step 2: \$5x + 4y = 2007 \Longrightarrow y = \\frac{2007 - 5x}{4}\$.
Step 3: Substituting y, we find that \$(x - 1)\left(-\\frac 54x + \\frac{2003}4\\right)\$.
Step 4: To get a quadratic, \$-\\frac 54x^2 + 502x - \\frac{2003}4\$.
Step 5: Use \$\\frac{-b}{2a}\$ to find the maximum possible value of the quadratic: \$x = \\frac{-502}{-2 \cdot \\frac 54} = \\frac{1004}5 \approx 201\$.
Step 6: his gives a non-integral answer for \$y\$. The closest two values that work are \$(199,253)\$ and \$(203,248)\$.
Step 7: We see that \$252 \cdot 198 = 49896 > 202 \cdot 247 = 49894\$.
Final answer: \(\\boxed{448}\)

### Now, apply what you have learned:

Question: """ + f"""{question}

Solution chain-of-thought reasoning: {sol_cot}

Solution steps:
"""


def student_step_func(problem, feedback, history):
    prompt = """You are given a question. Reason about the best solution using a step by step process and output the steps as well as the final solution in a boxed format: '\\boxed{final answer}'
    
### Here is an example of a question and the appropriate response:

Question: A  piece of paper measures 4 units by 5 units. Several s are drawn  to the edges of the paper. A rectangle determined by the s of some of these lines is called ''basic'' if
:(i) all four sides of the rectangle are segments of drawn line segments, and
:(ii) no s of drawn lines lie inside the rectangle.
Given that the total length of all lines drawn is exactly 2007 units, let \$N\$ be the maximum possible number of basic rectangles determined. Find the  when \$N\$ is divided by 1000.

Response:
Step 0: Denote the number of horizontal lines drawn as \$x\$, and the number of vertical lines drawn as \$y\$.
Step 1: The number of basic rectangles is \$(x - 1)(y - 1)\$.
Step 2: \$5x + 4y = 2007 \Longrightarrow y = \\frac{2007 - 5x}{4}\$.
Step 3: Substituting y, we find that \$(x - 1)\left(-\\frac 54x + \\frac{2003}4\\right)\$.
Step 4: To get a quadratic, \$-\\frac 54x^2 + 502x - \\frac{2003}4\$.
Step 5: Use \$\\frac{-b}{2a}\$ to find the maximum possible value of the quadratic: \$x = \\frac{-502}{-2 \cdot \\frac 54} = \\frac{1004}5 \approx 201\$.
Step 6: This gives a non-integral answer for \$y\$. The closest two values that work are \$(199,253)\$ and \$(203,248)\$.
Step 7: We see that \$252 \cdot 198 = 49896 > 202 \cdot 247 = 49894\$.
Final answer: \(\\boxed{448}\)"""

    if history:
        prompt += "\n\n### Learn from these additional context examples:" + history
    if feedback:
        prompt += "\n\nWhen formulating the response, consider this feedback or hint provided by an expert to help guide the reasoning process for this question: " + feedback
    prompt += "\nRemember to think step by step and then provide the final answer boxed in the format: '\\boxed{final answer}'."
    prompt += "\n\nQuestion: " + problem + "\n\nResponse:"
    return prompt


def teacher_step_prompt(question, student, label, history=None):
    return """You are given a reasoning attempt for answering a complex math question. Given the question and a step-by-step reference of the correct solutions, evaluate each step of the solution response against the student attempt and determine if the student was able to understand the reasoning step. Start from step 1 of the teacher solution.
For the first teacher step that is not present in the student attempt or cannot be correctly inferred from the student's reasoning, explain what is unique about that step's reasoning that is missing in the student approach.

Here is an example of some inputs and the appropriate response:

Question: Let \$ABCD\$ be a .  Extend \$\overline{DA}\$ through \$A\$ to a point \$P,\$ and let \$\overline{PC}\$ meet \$\overline{AB}\$ at \$Q\$ and \$\overline{DB}\$ at \$R.\$  Given that \$PQ = 735\$ and \$QR = 112,\$ find \$RC.\$

Student attempt: 
Step 0: Define the variables as given:
- Let \(PQ = 735\),
- Let \(QR = 112\),
- Let \(RC = x\).
Step 1: Recognize that \(\triangle PAQ \sim \triangle PDC\) by the properties of similar triangles (AA similarity criterion).
Step 2: The proportionality of corresponding sides gives \(\frac{AQ}{CD} = \frac{PQ}{PC}\).
Step 3: Express \(PC\) in terms of the given lengths:
- \(PC = PQ + QR + RC = 735 + 112 + x = 847 + x\).
Step 4: Write the proportion using the similarity of the triangles:\[\frac{PQ}{PC} = \frac{735}{847 + x}\]
Step 5: Since \(\triangle PAQ \sim \triangle PDC\), the ratio of the sides must be the same:
- We need to find the ratio of \(AQ\) to \(CD\).
Step 6: Use the fact that \(\triangle PAQ \sim \triangle PDC\) to set up the proportion:\[\frac{AQ}{CD} = \frac{735}{847 + x}\]
Step 7: Since \(AQ\) and \(CD\) are corresponding sides, and by the properties of similar triangles, we know:\[\frac{AQ}{CD} = \frac{735}{847 + x}\]
Step 8: We need to find \(RC\). From the similarity, the ratio of the segments on the line \(PC\) is the same. Therefore, we can use the fact that \(PQ : QR = 735 : 112\) to find \(RC\).
Step 9: Since \(PQ : QR = 735 : 112\), and \(QR = 112\), we can set up the proportion:\[\frac{PQ}{QR} = \frac{735}{112} = \frac{847 + x - 112}{112}\]
Step 10: Simplify the equation:\[\frac{735}{112} = \frac{735}{112} = \frac{735 + x - 112}{112}\]
Step 11: This simplifies to:\[735 = 735 + x - 112\]
Step 12: Solve for \(x\):\[735 = 623 + x \implies x = 112\]
Step 13: Thus, the length of \(RC\) is:\[\boxed{112}\]

Correct Solution:
Step 0: Define the variables: let \$PQ = 735\$, \$QR = 112\$, and let \$RC\$ be the unknown we need to find.
Step 1: Recognize that triangles \$\triangle PAQ\sim \triangle PDC\$ imply proportionality, and write the proportion:\[\frac{AQ}{CD} = \frac{PQ}{PC} = \frac{735}{112 + 735 + RC} = \frac{735}{847 + RC}\]
Step 2: Similarly, recognize that triangles \$\triangle BRQ\sim DRC\$ also imply proportionality, and write the proportion:\[\frac{QR}{RC} = \frac{112}{RC} = \frac{CD - AQ}{CD} = 1 - \frac{AQ}{CD}\]
Step 3: Express \$\frac{AQ}{CD}\$ in terms of \$RC\$:\[\frac{AQ}{CD} = 1 - \frac{112}{RC} = \frac{RC - 112}{RC}\]
Step 4: Substitute the expression from Step 3 into the proportion from Step 1:\[\frac{735}{847 + RC} = \frac{RC - 112}{RC}\]
Step 5: Cross-multiply to eliminate the fractions:\[735RC = (RC + 847)(RC - 112)\]
Step 6: Expand the right-hand side:\[735RC = RC^2 - 112RC + 847RC - 847 \cdot 112\]
This simplifies to:\[735RC = RC^2 + 735RC - 847 \cdot 112\]
Step 7: Rearranging gives:\[0 = RC^2 - 847 \cdot 112\]
Step 8: Solve for \$RC\$:\[RC = \sqrt{112 \cdot 847}\]
Step 9: Calculate the value:\[RC = \sqrt{112 \cdot 847} = 308\]
Final answer: \(\boxed{308}\)

Response:
Step 1: The student corrrectly identifies the proportionality of similar triangles PAQ and PDC in the student attempt steps 1 and 2.
Step 2: The student does not identify that triangles \$\triangle BRQ\sim DRC\$ also imply proportionality, and write the proportion:\[\frac{QR}{RC} = \frac{112}{RC} = \frac{CD - AQ}{CD} = 1 - \frac{AQ}{CD}\]

Feedback: Similar to PAQ \sim PAC, similarity of triangles \$\triangle BRQ\sim DRC\$ imply proportionality, so we can write the proportion:\[\frac{QR}{RC} = \frac{112}{RC}

### Now, apply what you have learned:

Question: """ + f"""{question}

Student attempt: {student}

Correct solution: {label}

Make sure to evaluate each step of the correct solution individually against the sequence of student attempt steps to see if the student is able to reach that reasoning step at any step in their attempt.
Ignore the teacher step 0, and start from step 1. Once a solution step that doesn't match the student's reasoning is found, we can stop considering all later solution steps and output feedback based on the solution step content to help the student attempt reach that step.
It is useful in the feedback to specifically provide expressions and values directly from the correct intermediate solution step that are missing from the student's response.
Specifically in the feedback, keep all the existing content in the missing solution step but provide some additional detail about how it can be reasoned toward from the previous step, and why this step is important for future steps.
Do not mention in the feedback what the student did wrong, only mention correct expressions for solving the problem and their reasoning.

Response:
"""


def targeted_step_feedback_prompt(question, student_attempt, reference_solution, history=None):
    """
    Prompt designed to elicit more targeted and specific feedback from the teacher.
    Focuses on identifying the precise error and providing a specific hint.
    
    Args:
        question: The math problem
        student_attempt: The student's current solution attempt
        reference_solution: The correct step-by-step solution
        history: Optional history of previous attempts and feedback
        
    Returns:
        A prompt string
    """
    history_str = ""
    if history:
        history_entries = []
        for i, entry in enumerate(history):
            if "pred_steps" in entry and "prev_feedback" in entry:
                history_entries.append(f"""
Attempt {i+1}:
{entry['pred_steps']}

Feedback provided:
{entry['prev_feedback']}
""")
        if history_entries:
            history_str = "\n\nPrevious attempts and feedback:\n" + "\n".join(history_entries)
    
    return f"""Analyze this student's math solution attempt and provide specific targeted feedback:

QUESTION:
{question}

STUDENT'S CURRENT SOLUTION ATTEMPT:
{student_attempt}

REFERENCE SOLUTION:
{reference_solution}
{history_str}

Your task is to identify PRECISELY where the student's reasoning first goes wrong and provide targeted feedback.

1. Identify which specific step or concept in the reference solution is missing or incorrect in the student's work.
2. Determine the exact nature of the error (conceptual misunderstanding, procedural mistake, or calculation error).
3. Provide targeted feedback that guides the student toward the correct approach WITHOUT revealing the solution.

Your feedback should:
- Point to the specific error without giving away the answer
- Explain the underlying concept/principle the student is missing
- Provide a hint that guides them in the right direction
- Be specific enough to be actionable
- Never reveal the final answer or complete solution

Format your response as:
"Step X: [Identify the specific step number where intervention is needed]
Error: [Brief description of what went wrong]
Feedback: [Your targeted guidance to help the student correct their approach]"
"""

def concept_focused_feedback_prompt(question, student_attempt, reference_solution, history=None):
    """
    Prompt designed to focus feedback on conceptual understanding rather than calculations.
    Emphasizes core mathematical principles and problem-solving strategies.
    
    Args:
        question: The math problem
        student_attempt: The student's current solution attempt
        reference_solution: The correct step-by-step solution
        history: Optional history of previous attempts and feedback
        
    Returns:
        A prompt string
    """
    return f"""Analyze this student's math solution attempt with a focus on conceptual understanding:

QUESTION:
{question}

STUDENT'S CURRENT SOLUTION ATTEMPT:
{student_attempt}

REFERENCE SOLUTION:
{reference_solution}

Your task is to identify any conceptual misunderstandings in the student's work and provide feedback that strengthens their understanding of the core mathematical principles involved.

Focus on:
1. Key mathematical concepts needed to solve this problem
2. Fundamental principles the student may have misunderstood
3. Problem-solving strategies that would lead to a correct solution

Your feedback should:
- Emphasize conceptual understanding over calculation details
- Connect this problem to relevant mathematical principles
- Provide an insight or perspective that helps the student see the problem differently
- Guide the student to approach the problem more effectively
- Never reveal the complete solution or final answer

Format your response as:
"Key Concept: [Identify the core mathematical concept needed]
Conceptual Feedback: [Your guidance on understanding and applying this concept]
Next Step: [Specific suggestion for how to proceed]"
"""

def step_validation_feedback_prompt(question, student_attempt, reference_solution, history=None):
    """
    Prompt designed to validate each step of the student's work and identify the first error.
    Provides targeted feedback on that specific error while acknowledging correct work.
    
    Args:
        question: The math problem
        student_attempt: The student's current solution attempt
        reference_solution: The correct step-by-step solution
        history: Optional history of previous attempts and feedback
        
    Returns:
        A prompt string
    """
    return f"""Analyze this student's math solution by validating each step:

QUESTION:
{question}

STUDENT'S CURRENT SOLUTION ATTEMPT:
{student_attempt}

REFERENCE SOLUTION:
{reference_solution}

Your task is to:
1. Validate each step of the student's solution one by one
2. Identify the FIRST point where their reasoning diverges from the correct path
3. Provide specific feedback on that error while acknowledging what they did correctly

Your feedback should:
- First acknowledge what steps the student performed correctly
- Identify precisely where their solution first went wrong
- Explain why that specific step is problematic
- Suggest a better approach for that particular step
- Never reveal the complete solution or final answer

Format your response as:
"Correct Steps: [List the steps or reasoning that are correct]
First Error at Step X: [Identify where the solution first goes wrong]
Feedback: [Your targeted guidance to help correct this specific error]"
"""

def progressive_hint_prompt(question, student_attempt, reference_solution, prev_feedback_step):
    """
    Prompt designed to provide progressive hints based on the current step.
    Reveals slightly more information with each iteration but never the full solution.
    
    Args:
        question: The math problem
        student_attempt: The student's current solution attempt
        reference_solution: The correct step-by-step solution
        prev_feedback_step: The step number reached in previous feedback
        
    Returns:
        A prompt string
    """
    return f"""Provide a progressive hint for this student's math solution attempt:

QUESTION:
{question}

STUDENT'S CURRENT SOLUTION ATTEMPT:
{student_attempt}

REFERENCE SOLUTION:
{reference_solution}

The student has received feedback up through step {prev_feedback_step}. Your task is to provide a progressive hint for the next step without giving away the complete solution.

Your progressive hint should:
- Build upon what the student already knows
- Reveal slightly more information than previous hints
- Guide them toward the next logical step in the solution process
- Be specific enough to be helpful, but not so specific that it solves the problem for them
- Never reveal the final answer

Format your response as:
"Step {prev_feedback_step + 1}: [A hint for this specific step in the solution process]"
"""

