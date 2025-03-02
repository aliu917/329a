from project.utils import generate_together

def student_default_prompt(add_prompt=""):
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

class StudentLMAgent:
    def __init__(
            self,
            model: str = "Qwen/Qwen2.5-7B-Instruct-Turbo",
            generation_temp: float = 0.7,
            log_dir: str = "",
    ):
        self.model = model
        self.generation_temp = generation_temp
        self.log_dir = log_dir
        self.log_idx = 0

    def _generate(self, prompt):
        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates responses to user queries."},
            {"role": "user", "content": prompt}
        ]
        response = generate_together(messages=messages, model=self.model, temperature=self.generation_temp)
        if self.log_dir:
            with open(f"{self.log_dir}/student_prompt{str(self.log_idx)}.txt", "w") as f:
                f.write(prompt)
            with open(f"{self.log_dir}/student_response{str(self.log_idx)}.txt", "w") as f:
                f.write(response)
            self.log_idx += 1
        return response

    def generate(self, problem, prompt_func=student_default_prompt, feedback=""):
        feedback = feedback_prompt(feedback)
        return self._generate(problem + "\n" + prompt_func(feedback))