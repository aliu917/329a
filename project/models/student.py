from project.utils import generate_together, generate_openai


def student_default_prompt(problem, feedback="", history=""):
    prompt = ""
    if history:
        prompt += history + "\n"
    prompt += problem + "\n"
    if feedback:
        prompt += feedback_prompt(feedback) + "\n"
    return prompt + "Think step by step and then provide the final answer boxed in the format: '\\boxed{final answer}'"


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
            max_log: int = 10
    ):
        self.model = model
        self.generation_temp = generation_temp
        self.log_dir = log_dir
        self.log_idx = 0
        self.max_log = max_log

    def _generate(self, prompt):
        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates responses to user queries."},
            {"role": "user", "content": prompt}
        ]
        if self.log_dir and self.log_idx < self.max_log:
            with open(f"{self.log_dir}/student_gen{str(self.log_idx)}.txt", "w") as f:
                f.write(prompt)
        if "gpt" in self.model:
            response = generate_openai(messages=messages, model=self.model, temperature=self.generation_temp)
        else:
            response = generate_together(messages=messages, model=self.model, temperature=self.generation_temp)
        if self.log_dir and self.log_idx < self.max_log:
            with open(f"{self.log_dir}/student_gen{str(self.log_idx)}.txt", "a") as f:
                f.write("-"*50 + "\n" + response)
            self.log_idx += 1
        return response

    def generate(self, problem, prompt_func=student_default_prompt, feedback="", history=""):
        return self._generate(prompt_func(problem, feedback, history))