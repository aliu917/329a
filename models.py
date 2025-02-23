from utils import *
import prompts

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

    def generate(self, problem, prompt_func=prompts.default_prompt, history=""):
        return self._generate(problem + "\n" + prompt_func(history))

class TeacherLMAgent():
    def __init__(
            self,
            model: str = "gpt-4o-mini",
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
        response = generate_openai(messages=messages, model=self.model, temperature=self.generation_temp)
        if self.log_dir:
            with open(f"{self.log_dir}/teacher_prompt{str(self.log_idx)}.txt", "w") as f:
                f.write(prompt)
            with open(f"{self.log_dir}/teacher_response{str(self.log_idx)}.txt", "w") as f:
                f.write(response)
            self.log_idx += 1
        return response

    def generate(self, question, student_steps, answer_steps, prompt_func = prompts.teacher_default_prompt):
        correct = False
        feedback = ""
        while not correct and not feedback:
            result = self._generate(prompt_func(question, student_steps, answer_steps))
            correct, feedback = result.split("Feedback:")
            correct = "yes" in correct.lower()
            feedback = feedback.strip()
            # TODO: confirm that feedback does not give away the solution, else empty.

        return correct, feedback
