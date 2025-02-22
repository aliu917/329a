from utils import *
import prompts

class StudentLMAgent:
    def __init__(
            self,
            model: str = "Qwen/Qwen2.5-7B-Instruct-Turbo",
            generation_temp: float = 0.7,
    ):
        self.model = model
        self.generation_temp = generation_temp

    def _generate(self, prompt):
        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates responses to user queries."},
            {"role": "user", "content": prompt}
        ]
        response = generate_together(messages=messages, model=self.model, temperature=self.generation_temp)
        return response

    def generate(self, problem, history=[]):
        #TODO: incorporate feedback in history
        return self._generate(problem + "\n" + prompts.default_prompt())

class TeacherLMAgent():
    def __init__(
            self,
            model: str = "gpt-4o-mini",
            generation_temp: float = 0.7,
    ):
        self.model = model
        self.generation_temp = generation_temp

    def _generate(self, prompt):
        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates responses to user queries."},
            {"role": "user", "content": prompt}
        ]
        response = generate_openai(messages=messages, model=self.model, temperature=self.generation_temp)
        return response

    def generate(self, question, student_steps, answer_steps):
        correct = False
        feedback = ""
        while not correct and not feedback:
            result = self._generate(prompts.teacher_default_prompt(question, student_steps, answer_steps))
            correct, feedback = result.split("Feedback:")
            correct = "yes" in correct.lower()
            feedback = feedback.strip()
            # TODO: confirm that feedback does not give away the solution, else empty.

        return correct, feedback
