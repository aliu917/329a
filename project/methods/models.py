from . import prompts
from . import verifiers
from project.utils import generate_openai, generate_together, extract_text_within_box

verifier = verifiers.MATH500Verifier()

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

    def generate(self, problem, prompt_func=prompts.default_prompt, feedback=""):
        feedback = prompts.feedback_prompt(feedback)
        return self._generate(problem + "\n" + prompt_func(feedback))

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

    def generate(self, question, student_steps, answer_steps, history=None, prompt_func = prompts.teacher_default_prompt):
        # TODO: remove predicting correct, we don't use it now
        correct = False
        feedback = ""
        while not correct and not feedback:
            prompt = prompt_func(question, student_steps, answer_steps, history)
            result = self._generate(prompt)
            correct, feedback = result.split("Feedback:")
            correct = "yes" in correct.lower()
            feedback = feedback.strip()
            # TODO: confirm that feedback does not give away the solution, else empty.

        return correct, feedback
    
    def evaluate(self, x, pred, label, history=None, prompt_func=None):
        is_correct = verifier(pred, label)
        pred_answer = extract_text_within_box(pred)
        label_answer = extract_text_within_box(label)
        if history:
            current_round = {
                "round": history[-1]["round"] + 1,
                "prev_feedback": history[-1]["feedback"],
                "pred_steps": pred,
                "correct": is_correct,
            }
            history = history + [current_round]
            _, feedback = self.generate(x, pred, label, history, prompt_func=prompt_func)
        else:
            _, feedback = self.generate(x, pred, label, prompt_func=prompt_func)
        return is_correct, pred_answer, label_answer, feedback
    
    def evaluate_old(self, x, pred, label):
        #TODO: better evaluation method (need to fix whitespace)
        pred_answer = extract_text_within_box(pred)
        label_answer = extract_text_within_box(label)
        teacher_eval_correct, feedback = self.generate(x, pred, label)
        return pred_answer == label_answer or teacher_eval_correct, pred_answer, label_answer, feedback
