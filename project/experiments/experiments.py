import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from project.models import StudentLMAgent, TeacherLMAgent
from . import prompts
from project.utils import extract_text_within_box
from project.models import verifiers
verifier = verifiers.MATH500Verifier()


def report_and_save_results(results, results_path):
    print("Accuracy:", np.mean([r["correct"] for r in results]))
    if results_path:
        with open(results_path, "w") as f:
            json.dump(results, f)

def run_single(x, y, student, teacher, input_feedback, prompt_func=None):
    # Student generation and evaluation
    response = student.generate(x, feedback=input_feedback)
    correct, pred, answer, output_feedback = teacher.evaluate(
        x, response, y, prompt_func=prompt_func
    )
    result = {
        "input_feedback": input_feedback,
        "question": x,
        "answer": answer,
        "pred": pred,
        "pred_steps": response,
        "answer_steps": y,
        "correct": correct,
        "output_feedback": output_feedback,
    }
    return result

class Experiment:
    def __init__(self, dataset, log_dir=None, num_examples=10, seed=2809):
        self.dataset = dataset
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True)
        self.student = StudentLMAgent(log_dir=log_dir)
        self.teacher = TeacherLMAgent(log_dir=log_dir)
        self.num_examples = num_examples
        self.seed = seed

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self.results_path = f"{log_dir}/results.json"
        else:
            self.results_path = None
    
    def run(self):
        torch.manual_seed(self.seed)
        results = self.get_results()
        report_and_save_results(results, self.results_path)

class Base(Experiment):
    def get_results(self):
        results = []
        for i, (data, labels) in tqdm(zip(range(self.num_examples), self.dataloader)):
            x = data["problem"][0]
            y = labels[0]
            result = run_single(x, y, self.student, self.teacher, "")
            results.append(result)
        return results

class SingleRound(Experiment):
    def get_results(self):
        results = []
        for i, (data, labels) in tqdm(zip(range(self.num_examples), self.dataloader)):
            x = data["problem"][0]
            y = labels[0]
            result = run_single(x, y, self.student, self.teacher, "")
            feedback = result["output_feedback"]
            result = run_single(x, y, self.student, self.teacher, feedback)
            results.append(result)
        return results

# TODO: Finish porting this!
class IterativeRefine(Experiment):
    def __init__(self, *args, n_rounds=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_rounds = n_rounds

    def get_results(self, dataset, student, teacher):
        results = []
        for i, (data, labels) in tqdm(zip(range(self.num_examples), dataset)):
            x = data["problem"][0]
            y = labels[0]
            prev_feedback = ""
            for refine_round in range(self.n_rounds):
                response = student.generate(data, feedback=prev_feedback)
                correct, pred, answer, feedback = teacher.evaluate(
                    x,
                    response,
                    y,
                    history=results,
                    prompt_func=prompts.teacher_iteration_prompt,
                )
                result = {
                    "question": x,
                    "answer": answer,
                    "pred": pred,
                    "pred_steps": response,
                    "answer_steps": y,
                    "correct": correct,
                    "feedback": feedback,
                    "prev_feedback": prev_feedback,
                    "round": refine_round + 1,
                    "feedback_in_answer": str(answer) in prev_feedback,
                }
                results.append(result)
                prev_feedback = feedback
            return results

class MultipleIteration(Experiment):
    """
    In this experiment, we train the student and teacher over multiple iterations, keeping the
    entire history of attempts and feedbacks in the prompt for learning/refinement.
    """
    def __init__(self, *args, iters=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.iters = iters

    def get_results(self):
        student_multi_iter_examples = ""
        teacher_multi_iter_examples = ""
        round_results = []
        for iter in range(self.iters):
            torch.manual_seed(self.seed)
            round_results = []
            for i, (data, labels) in tqdm(zip(range(self.num_examples), self.dataloader)):
                log_prompts = (iter + i < 3)
                x = data["problem"][0]
                y = labels[0]

                attempt = self.student._generate(prompts.student_prompt_with_examples(x, student_multi_iter_examples), log_prompts)
                feedback = self.teacher._generate(prompts.teacher_prompt_with_examples(x, attempt, y, teacher_multi_iter_examples), log_prompts)

                student_multi_iter_examples += prompts.build_next_student_example(x, attempt, feedback)
                teacher_multi_iter_examples += prompts.build_next_teacher_example(x, feedback, attempt)

                result = {
                    "question": x,
                    "answer": extract_text_within_box(y),
                    "pred": extract_text_within_box(attempt),
                    "pred_steps": attempt,
                    "answer_steps": y,
                    "correct": verifier(attempt, y),
                    "output_feedback": feedback,
                }
                round_results.append(result)
            path_name, file_ext = self.results_path.split(".")
            report_and_save_results(round_results, f"{path_name}_{iter}.{file_ext}")
        report_and_save_results(round_results, self.results_path)

        return round_results[-1]
