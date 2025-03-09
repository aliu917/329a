import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from project.models import StudentLMAgent, TeacherLMAgent
from . import prompts

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
        print("Accuracy:", np.mean([r["correct"] for r in results if r["final"]]))
        if self.results_path:
            with open(self.results_path, "w") as f:
                json.dump(results, f)

class Base(Experiment):
    def get_results(self):
        results = []
        for i, (data, labels) in tqdm(zip(range(self.num_examples), self.dataloader)):
            x = data["problem"][0]
            y = labels[0]
            result = run_single(x, y, self.student, self.teacher, "")
            result["final"] = True
            results.append(result)
        return results

class BestOfN(Experiment):
    def __init__(self, *args, n_rounds=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_rounds = n_rounds
    
    def get_results(self):
        results = []
        for i, (data, labels) in tqdm(zip(range(self.num_examples), self.dataloader)):
            for _ in range(self.n_rounds):
                x = data["problem"][0]
                y = labels[0]
                result = run_single(x, y, self.student, self.teacher, "")
                if result["correct"]:
                    break
            result["final"] = True
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
            result["final"] = True
            results.append(result)
        return results

class IterativeRefine(Experiment):
    def __init__(self, *args, n_rounds=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_rounds = n_rounds

    def get_results(self):
        results = []
        for i, (data, labels) in tqdm(zip(range(self.num_examples), self.dataloader)):
            x = data["problem"][0]
            y = labels[0]
            prev_feedback = ""
            for refine_round in range(self.n_rounds):
                response = self.student.generate(x, feedback=prev_feedback)
                correct, pred, answer, feedback = self.teacher.evaluate(
                    x,
                    response,
                    y,
                    history=results,
                    prompt_func=prompts.teacher_iteration_prompt if results else None,
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
                    "final": False,
                }
                results.append(result)
                prev_feedback = feedback
                if correct:
                    break
            results[-1]["final"] = True
        return results