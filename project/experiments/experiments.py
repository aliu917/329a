import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from project.models import StudentLMAgent, TeacherLMAgent, verifiers
from . import prompts
from ..utils import extract_text_within_box, extract_steps
verifier = verifiers.MATH500Verifier()
import re

def run_single(question_data, y, student, teacher, input_feedback, prompt_func=None):
    # Student generation and evaluation
    x = question_data["problem"][0]
    response = student.generate(x, feedback=input_feedback)
    correct, pred, answer, output_feedback = teacher.evaluate(
        question_data, response, y, prompt_func=prompt_func
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
    def __init__(self, dataset_cls, student_model, teacher_model, log_dir=None, num_examples=10, seed=2809):
        self.dataset = dataset_cls(mode="train")
        torch.manual_seed(seed)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True)
        self.student = StudentLMAgent(model=student_model, log_dir=log_dir)
        self.teacher = TeacherLMAgent(model=teacher_model, log_dir=log_dir)
        self.num_examples = num_examples
        self.seed = seed
        self.log_dir = log_dir

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self.results_path = f"{log_dir}/results.json"
        else:
            self.results_path = None
    
    def run(self):
        results = self.get_results()
        print("Accuracy:", np.mean([r["correct"] for r in results]))
        if self.results_path:
            with open(self.results_path, "w") as f:
                json.dump(results, f)

class Base(Experiment):
    def get_results(self):
        results = []
        for i, (data, labels) in tqdm(zip(range(self.num_examples), self.dataloader)):
            y = labels[0]
            result = run_single(data, y, self.student, self.teacher, "")
            results.append(result)
        return results

class BestOfN(Experiment):
    def __init__(self, *args, n_rounds=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_rounds = n_rounds
    
    def get_results(self):
        results = []
        for i, (data, labels) in tqdm(zip(range(self.num_examples), self.dataloader)):
            for _ in range(self.n_rounds):
                y = labels[0]
                result = run_single(data, y, self.student, self.teacher, "")
                if result["correct"]:
                    break
            results.append(result)
        return results

class SingleRound(Experiment):
    def get_results(self):
        results = []
        for i, (data, labels) in tqdm(zip(range(self.num_examples), self.dataloader)):
            y = labels[0]
            result = run_single(data, y, self.student, self.teacher, "")
            feedback = result["output_feedback"]
            result = run_single(data, y, self.student, self.teacher, feedback)
            results.append(result)
        return results

class IterativeRefine(Experiment):
    def __init__(self, *args, n_rounds=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_rounds = n_rounds

    def get_results(self):
        results = []
        for i, (data, labels) in tqdm(zip(range(self.num_examples), self.dataloader)):
            x = data["problem"][0]
            y = labels[0]
            prev_feedback = ""
            history = []
            for refine_round in range(self.n_rounds):
                response = self.student.generate(x, feedback=prev_feedback)
                correct, pred, answer, feedback = self.teacher.evaluate(
                    data,
                    response,
                    y,
                    history=history,
                    prompt_func=prompts.teacher_iteration_prompt if history else None,
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
                    "answer_in_feedback": str(answer) in prev_feedback,
                }
                history.append(result)
                prev_feedback = feedback
                if correct:
                    break
            results.append(result)
        return results


class IterativeStepBasedRefine(Experiment):
    def __init__(self, *args, trials=1, limit=0.8, **kwargs):
        super().__init__(*args, **kwargs)
        self.limit = limit
        self.trials = trials

    def label_steps_generator(self, q, label_cot):
        label_approaches = label_cot.split("=== Solution")
        clean_label_cots = [label.split("===")[-1].strip() for label in label_approaches]
        for label_cot in clean_label_cots:
            yield self.teacher._generate(prompts.cot_to_steps_prompt(q, label_cot))
    
    def step_feedback(self, x, response, y_steps_list, prev_responses, prev_feedbacks, prev_feedback_step, is_correct):
        return "\n".join(y_steps_list[:prev_feedback_step + 1]), prev_feedback_step + 1

    def get_results(self):
        base_results = []
        results = []
        iter_results_list = []
        for i, (data, labels) in tqdm(zip(range(self.num_examples), self.dataloader)):
            x = data["problem"][0]
            all_y = labels[0]
            is_correct = False
            first = True
            base_result = {"question": x}
            result = base_result
            numerical_answer = str(data["answer"].item())

            iter_results = []
            # Iterator is only for AIME case of multiple label approaches
            for label_round, y_steps in enumerate(self.label_steps_generator(x, all_y)):
                answer = extract_text_within_box(all_y)
                label_steps_list, extract_answer = extract_steps(y_steps, r"[Aa]nswer")
                if not answer:
                    answer = extract_text_within_box(extract_answer)
                prev_responses = []
                prev_feedbacks = []
                prev_feedback = ""
                prev_feedback_step = 1

                for refine_round in range(len(label_steps_list)):
                    result = base_result
                    for _ in range(self.trials):
                        response = self.student.generate(x, prompt_func=prompts.student_step_func, feedback=prev_feedback)
                        # Compare to numerical answer for AMC questions
                        is_correct = verifier(response, all_y) or verifier(response, numerical_answer)
                        if is_correct:
                            break
                    feedback, prev_feedback_step = self.step_feedback(x, response, label_steps_list, prev_responses, prev_feedbacks, prev_feedback_step, is_correct)

                    pred = extract_text_within_box(response)
                    result.update({
                        "answer": answer,
                        "pred": pred,
                        "label_round": label_round + 1,
                        "round": refine_round + 1,
                        "correct": is_correct,
                        "prev_feedback": prev_feedback,
                        "feedback": feedback,
                        "pred_steps": response,
                        "answer_steps": y_steps,
                        "answer_in_feedback": str(answer) in prev_feedback,
                    })
                    prev_feedback = feedback
                    prev_feedbacks.append(prev_feedback)
                    prev_responses.append(response)

                    print(f"Round {refine_round + 1}: pred {pred} answer {answer} feedback step {prev_feedback_step}/{str(len(label_steps_list)-1)}")

                    if first:
                        first = False
                        base_results.append(result.copy())

                    iter_results.append(result.copy())
                    if self.results_path:
                        with open(f"{self.log_dir}/refine_results_{i}.json", "w") as f:
                            json.dump(iter_results, f)

                    # Early return because correct
                    if is_correct:
                        print(f"Early stopped from correct feedback at iteration {refine_round + 1}/{str(len(label_steps_list)-1)}!")
                        # print("Feedback: ", prev_feedback)
                        break

                    # Early return because cannot give more feedback without giving away answer
                    # if prev_feedback_step >= len(label_steps_list)*self.limit - 1 and str(answer) in feedback:
                    if prev_feedback_step >= len(label_steps_list)*self.limit - 1:
                        break

                if is_correct:
                    break

            if not is_correct:
                print("Correct answer never reached.")

            results.append(result)
            iter_results_list.append(iter_results)
            if self.results_path:
                with open(f"{self.log_dir}/iterative_results.json", "w") as f:
                    json.dump(iter_results_list, f)
                with open(f"{self.log_dir}/base_results.json", "w") as f:
                    json.dump(base_results, f)
        return results

class TeacherIterativeStepBasedRefine(IterativeStepBasedRefine):
    """
    Iterative step refinement but we use teacher to provide specific step feedback.
    Teacher goes through solution and checks if each step is addressed by the student. If there is a mistake,
    then the teacher identifies the incorrect step and provides a hint of what the appropriate step should be.
    """
    def step_feedback(self, x, response, y_steps_list, prev_responses, prev_feedbacks, prev_feedback_step, is_correct):
        response_steps_list, _ = extract_steps(response)

        if is_correct:
            return "Student was correct", 1
        if len(prev_feedbacks) + 1 == len(y_steps_list):
            return "Last feedback round unneeded", 1

        # Find first teacher step concept missing from the student
        _, response = self.teacher.generate(x, "\n".join(response_steps_list), "\n".join(y_steps_list), prompt_func=prompts.teacher_step_prompt)
        feedback = ""
        step = 1
        try:
            steps_list, feedback = extract_steps(response, r"[Ff]eedback")
            try:
                step = re.findall(r"Step (\d+):", steps_list[-1])[0]
            except Exception as e:
                print("Could not extract step number")
                step = len(steps_list)
        except Exception as e:
            print(f"Count not extract teacher feedback steps. See output: teacher_gen{str(self.teacher.log_idx-1)}.txt")
            print("Error: ", e)
        return feedback, int(step)

class TeacherSearch(Experiment):
    def __init__(self, dataset_cls, *args, limit=0.8, **kwargs):
        super().__init__(dataset_cls, *args, **kwargs)
        val_dataset = dataset_cls(mode="val")
        self.val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
        self.limit = limit
        # TODO: remove this!
        self.limit = 1.0
        print("LIMIT 1.0")

    def label_steps_generator(self, q, label_cot):
        label_approaches = label_cot.split("=== Solution")
        clean_label_cots = [label.split("===")[-1].strip() for label in label_approaches]
        for label_cot in clean_label_cots:
            yield self.teacher._generate(prompts.cot_to_steps_prompt(q, label_cot))
    
    def step_feedback(self, x, response, y_steps_list, prev_responses, prev_feedbacks, prev_feedback_step, is_correct):
        return "\n".join(y_steps_list[:prev_feedback_step + 1]), prev_feedback_step + 1

    def get_results(self):
        base_results = []
        results = []
        iter_results_list = []
        context_examples = []
        prev_val_correct = 0
        for i, (data, labels) in tqdm(zip(range(self.num_examples), self.dataloader)):
            x = data["problem"][0]
            all_y = labels[0]
            is_correct = False
            first = True
            base_result = {"question": x}
            result = base_result
            numerical_answer = str(data["answer"].item())

            iter_results = []
            # Iterator is only for AIME case of multiple label approaches
            for label_round, y_steps in enumerate(self.label_steps_generator(x, all_y)):
                answer = extract_text_within_box(all_y)
                label_steps_list, extract_answer = extract_steps(y_steps, r"[Aa]nswer")
                if not answer:
                    answer = extract_text_within_box(extract_answer)
                prev_responses = []
                prev_feedbacks = []
                prev_feedback = ""
                prev_feedback_step = 1

                for refine_round in range(len(label_steps_list)):
                    result = base_result
                    response = self.student.generate(x, prompt_func=prompts.student_step_func, feedback=prev_feedback)
                    # Compare to numerical answer for AMC questions
                    is_correct = verifier(response, all_y) or verifier(response, numerical_answer)
                    feedback, prev_feedback_step = self.step_feedback(x, response, label_steps_list, prev_responses, prev_feedbacks, prev_feedback_step, is_correct)

                    pred = extract_text_within_box(response)
                    result.update({
                        "answer": answer,
                        "numerical_answer": numerical_answer,
                        "pred": pred,
                        "label_round": label_round + 1,
                        "round": refine_round + 1,
                        "correct": is_correct,
                        "prev_feedback": prev_feedback,
                        "feedback": feedback,
                        "pred_steps": response,
                        "answer_steps": y_steps,
                        "answer_in_feedback": str(numerical_answer) in prev_feedback,
                    })
                    prev_feedback = feedback
                    prev_feedbacks.append(prev_feedback)
                    prev_responses.append(response)

                    print(f"Round {refine_round + 1}: pred {pred} answer {answer} feedback step {prev_feedback_step}/{str(len(label_steps_list)-1)}")

                    if first:
                        first = False
                        base_results.append(result.copy())

                    iter_results.append(result.copy())
                    if self.results_path:
                        with open(f"{self.log_dir}/refine_results_{i}.json", "w") as f:
                            json.dump(iter_results, f)

                    # Early return because correct
                    if is_correct:
                        print(f"Early stopped from correct feedback at iteration {refine_round + 1}/{str(len(label_steps_list)-1)}!")
                        # print("Feedback: ", prev_feedback)
                        break

                    # Early return because cannot give more feedback without giving away answer
                    # if prev_feedback_step >= len(label_steps_list)*self.limit - 1 and str(answer) in feedback:
                    if prev_feedback_step >= len(label_steps_list)*self.limit - 1:
                        break

                if is_correct:
                    break

            if not is_correct:
                print("Correct answer never reached.")
            
            # Optimize feedback
            if is_correct:
                for _ in range(10):
                    _, feedback = self.teacher.generate(None, None, None, iter_results, prompt_func=prompts.teacher_search_prompt)
                    optimized_response = self.student.generate(x, prompt_func=prompts.student_step_func, feedback=feedback)
                    optimized_correct = verifier(optimized_response, all_y) or verifier(optimized_response, numerical_answer)
                    if optimized_correct:
                        break
                context_result = result.copy()
                if optimized_correct:
                    context_result["prev_feedback"] = feedback
                    context_result["pred_steps"] = optimized_response
                    context_result["answer_in_feedback"] = str(numerical_answer) in feedback
            
                val_correct = 0
                new_context_examples = context_examples + [context_result]
                for _, (val_data, val_labels) in zip(range(20), self.val_dataloader):
                    val_x = val_data["problem"][0]
                    val_all_y = val_labels[0]
                    val_numerical_answer = str(val_data["answer"].item())
                    val_context = prompts.get_eval_student_context(new_context_examples, ["prev_feedback"], self.num_examples)
                    val_response = self.student.generate(val_x, history=val_context)
                    val_correct += verifier(val_response, val_all_y) or verifier(val_response, val_numerical_answer)
                print(prev_val_correct, val_correct)
                if val_correct >= prev_val_correct:
                    context_examples = new_context_examples
                    prev_val_correct = val_correct
                
            results.append(result)
            iter_results_list.append(iter_results)
            if self.results_path:
                with open(f"{self.log_dir}/iterative_results.json", "w") as f:
                    json.dump(iter_results_list, f)
                with open(f"{self.log_dir}/optimized_results.json", "w") as f:
                    json.dump(context_examples, f)
                with open(f"{self.log_dir}/base_results.json", "w") as f:
                    json.dump(base_results, f)
        return results