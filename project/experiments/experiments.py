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
    def __init__(self, *args, trials=3, limit=0.8, **kwargs):
        super().__init__(*args, **kwargs)
        self.limit = limit
        self.trials = trials

    def label_steps_generator(self, q, label_cot):
        label_approaches = label_cot.split("=== Solution")
        clean_label_cots = [label.split("===")[-1].strip() for label in label_approaches]
        for label_cot in clean_label_cots:
            yield self.teacher._generate(prompts.cot_to_steps_prompt(q, label_cot), model="gpt-4o-mini")

    def step_feedback(self, x, response, y_steps_list, prev_responses, prev_feedbacks, prev_feedback_step, is_correct):
        return "\n".join(y_steps_list[:prev_feedback_step + 1]), prev_feedback_step + 1, ""

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
            prev_pred_steps = ""
            prev_teacher_response = ""

            iter_results = []
            # Iterator is only for AIME case of multiple label approaches
            for label_round, y_steps in enumerate(self.label_steps_generator(x, all_y)):
                answer = extract_text_within_box(all_y)
                label_steps_list, extract_answer = extract_steps(y_steps, r"[Aa]nswer")
                if not answer:
                    answer = extract_text_within_box(extract_answer)
                    if not answer:
                        answer = extract_answer.strip()
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
                    feedback, prev_feedback_step, teacher_response = self.step_feedback(x, response, label_steps_list, prev_responses, prev_feedbacks, prev_feedback_step, is_correct)
                    answer_in_feedback = str(answer) in prev_feedback
                    if "R1" in self.teacher.model:
                        feedback = feedback.replace(answer, "") # Answer occurs in feedback too often, need to manually remove

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
                        "answer_in_feedback": answer_in_feedback,
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

                        # Teacher logging for history-based teacher prompt. See TeacherIterativeStepBasedRefineWithHistory
                        if prev_pred_steps and prev_teacher_response:
                            teacher_result = result.copy()
                            teacher_result.update({
                                "teacher_response": prev_teacher_response,
                                "prev_pred_steps" : prev_pred_steps,
                            })
                            self.teacher.save_successful_result(teacher_result)

                        print(f"Early stopped from correct feedback at iteration {refine_round + 1}/{str(len(label_steps_list)-1)}!")
                        # print("Feedback: ", prev_feedback)
                        break

                    prev_pred_steps = response
                    prev_teacher_response = teacher_response

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
            return "Student was correct", 1, ""
        if len(prev_feedbacks) + 1 == len(y_steps_list):
            return "Last feedback round unneeded", 1, ""

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
        return feedback, int(step), response


class TeacherIterativeStepBasedRefineWithHistory(IterativeStepBasedRefine):
    """
    Adds two additional changes on top of teacher iterative step:
        1. provides the entire history of student and teacher attempts for the same problem to help the teacher learn/refine
        2. adds all previously successful teacher results in the example prompt
    """
    def step_feedback(self, x, response, y_steps_list, prev_responses, prev_feedbacks, prev_feedback_step, is_correct):
        response_steps_list, _ = extract_steps(response)

        if is_correct:
            return "Student was correct", 1, ""
        if len(prev_feedbacks) + 1 == len(y_steps_list):
            return "Last feedback round unneeded", 1, ""

        # Find first solution step concept missing from the student
        if "R1" in self.teacher.model:
            # Use gpt-4o-mini for first feedback bc reasoning does not do well
            if not prev_feedbacks:
                _, response = self.teacher.generate(x, "\n".join(response_steps_list), "\n".join(y_steps_list),
                                                    history=(
                                                    prev_feedbacks, prev_responses, self.teacher.successful_results),
                                                    prompt_func=prompts.refine_teacher_step_prompt,
                                                    model="gpt-4o-mini")
            else:
                _, response = self.teacher.generate(x, "\n".join(response_steps_list), "\n".join(y_steps_list),
                                                    history=(
                                                        prev_feedbacks, prev_responses,
                                                        self.teacher.successful_results),
                                                    prompt_func=prompts.refine_r1_teacher_step_prompt)
        else:
            _, response = self.teacher.generate(x, "\n".join(response_steps_list), "\n".join(y_steps_list), history=(prev_feedbacks, prev_responses, self.teacher.successful_results), prompt_func=prompts.refine_teacher_step_prompt)
        feedback = ""
        step = 1
        try:
            steps_list, feedback = extract_steps(response, r"[Ff]eedback")
            try:
                step = re.findall(r"Step (\d+):", steps_list[-1])[0]
            except Exception as e:
                print("Could not extract step number")
                step = 1
        except Exception as e:
            print(f"Count not extract teacher feedback steps. See output: teacher_gen{str(self.teacher.log_idx-1)}.txt")
            print("Error: ", e)

        final_step = min(int(step), 0.6*len(y_steps_list)) # if step > 60%, most likely a mistake
        return feedback, final_step, response


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



class EnhancedTeacherRefine(TeacherIterativeStepBasedRefine):
    """
    Enhanced teacher that focuses on specifically identifying error points in student reasoning
    and provides targeted feedback to correct misconceptions without revealing the solution.
    
    Key improvements:
    1. More precise error identification
    2. Error categorization (conceptual, procedural, or calculation)
    3. Targeted hint generation
    4. Progress tracking across multiple attempts
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_categories = ["conceptual", "procedural", "calculation"]
        
    def step_feedback(self, x, response, y_steps_list, prev_responses, prev_feedbacks, prev_feedback_step, is_correct):
        response_steps_list, _ = extract_steps(response)
        
        if is_correct:
            return "Student was correct", 1
        if len(prev_feedbacks) + 1 == len(y_steps_list):
            return "Last feedback round unneeded", 1
            
        # Current student attempt history to track progress
        attempt_history = ""
        if prev_responses:
            attempt_history = "\n\nPrevious attempts and feedback:\n" + "\n".join([
                f"Attempt {i+1}:\n{resp}\n\nFeedback provided:\n{fb}"
                for i, (resp, fb) in enumerate(zip(prev_responses, prev_feedbacks))
            ])
        
        y_steps_str = "\n".join(y_steps_list)
        
        # Enhanced prompt for targeted feedback
        enhanced_prompt = f"""Analyze this student's math solution attempt:

QUESTION:
{x}

STUDENT'S CURRENT SOLUTION ATTEMPT:
{response}

REFERENCE SOLUTION STEPS:
{y_steps_str}
{attempt_history}

Your task is to identify PRECISELY where the student's reasoning first deviates from the correct solution path.

1. First, identify which step of the REFERENCE SOLUTION is first missing or incorrect in the student's work.
2. Determine the specific error type:
   - Conceptual error: Fundamental misunderstanding of a concept
   - Procedural error: Correct concept but wrong procedure/approach
   - Calculation error: Arithmetic mistake or algebraic manipulation error

3. Provide targeted feedback that:
   - Points to the specific error without revealing the complete correct step
   - Explains the underlying concept/principle the student is missing
   - Guides them toward the correct approach with a hint
   - NEVER reveals the final answer

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
Step X: [Step number where error first occurs]
Error Type: [conceptual/procedural/calculation]
Feedback: [Your detailed guidance on fixing the error]
"""
        
        # Generate enhanced feedback - MODIFIED TO MATCH YOUR API
        _, response = self.teacher.generate(
            question=x,
            student_steps=response,
            answer_steps=enhanced_prompt,
            history=None,
            prompt_func=None
        )
        
        # Extract the step number and feedback
        feedback = response
        step = prev_feedback_step
        
        try:
            # Extract step number
            step_match = re.search(r"Step (\d+):", response)
            if step_match:
                step = int(step_match.group(1))
            
            # If unable to extract step properly, use the next step from previous feedback
            if not step_match:
                step = prev_feedback_step + 1
                
        except Exception as e:
            print(f"Error extracting step information: {e}")
            print(f"Using fallback step: {prev_feedback_step + 1}")
            step = prev_feedback_step + 1
        
        return feedback, step, response


class SpecializedTeacherRefine(TeacherIterativeStepBasedRefine):
    """
    A specialized teacher refinement experiment that implements both targeted feedback
    and self-critique capabilities.
    
    This experiment enhances the teacher feedback by:
    1. Precisely identifying the specific point where student reasoning deviates
    2. Categorizing the error type (conceptual, procedural, calculation)
    3. Providing targeted feedback specifically for that error
    4. Learning from previous feedback attempts with self-critique
    5. Progressively revealing more guidance without giving away the answer
    
    Can be used as a drop-in replacement for TeacherIterativeStepBasedRefine.
    """
    
    def __init__(self, *args, use_self_critique=True, **kwargs):
        """
        Initialize the specialized teacher experiment.
        
        Args:
            *args: Arguments to pass to TeacherIterativeStepBasedRefine
            use_self_critique: Whether to use self-critique (for rounds after the first)
            **kwargs: Keyword arguments to pass to TeacherIterativeStepBasedRefine
        """
        super().__init__(*args, **kwargs)
        self.use_self_critique = use_self_critique
    
    def step_feedback(self, x, response, y_steps_list, prev_responses, prev_feedbacks, prev_feedback_step, is_correct):
        """
        Generate specialized step-based feedback using enhanced teacher strategies.
        
        Args:
            x: The question
            response: The student's current response
            y_steps_list: List of correct solution steps
            prev_responses: List of previous student responses
            prev_feedbacks: List of previous feedback messages
            prev_feedback_step: The step number reached in previous feedback
            is_correct: Whether the current response is correct
            
        Returns:
            Tuple of (feedback, step_number)
        """
        # No need for feedback if the student is correct
        if is_correct:
            return "Student was correct", 1
            
        # Check if we've reached the limit of feedback steps
        if len(prev_feedbacks) + 1 == len(y_steps_list):
            return "Last feedback round unneeded", 1
        
        # First attempt - use targeted feedback approach
        if not prev_responses:
            return self._generate_targeted_feedback(x, response, y_steps_list, prev_feedback_step)
        
        # Subsequent attempts - incorporate self-critique if enabled
        if self.use_self_critique and prev_responses and prev_feedbacks:
            return self._generate_self_critique_feedback(x, response, y_steps_list, prev_responses, prev_feedbacks, prev_feedback_step)
        else:
            return self._generate_targeted_feedback(x, response, y_steps_list, prev_feedback_step)
    
    def _generate_targeted_feedback(self, x, response, y_steps_list, prev_feedback_step):
        """
        Generate targeted feedback focused on specific error identification.
        
        Args:
            x: The question
            response: The student's current response
            y_steps_list: List of correct solution steps
            prev_feedback_step: The step number reached in previous feedback
            
        Returns:
            Tuple of (feedback, step_number)
        """
        # Use a custom prompt function for targeted feedback
        def targeted_feedback_prompt(question, student, label, history=None):
            return f"""Analyze this student's math solution attempt with precision:

QUESTION:
{question}

STUDENT'S SOLUTION ATTEMPT:
{student}

CORRECT SOLUTION STEPS:
{label}

Your task is to identify PRECISELY where the student's reasoning first deviates from the correct solution path.

1. First, identify which step of the correct solution is first missing or incorrect in the student's work.
2. Determine the specific error type:
   - Conceptual error: Fundamental misunderstanding of a concept
   - Procedural error: Correct concept but wrong procedure/approach
   - Calculation error: Arithmetic mistake or algebraic manipulation error

3. Provide targeted feedback that:
   - Points to the specific error without revealing the complete correct step
   - Explains the underlying concept/principle the student is missing
   - Guides them toward the correct approach with a hint
   - NEVER reveals the final answer

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
Step X: [Step number where error first occurs]
Error Type: [conceptual/procedural/calculation]
Feedback: [Your detailed guidance on fixing the error]
"""
        
        # Use the teacher's generate method with correct signature
        # Note: We're passing y_steps_list as a string here to match the interface
        solution_steps = "\n".join(y_steps_list)
        _, feedback = self.teacher.generate(
            question=x,
            student_steps=response,
            answer_steps=solution_steps,
            prompt_func=targeted_feedback_prompt
        )
        
        # Extract the step number from the feedback
        step = prev_feedback_step
        try:
            # Extract step number
            step_match = re.search(r"Step (\d+):", feedback)
            if step_match:
                step = int(step_match.group(1))
            else:
                # If unable to extract step properly, use the next step from previous feedback
                step = prev_feedback_step + 1
                
        except Exception as e:
            print(f"Error extracting step information: {e}")
            print(f"Using fallback step: {prev_feedback_step + 1}")
            step = prev_feedback_step + 1
        
        return feedback, step, response
    
    def _generate_self_critique_feedback(self, x, response, y_steps_list, prev_responses, prev_feedbacks, prev_feedback_step):
        """
        Generate feedback that incorporates self-critique of previous feedback effectiveness.
        
        Args:
            x: The question
            response: The student's current response
            y_steps_list: List of correct solution steps
            prev_responses: List of previous student responses
            prev_feedbacks: List of previous feedback messages
            prev_feedback_step: The step number reached in previous feedback
            
        Returns:
            Tuple of (feedback, step_number)
        """
        # Create a timeline of previous interactions for the prompt
        interaction_history = ""
        for i, (prev_resp, prev_fb) in enumerate(zip(prev_responses, prev_feedbacks)):
            interaction_history += f"""
ROUND {i+1}:
Student Attempt:
{prev_resp}

Feedback Provided:
{prev_fb}
"""
        
        # Custom prompt function for self-critique feedback
        def self_critique_prompt(question, student, label, history=None):
            return f"""Review this interaction between teacher and student on a math problem:

QUESTION:
{question}

CORRECT SOLUTION STEPS:
{label}

HISTORY OF FEEDBACK AND ATTEMPTS:
{interaction_history}

CURRENT STUDENT ATTEMPT:
{student}

First, analyze the effectiveness of previous feedback:
1. What aspects of your feedback were helpful to the student?
2. What aspects didn't lead to improvement?
3. Which concepts is the student still struggling with?
4. What feedback strategy would work better now?

Then, generate new refined feedback that:
- Addresses specific errors in the current attempt
- Takes a different approach for concepts they're still struggling with
- Focuses on the earliest point where reasoning diverges from the correct solution
- Never reveals the complete solution or final answer

FORMAT YOUR RESPONSE:
Self-Analysis: [A brief analysis of previous feedback effectiveness]
Step X: [The specific step number where intervention is needed]
Refined Feedback: [Your improved feedback goes here]
"""

        # Use the teacher's generate method with correct signature
        solution_steps = "\n".join(y_steps_list)
        _, feedback = self.teacher.generate(
            question=x,
            student_steps=response,
            answer_steps=solution_steps,
            prompt_func=self_critique_prompt
        )
        
        # Extract the step number and refined feedback
        step = prev_feedback_step
        try:
            # Extract step number
            step_match = re.search(r"Step (\d+):", feedback)
            if step_match:
                step = int(step_match.group(1))
            
            # Extract just the refined feedback if possible
            refined_section = re.search(r"Refined Feedback:(.*?)(?:$|Step \d+:)", feedback, re.DOTALL)
            if refined_section:
                feedback = "Feedback: " + refined_section.group(1).strip()
            
            # If unable to extract step properly, use the next step from previous feedback
            if not step_match:
                step = prev_feedback_step + 1
                
        except Exception as e:
            print(f"Error extracting self-critique information: {e}")
            print(f"Using fallback step: {prev_feedback_step + 1}")
            step = prev_feedback_step + 1
        
        return feedback, step, response