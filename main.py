from load_data import *
from project.utils import *
from tqdm import tqdm
import json
import project.methods.prompts as prompts
from project.methods.models import *
from functools import partial


def run(run_name, limit=10):
    run_dir = f"eval_runs/{run_name}"
    os.makedirs(run_dir, exist_ok=True)

    # Test run single correct/incorrect iteration and log prompts
    # test_run_dir = f"runs/{run_name}/test"
    # os.makedirs(test_run_dir, exist_ok=True)
    # student = StudentLMAgent(log_dir=test_run_dir)
    # teacher = TeacherLMAgent(log_dir=test_run_dir)
    # test_dataset = MathDataset(dir="MATH_debug", mode="debug")
    # run_all(test_dataset, test_run_dir, student, teacher, limit=5)

    # Actual run
    student = StudentLMAgent()
    teacher = TeacherLMAgent()
    dataset = MathDataset()
    # run_all(dataset, run_dir, student, teacher, limit=100, evaluate=True, prompt_func=prompts.teacher_best_feedback_prompt)
    run_all(
        dataset,
        run_dir,
        student,
        teacher,
        limit=100,
        evaluate=True,
        prompt_func=partial(
            prompts.teacher_best_from_dataset_prompt, "runs/iterative/iterative_results.json"
        ),
    )


def run_all(
    dataset, run_dir, student, teacher, limit=10, evaluate=False, prompt_func=None
):
    torch.manual_seed(2809)
    if evaluate:
        last_500_dataset = torch.utils.data.Subset(
            dataset, range(len(dataset) - 500, len(dataset))
        )

        dataloader = DataLoader(last_500_dataset, batch_size=1, shuffle=True)
    else:
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # iterative_rounds = 10

    initial_correct_count = 0
    after_correct_count = 0
    # iterative_correct_counts = [0] * iterative_rounds

    results = []
    base_results = []
    # iterative_results = []
    for i, (data, labels) in tqdm(
        enumerate(dataloader)
    ):  # ignore for now, don't need dataloader
        if i >= limit:
            break
        x = data["problem"][0]
        y = labels[0]

        # TODO: rm this chunk (baseline)
        result = run_single(x, y, student, teacher, "", prompt_func=prompt_func)
        base_results.append(result)
        with open(f"{run_dir}/base_results.json", "w") as f:
            json.dump(base_results, f)
        initial_correct_count += result["correct"]
        feedback = result["feedback"] if not result["correct"] else ""

        result = run_single(x, y, student, teacher, feedback, prompt_func=prompt_func)
        results.append(result)
        with open(f"{run_dir}/results.json", "w") as f:
            json.dump(results, f)
        after_correct_count += result["correct"]

        # cur_iterative_results = run_iterative(x, y, student, teacher, n_rounds=iterative_rounds, initial_feedback=feedback)
        # iterative_results.extend(cur_iterative_results)
        # with open(f"{run_dir}/iterative_results.json", "w") as f:
        #     json.dump(iterative_results, f)
        # for r, round_result in enumerate(cur_iterative_results):
        #     iterative_correct_counts[r] += round_result["correct"]

    print("initial acc: ", initial_correct_count / len(results))
    print("final acc: ", after_correct_count / len(results))
    # for r, round_count in enumerate(iterative_correct_counts):
    # print(f"Refinement Round {r} Accuracy:", round_count/len(results))


def run_single(x, y, student, teacher, prev_feedback, prompt_func=None):
    # TODO: teacher learning
    feedback = prev_feedback  # update with teacher learning process

    # Student generation and evaluation
    response = student.generate(x, feedback=feedback)
    correct, pred, answer, feedback = teacher.evaluate(
        x, response, y, prompt_func=prompt_func
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
    }
    return result


def run_iterative(x, y, student, teacher, n_rounds=3, initial_feedback=""):
    prev_feedback = initial_feedback
    results = []
    for refine_round in range(n_rounds):
        response = student.generate(x, feedback=prev_feedback)
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


if __name__ == "__main__":
    run("baseline", limit=20)
