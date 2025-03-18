import argparse
import os
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from project.models import StudentLMAgent
from project.experiments import prompts
from project import datasets
from project.models import verifiers

verifier = verifiers.MATH500Verifier()


def score_results(
    student, context_results, val_dataloader, context_keys, num_val_examples
):
    student_context_examples = prompts.get_eval_student_context(
        context_results, context_keys, len(context_results)
    )
    num_correct = 0
    for _, (data, labels) in zip(range(num_val_examples), val_dataloader):
        x = data["problem"][0]
        y = labels[0]
        answer = str(data["answer"].item())
        student_pred = student.generate(x, history=student_context_examples)
        is_correct = verifier(student_pred, y) or verifier(student_pred, answer)
        num_correct += is_correct
    return num_correct / num_val_examples


def val_filter(
    dataset_cls, train_result_path, student_model, context_keys, num_val_examples, seed=2809
):
    with open(train_result_path, "r") as file:
        train_results = json.load(file)

    torch.manual_seed(seed)
    dataset = vars(datasets)[dataset_cls](mode="val")
    val_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    student = StudentLMAgent(model=student_model)

    filtered_results = []
    best_accuracy = 0
    for result in tqdm(train_results):
        candidate_results = filtered_results + [result]
        candidate_accuracy = score_results(
            student, filtered_results, val_dataloader, context_keys, num_val_examples
        )
        print("Best accuracy:", best_accuracy)
        print("Candidate accuracy:", candidate_accuracy)
        if candidate_accuracy >= best_accuracy:
            filtered_results = candidate_results
            best_accuracy = candidate_accuracy

    val_filter_path = os.path.dirname(train_result_path) + "/filtered_results.json"
    with open(val_filter_path, "w") as f:
        json.dump(filtered_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, required=True)
    parser.add_argument("--train_result_path", "-f", type=str, required=True)
    parser.add_argument(
        "--student_model", "-s", type=str, default="Qwen/Qwen2.5-7B-Instruct-Turbo"
    )
    parser.add_argument("--keys", "-k", type=str, default="prev_feedback,pred_steps")
    parser.add_argument("--num_val_samples", "-n", type=int, default=30)
    args = parser.parse_args()
    val_filter(
        args.dataset,
        args.train_result_path,
        args.student_model,
        args.keys.split(","),
        args.num_val_samples,
    )
