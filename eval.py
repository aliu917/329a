import argparse
import os
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from project.models import StudentLMAgent
from project.experiments import prompts
from project import datasets
from project.utils import extract_text_within_box
from project.models import verifiers
verifier = verifiers.MATH500Verifier()

def eval(dataset, train_result_path, keys, log_dir=None, mode="test", num_examples=10, trials=1, num_context_examples=10, seed=2809):
    dataset = vars(datasets)[dataset](mode=mode)
    if not log_dir:
        log_dir = os.path.dirname(train_result_path) + "/eval"
    os.makedirs(log_dir, exist_ok=True)

    torch.manual_seed(seed)
    with open(train_result_path, "r") as file:
        train_result_data = json.load(file)
    student_context_examples = prompts.get_eval_student_context(train_result_data, keys, num_context_examples)
    student = StudentLMAgent(log_dir=log_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    results = []
    acc = 0
    for i, (data, labels) in tqdm(zip(range(num_examples), dataloader)):
        x = data["problem"][0]
        y = labels[0]
        answer = str(data["answer"].item())
        for _ in range(trials):
            # student_context_examples = prompts.get_eval_student_context(train_result_data[i:], keys, 1)
            student_pred = student.generate(x, history=student_context_examples)
            # Compare to numerical answer as well for multiple choice questions
            is_correct = verifier(student_pred, y) or verifier(student_pred, answer)
            if is_correct:
                break
        acc += is_correct
        result = {
            "question": x,
            "answer": answer,
            "pred": extract_text_within_box(student_pred),
            "pred_steps": student_pred,
            "answer_steps": y,
            "correct": is_correct,
        }
        results.append(result)

    with open(log_dir + "/eval_results.json", "w") as f:
        json.dump(results, f)

    final_acc = acc / num_examples
    print("final acc: ", final_acc)
    return final_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, default=None)
    parser.add_argument("--log_dir", "-l", type=str, default=None)
    parser.add_argument("--train_result_path", "-f", type=str, default=None)
    parser.add_argument("--keys", "-k", type=str, default="answer_steps")
    parser.add_argument("--mode", "-m", type=str, default="test")
    parser.add_argument("--num_eval_samples", "-n", type=int, default=10)
    parser.add_argument("--num_context_examples", "-c", type=int, default=10)
    parser.add_argument("--trials", "-t", type=int, default=1)
    args = parser.parse_args()
    eval(
        args.dataset,
        args.train_result_path,
        args.keys.split(","),
        log_dir=args.log_dir,
        mode=args.mode,
        num_examples=args.num_eval_samples,
        trials=args.trials,
        num_context_examples=args.num_context_examples,
    )