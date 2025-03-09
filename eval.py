import argparse
from project.models import StudentLMAgent
from project.experiments import prompts
from project import datasets
import os
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from project.utils import extract_text_within_box
from project.models import verifiers
verifier = verifiers.MATH500Verifier()

def eval(dataset, train_result_path, keys, log_dir=None, num_examples=10, num_context_examples=10):
    dataset = vars(datasets)[dataset](mode="test")
    if not log_dir:
        log_dir = os.path.dirname(train_result_path)

    student_context_examples = prompts.get_eval_student_context(train_result_path, keys, num_context_examples)
    student = StudentLMAgent(log_dir=log_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    results = []
    acc = 0
    for i, (data, labels) in tqdm(zip(range(num_examples), dataloader)):
        x = data["problem"][0]
        y = labels[0]
        student_pred = student.generate(x, history=student_context_examples)
        is_correct = verifier(student_pred, y)
        acc += is_correct
        result = {
            "question": x,
            "answer": extract_text_within_box(y),
            "pred": extract_text_within_box(student_pred),
            "pred_steps": student_pred,
            "answer_steps": y,
            "correct": is_correct,
        }
        results.append(result)

    final_acc = acc / num_examples
    print("final acc: ", final_acc)
    return final_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, default=None)
    parser.add_argument("--log_dir", "-l", type=str, default=None)
    parser.add_argument("--train_result_path", "-f", type=str, default=None)
    parser.add_argument("--keys", "-", type=str, default="solution")
    parser.add_argument("--num_eval_samples", "-n", type=int, default=10)
    parser.add_argument("--num_context_examples", "-c", type=int, default=10)
    args = parser.parse_args()
    eval(args.dataset, args.train_result_path, args.keys.split(","), log_dir=args.log_dir, num_examples=args.num_examples)