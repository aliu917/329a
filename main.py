from load_data import *
from project.utils import *
from tqdm import tqdm
import json
import project.methods.prompts as prompts
from project.methods.models import *

def run(run_name, limit=10):
    run_dir = f"runs/{run_name}"
    os.makedirs(run_dir, exist_ok=True)

    # Test run single correct/incorrect iteration and log prompts
    test_run_dir = f"runs/{run_name}/test"
    os.makedirs(test_run_dir, exist_ok=True)
    student = StudentLMAgent(log_dir=test_run_dir)
    teacher = TeacherLMAgent(log_dir=test_run_dir)
    test_dataset = MathDataset(dir="MATH_debug", mode="debug")
    run_all(test_dataset, test_run_dir, student, teacher, limit=2)

    # Actual run
    student = StudentLMAgent()
    teacher = TeacherLMAgent()
    dataset = MathDataset()
    run_all(dataset, run_dir, student, teacher, limit)

def run_all(dataset, run_dir, student, teacher, limit=10):

    torch.manual_seed(2809)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    initial_correct_count = 0
    after_correct_count = 0
    results = []
    base_results = []
    for i, (data, labels) in tqdm(enumerate(dataloader)): # ignore for now, don't need dataloader
        if i >= limit:
            break
        x = data["problem"][0]
        y = labels[0]

        # TODO: rm this chunk (baseline)
        result = run_single(x, y, student, teacher, "")
        base_results.append(result)
        with open(f"{run_dir}/base_results.json", "w") as f:
            json.dump(base_results, f)
        initial_correct_count += result["correct"]
        feedback = result["feedback"] if not result["correct"] else ""

        result = run_single(x, y, student, teacher, feedback)
        results.append(result)
        with open(f"{run_dir}/results.json", "w") as f:
            json.dump(results, f)
        after_correct_count += result["correct"]

    print("initial acc: ", initial_correct_count / len(results))
    print("final acc: ", after_correct_count / len(results))


def run_single(x, y, student, teacher, prev_feedback):
        # TODO: teacher learning
        feedback = prev_feedback # update with teacher learning process

        # Student generation and evaluation
        response = student.generate(x, feedback=feedback)
        correct, pred, answer, feedback = teacher.evaluate(x, response, y)
        result = {
            "question": x,
            "answer": answer,
            "pred": pred,
            "pred_steps": response,
            "answer_steps": y,
            "correct": correct,
            "feedback" : feedback,
        }
        return result

if __name__ == '__main__':
    run("baseline_test", limit=10)
