import argparse
import project

def train(dataset, method, log_dir, student_model, teacher_model, num_examples=10):
    # test_dataset = MathDataset(dir="MATH_debug", mode="debug")
    dataset = vars(project.datasets)[dataset]()
    experiment_cls = vars(project.experiments)[method]
    experiment = experiment_cls(dataset, student_model, teacher_model, log_dir, num_examples=num_examples)
    experiment.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, default=None)
    parser.add_argument("--method", "-m", type=str, default=None)
    parser.add_argument("--log_dir", "-l", type=str, default=None)
    parser.add_argument("--num_examples", "-n", type=int, default=10)
    parser.add_argument("--student_model", "-s", type=str, default="Qwen/Qwen2.5-7B-Instruct-Turbo")
    parser.add_argument("--teacher_model", "-t", type=str, default="gpt-4o-mini")
    args = parser.parse_args()
    train(args.dataset, args.method, args.log_dir, args.student_model, args.teacher_model, num_examples=args.num_examples)