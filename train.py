import argparse
import project

def train(dataset, method, log_dir, num_examples=10):
    # test_dataset = MathDataset(dir="MATH_debug", mode="debug")
    dataset_cls = vars(project.datasets)[dataset]
    experiment_cls = vars(project.experiments)[method]
    experiment = experiment_cls(dataset_cls, log_dir, num_examples=num_examples)
    experiment.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, default=None)
    parser.add_argument("--method", "-m", type=str, default=None)
    parser.add_argument("--log_dir", "-l", type=str, default=None)
    parser.add_argument("--num_examples", "-n", type=int, default=10)
    args = parser.parse_args()
    train(args.dataset, args.method, args.log_dir, num_examples=args.num_examples)