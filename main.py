from load_data import *
from models import *
from utils import *


torch.manual_seed(2809)

def evaluate(pred, label):
    #TODO: better evaluation method
    return extract_text_within_box(pred) == extract_text_within_box(label)

def train():
    dataset = MathDataset()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    student = StudentLMAgent()
    teacher = TeacherLMAgent()

    for data, labels in dataloader:
        results = []
        for x,y in zip(data["problem"], labels):
            response = student.generate(x)
            result = {
                "question" : x,
                "answer" : extract_text_within_box(y),
                "pred" : extract_text_within_box(response),
                "pred_steps" : response,
                "correct" : evaluate(response, y)
            }
            results.append(result)

        for x, y, student_res in zip(data["problem"], labels, results):
            if not student_res["correct"]:
                correct, feedback = teacher.generate(x, student_res["pred_steps"], y)
                student_res["feedback"] = feedback

        pprint(results)
        break


if __name__ == '__main__':
    train()
