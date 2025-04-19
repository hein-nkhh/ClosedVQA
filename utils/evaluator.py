import torch
import torch.nn.functional as F

class Evaluator_Pretrain:
    def __init__(self, images, texts, model, device):
        self.images = images
        self.texts = texts
        self.model = model
        self.device = device

    def evaluate(self, val_loader):
        self.model.eval()

        total_similarity = 0.0
        correct_retrievals_r1 = 0
        correct_retrievals_r5 = 0
        correct_retrievals_r10 = 0
        total_samples = 0

        with torch.no_grad():
            for images, texts in val_loader:
                image_embeddings, text_embeddings = self.model(texts, images)

                similarity = torch.matmul(image_embeddings, text_embeddings.T)  # (B, B)

                labels = torch.arange(len(images)).to(self.device)
                top1 = similarity.argmax(dim=1)
                correct_retrievals_r1 += (top1 == labels).sum().item()

                top5 = similarity.topk(5, dim=1).indices
                correct_retrievals_r5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()

                top10 = similarity.topk(10, dim=1).indices
                correct_retrievals_r10 += (top10 == labels.unsqueeze(1)).any(dim=1).sum().item()

                total_similarity += torch.diag(similarity).mean().item()
                total_samples += len(images)

        return {
            "Recall@1": correct_retrievals_r1 / total_samples,
            "Recall@5": correct_retrievals_r5 / total_samples,
            "Recall@10": correct_retrievals_r10 / total_samples,
            "Avg Cosine Similarity": total_similarity / len(val_loader)
        }
