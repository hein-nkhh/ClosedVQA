import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.pretrain_model import PretrainModel
from configs.path import Paths
from utils.dataset import PretrainDataset
from loss_function import ContrastiveLoss
from utils.evaluator import Evaluator_Pretrain

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset & DataLoader
    pretrain_dataset_train, pretrain_dataset_val = PretrainDataset(Paths.PRETRAIN_TRAIN_JSON, Paths.PRETRAIN_VALID_JSON)
    pretrain_loader_train = DataLoader(pretrain_dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    pretrain_loader_val = DataLoader(pretrain_dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model, Loss, Optimizer
    model = PretrainModel(device=device).to(device)
    contrastive_loss = ContrastiveLoss(temperature=args.temperature, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    evaluator = Evaluator_Pretrain(images=None, texts=None, model=model, device=device)

    best_r1 = 0.0
    current_epoch = 0

    # Nếu có checkpoint, load mô hình và optimizer
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_r1 = checkpoint['best_r1']  # Cập nhật best_r1 nếu checkpoint đã lưu
        current_epoch = checkpoint['epoch']

        print(f"Resumed training from checkpoint {args.checkpoint}")

    for epoch in range(current_epoch, args.epochs):
        model.train()
        epoch_loss = 0

        for images, texts in tqdm(pretrain_loader_train, desc=f"Epoch {epoch + 1}", unit="batch"):
            images = images.to(device)
            texts = texts.to(device)

            optimizer.zero_grad()
            image_embeddings, text_embeddings = model(texts, images)

            logits_per_image = torch.matmul(image_embeddings, text_embeddings.T)
            logits_per_text = logits_per_image.T

            loss = contrastive_loss(logits_per_image, logits_per_text)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(pretrain_loader_train)
        print(f"[Epoch {epoch + 1}] Train Loss: {avg_loss:.4f}")

        # Evaluation
        metrics = evaluator.evaluate(pretrain_loader_val)
        print(f"[Eval] R@1: {metrics['Recall@1']:.4f}, R@5: {metrics['Recall@5']:.4f}, "
              f"R@10: {metrics['Recall@10']:.4f}, CosSim: {metrics['Avg Cosine Similarity']:.4f}")

        # Save best model if Recall@1 improves
        if metrics['Recall@1'] > best_r1:
            best_r1 = metrics['Recall@1']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_r1': best_r1,
            }, "best_pretrain_model.pt")
            print(">> Saved best model!\n")

        # Save the latest model after each epoch
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_r1': best_r1,
        }, "latest_pretrain_model.pt")
        print(">> Saved latest model!\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--step_size", type=int, default=5, help="Step size for LR scheduler")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma for LR scheduler")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature for contrastive loss")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file to resume training")

    args = parser.parse_args()
    train(args)
