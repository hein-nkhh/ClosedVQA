import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, device=None):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits_per_image, logits_per_text):
        """
        Args:
            logits_per_image (Tensor): [batch_size, batch_size]
            logits_per_text (Tensor): [batch_size, batch_size]
        Returns:
            Tensor: scalar contrastive loss
        """
        labels = torch.arange(logits_per_image.shape[0], device=self.device)
        loss_img = self.criterion(logits_per_image / self.temperature, labels)
        loss_text = self.criterion(logits_per_text / self.temperature, labels)
        return (loss_img + loss_text) / 2
