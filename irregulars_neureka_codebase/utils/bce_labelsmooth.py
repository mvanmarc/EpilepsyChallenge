
import torch
import torch.nn as nn


class BinaryCrossEntropyWithLabelSmoothingAndWeights(nn.Module):
    def __init__(self, smoothing=0.2, n_bckg=1.0, n_seiz=0.1):
        super(BinaryCrossEntropyWithLabelSmoothingAndWeights, self).__init__()
        self.smoothing = smoothing
        self.n_bckg = n_bckg
        self.n_seiz = n_seiz
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')  # Keep reduction as 'none' to calculate sample-wise loss

    def forward(self, inputs, targets):
        # Apply label smoothing
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing

        # Compute BCE loss for each sample
        loss = self.bce_loss(inputs, targets)

        # Calculate sample weights based on the formula provided
        sample_weight = (self.n_bckg / self.n_seiz) * targets + (1 - targets)

        # Apply sample weights to the loss
        weighted_loss = loss * sample_weight

        # Return the mean loss (or sum, depending on the use case)
        return weighted_loss.mean()

if __name__ == '__main__':
    # Test the custom loss function
    loss = BinaryCrossEntropyWithLabelSmoothingAndWeights()
    inputs = torch.tensor([[0.1], [0.9], [0.5]])
    targets = torch.tensor([[0.0], [1.0], [0.5]])
    output = loss(inputs, targets)
    print(output)