import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.5):
    # KL Divergence for "soft" knowledge 
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T ** 2)
    
    # Standard Cross-Entropy for "hard" labels [cite: 234]
    hard_loss = F.cross_entropy(student_logits, labels)
    
    return alpha * hard_loss + (1 - alpha) * soft_loss