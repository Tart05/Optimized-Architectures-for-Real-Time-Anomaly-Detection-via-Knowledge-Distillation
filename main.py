from data_loader import preprocess_data, get_loaders
from models import TeacherModel, StudentModel
from distillation import distillation_loss
from quantize import apply_qat
import torch.optim as optim

# 1. Prepare Data
X, y = preprocess_data("mqtt_dataset.csv")
loader = get_loaders(X, y)

# 2. Setup Models
teacher = TeacherModel(input_dim=X.shape[1]).eval() # Assume pre-trained [cite: 231, 232]
student = StudentModel(input_dim=X.shape[1])
optimizer = optim.AdamW(student.parameters(), lr=1e-3)

# 3. Distillation Loop [cite: 232]
for epoch in range(5):
    for data, target in loader:
        optimizer.zero_grad()
        with torch.no_grad():
            t_output = teacher(data)
        s_output = student(data)
        loss = distillation_loss(s_output, t_output, target)
        loss.backward()
        optimizer.step()

# 4. Quantization for Edge Efficiency [cite: 177, 240]
optimized_student = apply_qat(student, loader)
torch.save(optimized_student.state_dict(), "distill_guard_edge.pt")