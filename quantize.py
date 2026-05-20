import torch.ao.quantization as quant

def apply_qat(model, train_loader, epochs=1):
    # Configure for edge hardware [cite: 221, 238]
    model.qconfig = quant.get_default_qat_qconfig('fbgemm')
    quant.prepare_qat(model, inplace=True)
    
    # Short fine-tuning phase [cite: 238, 239]
    # (Insert standard training loop here)
    
    # Convert to fully quantized INT8 model [cite: 222, 240]
    quant.convert(model.eval(), inplace=True)
    return model