from transformers import AutoModel, AutoTokenizer
import torch
import os

model_id = "jinaai/jina-embeddings-v5-text-nano"
save_dir = "./src/models/jina_v5_nano_onnx"

os.makedirs(save_dir, exist_ok=True)

# Load model
print(f"Loading {model_id}...")
model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Export to ONNX using torch.onnx.export with legacy exporter
onnx_model_path = os.path.join(save_dir, "model.onnx")

if not os.path.exists(onnx_model_path):
    print(f"Exporting to ONNX format to {onnx_model_path}...")
    
    # Prepare dummy inputs
    sentences = ["This is a sequence to embed", "Another example sentence"]
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    
    # Create a wrapper that handles the forward properly
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, input_ids, attention_mask):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state
    
    wrapper_model = ONNXWrapper(model)
    wrapper_model.eval()
    
    # Use the legacy ONNX exporter path
    with torch.no_grad():
        torch.onnx.export(
            wrapper_model,
            (inputs["input_ids"], inputs["attention_mask"]),
            onnx_model_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            opset_version=18,
            do_constant_folding=False,
            export_params=True,
            training=torch.onnx.TrainingMode.EVAL
        )
    
    # Save tokenizer config
    tokenizer.save_pretrained(save_dir)
    
    print(f"✓ ONNX model saved to {onnx_model_path}")
else:
    print(f"ONNX model already exists at {onnx_model_path}")

sentences = ["This is a sequence to embed", "Another example sentence"]
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

# 3. Generate Raw Embeddings
with torch.no_grad():
    outputs = model(**inputs)
    # The 'last_hidden_state' contains the embeddings for every token
    embeddings = outputs.last_hidden_state

# 4. Perform Mean Pooling (Standard for Jina/Sentence Transformers)
attention_mask = inputs['attention_mask']
mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
final_embeddings = sum_embeddings / sum_mask

print(final_embeddings.shape) # Should be [2, 768] for the nano model
