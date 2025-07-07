import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
import torch.distributed as dist
from time import perf_counter as pc
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from config import get_parser
from models import get_model
from dataset import get_dataset
from utils import (
    setup_distributed_env, 
    initialize_process_group, 
    log_training_metrics, 
    calculate_classification_accuracy,
    calculate_span_prediction_accuracy,
    hadamard_hook_cuda,
    is_hadamard_available,
    thc_compress_hook
)

def train(args, file_prefix):
    # Global variables for Hadamard transform
    global hadamard, random_diag_encode, random_diag_decode
    
    # Setup random seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Get model and dataset
    model, tokenizer, optimizer, scheduler = get_model(args.model)
    dataset = get_dataset(args.model, tokenizer=tokenizer)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Distributed Data Parallel setup
    if args.model == "gpt2":
        model._ddp_params_and_buffers_to_ignore = [name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool] # This is the trick, you ask DDP to ignore all buffers that are in torch.bool because GLOO doesn't support bool.
    model = DDP(model, bucket_cap_mb=1350)
    
    # Hadamard transform setup
    if args.hadamard:
        if not is_hadamard_available:
            raise ImportError("CUDA Hadamard transform is not available. Install hadamard_cuda.")        
        # Register Hadamard communication hook if enabled
        model.register_comm_hook(model.process_group, hadamard_hook_cuda)

    if args.uhc:
        params = {}
        params["nclients"] = args.nodes
        params["d"] = {}
        params["size"] = {}
        params["device"] = "cuda:0"
        params["ef"] = False
        params["proportion"] = 0.001
        params["compression_method"] = "uhc"
        params["args"] = args
        params["quantization_levels"] = args.quantization_levels
        params["partial_rotation_times"] = args.partial_rotation_times
        hook_state = {"batch_idx": -1, "params": params, "start_idx": {}, "partition_len": {}, "ret_tensor": {}, "start_interm_idx": {}, "interm_reduce_tensor": {}, "args": args}
        model.register_comm_hook(state=[hook_state], hook=thc_compress_hook)

    # Dataloader setup
    train_sampler = DistributedSampler(dataset, num_replicas=args.nodes)
    train_loader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True,
                            drop_last=True,
                            sampler=train_sampler)
    
    # Optimizer setup
    total_steps = len(train_loader)
    num_training_steps = args.epochs * len(train_loader)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, 
                    num_warmup_steps=0, num_training_steps=num_training_steps)
    
    # Criterion setup for VGG19
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Tracking metrics
    epoch_times, epoch_acc, epoch_loss = [], [], []
    
    # Training loop
    for epoch in range(args.epochs):
        train_loss = 0.0
        train_acc = 0.0
        epoch_start = pc()
        
        for batch in tqdm(train_loader):
            # Zero gradients at the beginning of each iteration
            optimizer.zero_grad()
            
            if args.model in ["bert", "roberta"]:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                start_positions = batch['start_positions'].to(device)
                end_positions = batch['end_positions'].to(device)
                
                # Forward pass
                outputs = model(
                    input_ids, 
                    attention_mask=attention_mask,
                    start_positions=start_positions,
                    end_positions=end_positions
                )
                loss = outputs.loss
                
                # Backward pass and optimization
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                # Metrics - use span prediction accuracy
                train_loss += loss.item()
                train_acc += calculate_span_prediction_accuracy(outputs, 
                                start_positions, end_positions).item()
                
            elif args.model in ["gpt2", "bart"]:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[3].to(device)
                
                # Forward pass
                outputs = model(input_ids, attention_mask=attention_mask, 
                                labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                
                # Metrics - use standard classification accuracy
                train_loss += loss.item()
                train_acc += calculate_classification_accuracy(logits, labels).item()
                
            else: # VGG19
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                # Metrics - use standard classification accuracy
                train_loss += loss.item()
                train_acc += calculate_classification_accuracy(outputs, labels).item()
            
        # Normalize metrics by number of batches (consistent across all model types)
        train_loss /= total_steps
        train_acc /= total_steps
        epoch_time = pc() - epoch_start
        
        # Store metrics
        epoch_acc.append(train_acc)
        epoch_loss.append(train_loss)
        epoch_times.append(epoch_time)
        
        # Apply learning rate scheduler for models that need it
        if args.model not in ["gpt2", "bart"]:
            scheduler.step()
            
        print(f"Epoch {epoch+1}:\nEpoch Time: {epoch_time:.3f}s Train Loss: {train_loss:.3f}, Accuracy: {train_acc:.3f}")
        log_training_metrics(epoch_times, epoch_acc, epoch_loss, file_prefix)
    
    return model

def main():
    parser = get_parser()
    args = parser.parse_args()
    file_path = setup_distributed_env(args)
    initialize_process_group(args)
    start_time = datetime.now()
    train(args, file_path)
    end_time = datetime.now()
    dist.destroy_process_group()
    print(f"Training completed in {end_time - start_time}. Log files saved as {file_path}")

if __name__ == '__main__':
    main()
