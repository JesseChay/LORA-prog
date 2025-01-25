import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import pairwise_distances



def check_image(file_path):
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            return True, width, height
    except:
        return False, None, None

def analyze_images(df, base_path='crawler'):
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        file_path = row['File Path']
        file_path = os.path.join(base_path, file_path)
        if os.path.exists(file_path):
            is_valid, width, height = check_image(file_path)
            if is_valid:
                aspect_ratio = width / height if height != 0 else None
            else:
                aspect_ratio = None
        else:
            is_valid = False
            aspect_ratio = None
        results.append({'is_valid': is_valid, 'aspect_ratio': aspect_ratio})
    return pd.DataFrame(results)




class ContrastiveDataset(Dataset):
    def __init__(self, df, transform=None, num_negatives=1):
        self.data = df.reset_index(drop=True)
        self.transform = transform
        self.num_negatives = num_negatives

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor_row = self.data.iloc[idx]
        anchor_image = self.load_image(anchor_row['File Path'])

        positive_idx = self.find_positive_pair(idx)
        positive_row = self.data.iloc[positive_idx]
        positive_image = self.load_image(positive_row['File Path'])

        negative_images = []
        for _ in range(self.num_negatives):
            negative_idx = self.find_negative_pair(idx, positive_idx)
            negative_row = self.data.iloc[negative_idx]
            negative_image = self.load_image(negative_row['File Path'])
            negative_images.append(negative_image)

        return anchor_image, positive_image, negative_images

    def load_image(self, image_path, base_path='crawler'):
        image_path = os.path.join(base_path, image_path)

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

    def find_positive_pair(self, idx):
        current_row = self.data.iloc[idx]
        same_model = self.data[self.data['Model ID'] == current_row['Model ID']]
        same_creator = self.data[self.data['Creator'] == current_row['Creator']]

        potential_pairs = pd.concat([same_model, same_creator])
        potential_pairs = potential_pairs[potential_pairs.index != idx]

        if len(potential_pairs) == 0:
            potential_pairs = self.data[self.data.index != idx]

        return random.choice(potential_pairs.index)

    def find_negative_pair(self, anchor_idx, positive_idx):
        anchor_row = self.data.iloc[anchor_idx]

        negative_candidates = self.data[
            (self.data['Model ID'] != anchor_row['Model ID']) & 
            (self.data['Creator'] != anchor_row['Creator'])
        ]

        if len(negative_candidates) == 0:
            negative_candidates = self.data[~self.data.index.isin([anchor_idx, positive_idx])]

        return random.choice(negative_candidates.index)

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, anchor, positive, negatives):
        targets = torch.cat([positive] + negatives, dim=0)
        sim_pos = torch.einsum('nc,nc->n', [anchor, positive]).unsqueeze(-1)
        sim_negs = torch.einsum('nc,nkc->nk', [anchor, torch.stack(negatives, dim=1)])
        similarities = torch.cat([sim_pos, sim_negs], dim=1) / self.temperature
        labels = torch.zeros(similarities.shape[0], dtype=torch.long, device=similarities.device)
        loss = self.cross_entropy(similarities, labels)
        return loss

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negatives):
        positive_dist = torch.norm(anchor - positive, dim=1)
        negative_dist = torch.norm(anchor - negatives[0], dim=1)
        losses = torch.relu(positive_dist - negative_dist + self.margin)
        return losses.mean()

def get_model(model_name, pretrained=True):
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Identity()
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Identity()
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier = nn.Identity()
    else:
        raise ValueError(f"Model {model_name} not supported")
    return model

def train(args, model, train_loader, val_loader, criterion, optimizer, scheduler, device):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for anchor, positive, negatives in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            anchor, positive = anchor.to(device), positive.to(device)
            negatives = [neg.to(device) for neg in negatives]

            optimizer.zero_grad()

            anchor_features = model(anchor)
            positive_features = model(positive)
            negative_features = [model(neg) for neg in negatives]

            loss = criterion(anchor_features, positive_features, negative_features)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for anchor, positive, negatives in val_loader:
                anchor, positive = anchor.to(device), positive.to(device)
                negatives = [neg.to(device) for neg in negatives]

                anchor_features = model(anchor)
                positive_features = model(positive)
                negative_features = [model(neg) for neg in negatives]

                loss = criterion(anchor_features, positive_features, negative_features)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))

        # Learning rate scheduling
        scheduler.step()

    # Save the final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))

    # Save training history
    history = pd.DataFrame({
        'epoch': range(1, args.epochs + 1),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    history.to_csv(os.path.join(args.output_dir, 'training_history.csv'), index=False)

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for anchor, positive, negatives in tqdm(test_loader, desc="Testing"):
            anchor, positive = anchor.to(device), positive.to(device)
            negatives = [neg.to(device) for neg in negatives]

            anchor_features = model(anchor)
            positive_features = model(positive)
            negative_features = [model(neg) for neg in negatives]

            loss = criterion(anchor_features, positive_features, negative_features)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")
    return avg_test_loss

def calculate_hit_rates(model, dataloader, device):
    model.eval()
    features = []
    labels = []
    acc = 0

    with torch.no_grad():
        for images, _, _ in tqdm(dataloader, desc="Extracting features"):
            indexs = np.arange(len(images)) + acc
            images = images.to(device)
            output = model(images)
            features.append(output.cpu().numpy())
            labels.extend(dataloader.dataset.data.iloc[indexs]['Model ID'].tolist())

    features = np.vstack(features)
    labels = np.array(labels)

    distances = pairwise_distances(features)

    top1_hits = 0
    top10_hits = 0
    total_queries = len(labels)

    for i in tqdm(range(total_queries), desc="Calculating hit rates"):
        sorted_indices = np.argsort(distances[i])[1:]
        if labels[sorted_indices[0]] == labels[i]:
            top1_hits += 1
        if labels[i] in labels[sorted_indices[:10]]:
            top10_hits += 1

    top1_hit_rate = top1_hits / total_queries
    top10_hit_rate = top10_hits / total_queries

    print(f"Top-1 Hit Rate: {top1_hit_rate:.4f}")
    print(f"Top-10 Hit Rate: {top10_hit_rate:.4f}")

    return top1_hit_rate, top10_hit_rate

def main(args):
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load and preprocess data
    df = pd.read_csv(args.data_csv)
    results = analyze_images(df)
    df = pd.concat([df, results], axis=1)

    df = df[df['is_valid'] == True].reset_index(drop=True)

    # Split data
    gss = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=args.seed)
    train_idx, temp_idx = next(gss.split(df, groups=df['Model ID']))
    gss_val = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=args.seed)
    val_idx, test_idx = next(gss_val.split(df.iloc[temp_idx], groups=df.iloc[temp_idx]['Model ID']))
    val_idx = temp_idx[val_idx]
    test_idx = temp_idx[test_idx]

    train_df, val_df, test_df = df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]

    # Data augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets and data loaders
    train_dataset = ContrastiveDataset(train_df, transform=train_transform, num_negatives=args.num_negatives)
    val_dataset = ContrastiveDataset(val_df, transform=test_transform, num_negatives=args.num_negatives)
    test_dataset = ContrastiveDataset(test_df, transform=test_transform, num_negatives=args.num_negatives)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Set up model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.model_name, pretrained=True).to(device)

    if args.loss == 'infonce':
        criterion = InfoNCELoss(temperature=args.temperature).to(device)
    elif args.loss == 'triplet':
        criterion = TripletLoss(margin=args.margin).to(device)
    else:
        raise ValueError(f"Loss function {args.loss} not supported")

    if args.finetune == 'all':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.finetune == 'head':
        for param in model.parameters():
            param.requires_grad = False
        if isinstance(model, models.ResNet):
            model.fc.requires_grad = True
        elif isinstance(model, models.EfficientNet):
            model.classifier.requires_grad = True
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    else:
        raise ValueError(f"Finetune option {args.finetune} not supported")

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Train the model
    train(args, model, train_loader, val_loader, criterion, optimizer, scheduler, device)

    # Test the model
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pth')))
    test_loss = test(model, test_loader, criterion, device)

    # Calculate hit rates
    top1_rate, top10_rate = calculate_hit_rates(model, test_loader, device)

    # Save results
    results = {
        'test_loss': test_loss,
        'top1_hit_rate': top1_rate,
        'top10_hit_rate': top10_rate
    }
    pd.DataFrame([results]).to_csv(os.path.join(args.output_dir, 'test_results.csv'), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contrastive Learning for Image Similarity")
    parser.add_argument('--data_csv', type=str, required=True, help='Path to the CSV file containing image data')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output files')
    parser.add_argument('--model_name', type=str, default='resnet50', choices=['resnet50', 'resnet18', 'efficientnet_b0'], help='Base model architecture')
    parser.add_argument('--finetune', type=str, default='all', choices=['all', 'head'], help='Finetuning strategy')
    parser.add_argument('--num_negatives', type=int, default=1, help='Number of negative samples per positive pair')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes for data loading')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--loss', type=str, default='infonce', choices=['infonce', 'triplet'], help='Loss function')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature parameter for InfoNCE loss')
    parser.add_argument('--margin', type=float, default=1.0, help='Margin parameter for Triplet loss')

    args = parser.parse_args()

    main(args)