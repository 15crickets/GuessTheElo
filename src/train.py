import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from data_formatting import format_data
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class MoveList(Dataset):
    def __init__(self, moves, ratings):
        self.moves = moves
        self.ratings = ratings

    def __len__(self):
        return len(self.moves)

    def __getitem__(self, idx):
        move_scores = self.moves[idx]
        rating = self.ratings[idx]

        move_tensor = torch.from_numpy(move_scores)
        rating = (rating-300)/2700

        sample={
            "input": move_tensor,
            "label": rating
        }

        return sample


class eloModel(nn.Module):
    def __init__(self, input_dim, hidden):
        super().__init__()
        self.move = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),

        )

        self.pool = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x/5
        x = self.move(x)
        x = x.mean(dim=1)
        x = self.pool(x).squeeze(-1)
        return x


def collate_fn(batch):
    out = {}
    for k in batch[0]:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out

scores = []
ratings = []

with open("lichess_db_standard_rated_2025-09.pgn") as f:
    lines = list(islice(f, 1000))


data = format_data(lines)
scores = data[0]
ratings = data[1]


train_xs, val_xs, train_ys, val_ys = train_test_split(
    scores, ratings, test_size=0.1, random_state=42
)


train_dataset = MoveList(train_xs, train_ys)
val_dataset   = MoveList(val_xs, val_ys)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, collate_fn=collate_fn)



model = eloModel(1, 32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


params = model.parameters()

optimizer = torch.optim.Adam(params, lr=1e-5)


loss_fn = nn.MSELoss()


for epoch in range(20):
    model.train()
    loop = tqdm(train_loader, leave=True)

    for batch in loop:
        scores = batch["input"].to(device)
        ratings = batch["label"].to(device)

        out = model(scores)
        loss = loss_fn(out, ratings.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                scores = batch["input"].to(device)
                ratings = batch["label"].to(device)
                out = model(scores)
                val_losses.append(loss_fn(out, ratings.float()).item())
        print(f"Epoch {epoch} val loss: {sum(val_losses)/len(val_losses):.4f}")

