import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle

class MoveList(Dataset):
    def __init__(self, moves, ratings):
        self.moves = moves
        self.ratings = ratings

    def __len__(self):
        return len(self.moves)

    def __getitem__(self, idx):
        move_scores = self.moves[idx]
        rating = self.ratings[idx]
        move_tensor = torch.tensor(move_scores, dtype=torch.float32)
        rating = torch.tensor((rating - 300) / 2700, dtype=torch.float32)

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
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch, seq_len) -> (batch, seq_len, 1)
        x = x/5
        x = self.move(x)
        x = x.mean(dim=1)
        x = self.pool(x).squeeze(-1)
        return x


def collate_fn(batch):
    inputs = [b["input"] for b in batch]
    labels = torch.stack([b["label"] for b in batch])
    inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    return {"input": inputs_padded, "label": labels}


if __name__ == "__main__":

    scores = []
    ratings = []

    initial_scores = []
    with open("../data/all_games.pkl", "rb") as f:
        while True:
            try:
                initial_scores.append(pickle.load(f))
            except EOFError:
                break

    initial_ratings = []
    with open("../data/game_ratings.pkl", "rb") as f:
        while True:
            try:
                initial_ratings.append(pickle.load(f))
            except EOFError:
                break


    for item in initial_scores:
        white_array = []
        black_array = []
        for i in range(len(item)):
            if i % 2 == 0:
                white_array.append(item[i])
            else:
                black_array.append(item[i])
        scores.append(white_array)
        scores.append(black_array)

    for item in initial_ratings:
        ratings.append(item[0])
        ratings.append(item[1])


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


    for epoch in range(10):
        model.train()
        loop = tqdm(train_loader, leave=True)
        running_loss = 0

        for batch in loop:
            scores = batch["input"].to(device)
            ratings = batch["label"].to(device)

            out = model(scores)
            loss = loss_fn(out, ratings.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{running_loss/(i+1):.4f}")


        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                scores = batch["input"].to(device)
                ratings = batch["label"].to(device)
                out = model(scores)
                val_losses.append(loss_fn(out, ratings.float()).item())
        print(f"Epoch {epoch} val loss: {sum(val_losses)/len(val_losses):.4f}")

    torch.save(model.state_dict(), "../data/elo_model.pth")
