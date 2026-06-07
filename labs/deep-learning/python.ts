// Runnable Python exports for the Deep Learning labs (PyTorch).

export const resnetPython = (depth = 20, residual = true, gain = 0.9) => `import torch, torch.nn as nn

# Plain vs residual deep net — mirrors the lab (depth=${depth}, residual=${residual})
# Watch the gradient norm reaching the FIRST layer: it vanishes in the plain net
# but stays ~O(1) through the skip connections of the residual net.
class Block(nn.Module):
    def __init__(self, d, residual):
        super().__init__(); self.fc = nn.Linear(d, d); self.act = nn.Tanh(); self.residual = residual
        nn.init.normal_(self.fc.weight, std=${gain})   # weight scale (gain)
    def forward(self, x):
        y = self.act(self.fc(x))
        return x + y if self.residual else y

d, depth, residual = 16, ${depth}, ${residual}
net = nn.Sequential(*[Block(d, residual) for _ in range(depth)])
x = torch.randn(8, d, requires_grad=True)
out = net(x).sum(); out.backward()

# gradient norm at each layer's weights (layer 0 = closest to input)
for i, b in enumerate(net):
    print(i, float(b.fc.weight.grad.norm()))
`;

export const batchNormPython = (depth = 12, useBN = true, initScale = 1.4) => `import torch, torch.nn as nn

# Activation statistics with vs without BatchNorm — mirrors the lab
# (depth=${depth}, batchnorm=${useBN}, init scale=${initScale})
class Net(nn.Module):
    def __init__(self, d, depth, bn):
        super().__init__()
        layers = []
        for _ in range(depth):
            lin = nn.Linear(d, d); nn.init.normal_(lin.weight, std=${initScale})
            layers += [lin] + ([nn.BatchNorm1d(d)] if bn else []) + [nn.Tanh()]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        acts = []
        for layer in self.net:
            x = layer(x)
            if isinstance(layer, nn.Tanh): acts.append((x.mean().item(), x.std().item()))
        return acts

net = Net(32, ${depth}, ${useBN}); net.train()
for i, (m, s) in enumerate(net(torch.randn(256, 32))):
    print(f"layer {i}: mean {m:+.3f}  std {s:.3f}")   # without BN these drift/collapse
`;

export const dropoutPython = (p = 0.3, epochs = 200) => `import torch, torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# Dropout as regularisation — mirrors the lab (dropout p=${p})
X, y = make_moons(n_samples=200, noise=0.3, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0)
Xtr, ytr = torch.tensor(Xtr).float(), torch.tensor(ytr).float()
Xte, yte = torch.tensor(Xte).float(), torch.tensor(yte).float()

net = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Dropout(${p}),
                    nn.Linear(64, 64), nn.ReLU(), nn.Dropout(${p}), nn.Linear(64, 1))
opt = torch.optim.Adam(net.parameters(), 1e-2); loss = nn.BCEWithLogitsLoss()
for _ in range(${epochs}):
    net.train(); opt.zero_grad()
    loss(net(Xtr).squeeze(), ytr).backward(); opt.step()

net.eval()   # dropout OFF at eval
acc = lambda X, y: ((net(X).squeeze() > 0).float() == y).float().mean().item()
print("train acc:", acc(Xtr, ytr), " val acc:", acc(Xte, yte))   # smaller gap with dropout
`;

export const transferPython = (nLabeled = 20, freeze = true) => `import torch, torch.nn as nn

# Transfer learning vs from-scratch on few labels — mirrors the lab
# (labelled examples=${nLabeled}, freeze backbone=${freeze})
backbone = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
# ... assume 'backbone' was pretrained on a large related dataset and loaded here ...
if ${freeze ? 'True' : 'False'}:
    for prm in backbone.parameters(): prm.requires_grad_(False)

head = nn.Linear(32, 1)                     # small task-specific head
model = nn.Sequential(backbone, head)
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.Adam(params, 1e-2)        # only the head (and unfrozen layers) train
# Train 'model' on your ${nLabeled} labelled points — a frozen pretrained backbone
# reaches high accuracy from far fewer labels than training everything from scratch.
print("trainable params:", sum(p.numel() for p in params))
`;

export const optimizersPython = (optimizer = 'adam', lr = 0.05, schedule = 'cosine') => `import torch

# Optimizers & LR schedules on a hard 2-D loss (Rosenbrock) — mirrors the lab
# (optimizer=${optimizer}, lr=${lr}, schedule=${schedule})
def rosenbrock(p): return (1 - p[0])**2 + 100 * (p[1] - p[0]**2)**2

p = torch.tensor([-1.5, 2.0], requires_grad=True)
opt = {
    'sgd':      lambda: torch.optim.SGD([p], lr=${lr}),
    'momentum': lambda: torch.optim.SGD([p], lr=${lr}, momentum=0.9),
    'rmsprop':  lambda: torch.optim.RMSprop([p], lr=${lr}),
    'adam':     lambda: torch.optim.Adam([p], lr=${lr}),
}['${optimizer}']()
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200) if '${schedule}' == 'cosine' else None

for step in range(200):
    opt.zero_grad(); rosenbrock(p).backward(); opt.step()
    if sched: sched.step()
print("final point:", p.detach().tolist(), " loss:", float(rosenbrock(p)))
`;
