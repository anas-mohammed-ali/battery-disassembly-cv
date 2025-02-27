import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

######################################
# 1. Dataset Class for Paired Data   #
######################################
class PairedDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = sorted(os.listdir(self.images_dir))
        self.mask_files = sorted(os.listdir(self.masks_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Build full paths
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])

        # Open images
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return mask, image


######################################
# 2. Define the U-Net Generator      #
######################################
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super(UNetGenerator, self).__init__()
        # Encoder (contracting path)
        self.down1 = self.contract_block(in_channels, features, 4, 2, 1)         # 256 -> 128
        self.down2 = self.contract_block(features, features*2, 4, 2, 1)            # 128 -> 64
        self.down3 = self.contract_block(features*2, features*4, 4, 2, 1)          # 64 -> 32
        self.down4 = self.contract_block(features*4, features*8, 4, 2, 1)          # 32 -> 16

        # Decoder (expanding path)
        self.up1 = self.expand_block(features*8, features*4, 4, 2, 1)              # 16 -> 32
        self.up2 = self.expand_block(features*8, features*2, 4, 2, 1)              # 32 -> 64 (skip from down3)
        self.up3 = self.expand_block(features*4, features, 4, 2, 1)                # 64 -> 128 (skip from down2)
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(features*2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()                                                             # Output range [-1,1]
        )

    def contract_block(self, in_channels, out_channels, kernel_size, stride, padding):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        return block

    def expand_block(self, in_channels, out_channels, kernel_size, stride, padding):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block

    def forward(self, x):
        # Encode
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        # Decode with skip connections
        u1 = self.up1(d4)
        u2 = self.up2(torch.cat([u1, d3], 1))
        u3 = self.up3(torch.cat([u2, d2], 1))
        u4 = self.up4(torch.cat([u3, d1], 1))
        return u4

######################################
# 3. Define the PatchGAN Discriminator #
######################################
class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=6, features=64):
        super(PatchGANDiscriminator, self).__init__()
        # in_channels = condition (mask) concatenated with image (real or fake)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features, features*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features*2, features*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features*4, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)

######################################
# 4. Training Setup and Loop         #
######################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 2e-4
num_epochs = 200
batch_size = 4

# Transformation: Resize to 256x256, convert to tensor, and normalize to [-1, 1]
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

# Create dataset and dataloader from the 'data/train' folder
dataset = PairedDataset(
    images_dir='data/images/train',
    masks_dir='data/masks/train',
    transform=transform
)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize generator and discriminator
gen = UNetGenerator().to(device)
disc = PatchGANDiscriminator().to(device)

# Loss functions
criterion_GAN = nn.BCEWithLogitsLoss()
criterion_L1 = nn.L1Loss()

# Optimizers
opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

print("Starting training...")
for epoch in range(num_epochs):
    for i, (mask, real_image) in enumerate(dataloader):
        mask = mask.to(device)
        real_image = real_image.to(device)

        # ---- Train Discriminator ----
        disc.zero_grad()
        # Real pair: (mask, real image)
        real_pair = torch.cat([mask, real_image], dim=1)
        pred_real = disc(real_pair)
        loss_disc_real = criterion_GAN(pred_real, torch.ones_like(pred_real))

        # Fake pair: (mask, fake image)
        fake_image = gen(mask)
        fake_pair = torch.cat([mask, fake_image.detach()], dim=1)
        pred_fake = disc(fake_pair)
        loss_disc_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))

        loss_disc = (loss_disc_real + loss_disc_fake) * 0.5
        loss_disc.backward()
        opt_disc.step()

        # ---- Train Generator ----
        gen.zero_grad()
        # Recompute fake pair for generator update
        fake_pair = torch.cat([mask, fake_image], dim=1)
        pred_fake = disc(fake_pair)
        loss_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
        loss_L1 = criterion_L1(fake_image, real_image) * 100  # L1 loss weight can be tuned
        loss_gen = loss_GAN + loss_L1
        loss_gen.backward()
        opt_gen.step()

        if i % 50 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch {i}/{len(dataloader)} "
                  f"Loss D: {loss_disc.item():.4f}, Loss G: {loss_gen.item():.4f}")

    # Save checkpoints periodically
    if epoch % 50 == 0:
        torch.save(gen.state_dict(), f"gen_epoch_{epoch}.pth")
        torch.save(disc.state_dict(), f"disc_epoch_{epoch}.pth")

print("Training complete.")
