# Training Loop
from torch import optim
import os
import utils
from model import Generator, Discriminator
import torch


# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0
savedstate_dir = "savedstate"

# Initialize Generator and Discriminator
netG = Generator(utils.ngpu).to(utils.device)
netD = Discriminator(utils.ngpu).to(utils.device)

# Optimizers
optimizerD = optim.Adam(netD.parameters(), lr=utils.lr, betas=(utils.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=utils.lr, betas=(utils.beta1, 0.999))

print("Starting Training Loop...")

# Training Loop starts
for epoch in range(utils.num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(utils.dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(utils.device)
        b_size = real_cpu.size(0)
        label = utils.torch.full((b_size,), utils.real_label, dtype=utils.torch.float, device=utils.device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = utils.criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        # Train with all-fake batch
        noise = utils.torch.randn(b_size, utils.nz, 1, 1, device=utils.device)
        fake = netG(noise)
        label.fill_(utils.fake_label)
        output = netD(fake.detach()).view(-1)
        errD_fake = utils.criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(utils.real_label)  # fake labels are real for generator cost
        output = netD(fake).view(-1)
        errG = utils.criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, utils.num_epochs, i, len(utils.dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == utils.num_epochs - 1) and (i == len(utils.dataloader) - 1)):
            with utils.torch.no_grad():
                fake = netG(utils.fixed_noise).detach().cpu()
            img_list.append(utils.vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

        # Save the models and optimizers after epochs 50 and 500
        if (epoch + 1) == 20 or (epoch + 1) == 500 or (epoch + 1) == 100 or (epoch + 1) == 700:
            checkpoint_file = os.path.join(savedstate_dir, f'checkpoint_epoch_{epoch + 1}.pth')

            torch.save({
                'epoch': epoch,
                'generator_state_dict': netG.state_dict(),
                'discriminator_state_dict': netD.state_dict(),
                'optimizerG_state_dict': optimizerG.state_dict(),
                'optimizerD_state_dict': optimizerD.state_dict(),
                'lossG': G_losses,
                'lossD': D_losses,
                'img_list': img_list,
                'iterations': iters
            }, checkpoint_file)
            print(f"Saved checkpoint at epoch {epoch + 1}.")

print("Training complete. Checkpoints saved.")