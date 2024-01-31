# Training Loop
from utils import *

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0
savedstate_dir = "../savedstate"

# Check if the savedstate directory exists, if not, create it
if not os.path.exists(savedstate_dir):
    os.makedirs(savedstate_dir)

# List all files in the savedstate directory
files = os.listdir(savedstate_dir)

# Filter checkpoint files
checkpoint_files = [file for file in files if file.startswith("checkpoint_epoch")]

# Sort checkpoint files by epoch number
checkpoint_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

if checkpoint_files:
    # Get the latest checkpoint file
    latest_checkpoint_file = checkpoint_files[-1]

    # Load the states from the latest checkpoint file
    savedstate_path = os.path.join(savedstate_dir, latest_checkpoint_file)
    checkpoint = torch.load(savedstate_path)
    netG.load_state_dict(checkpoint['generator_state_dict'])
    netD.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
    optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    G_losses = [checkpoint['lossG']]
    D_losses = [checkpoint['lossD']]
    print(f"Loaded checkpoint from {savedstate_path}. Resuming training from epoch {start_epoch}.")
else:
    start_epoch = 0
    print("No checkpoint found. Starting training from epoch 0.")
print("Starting Training Loop...")
# Training Loop starts from the loaded or default epoch
for epoch in range(start_epoch, num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

        # Save the models and optimizers after every 50 epochs
        if (epoch + 1) % savepoint == 0:
            checkpoint_file = os.path.join(savedstate_dir, f'checkpoint_epoch_{epoch + 1}.pth')

            torch.save({
                'epoch': epoch,
                'generator_state_dict': netG.module.state_dict(),
                'discriminator_state_dict': netD.module.state_dict(),
                'optimizerG_state_dict': optimizerG.state_dict(),
                'optimizerD_state_dict': optimizerD.state_dict(),
                'lossG': G_losses,
                'lossD': D_losses,
                'img_list': img_list,
                'iterations': iters
            }, checkpoint_file)
print("Training complete. Checkpoints saved.")
