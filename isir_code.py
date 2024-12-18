import os
import click
from tqdm.auto import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import tensorflow as tf
import io
from torchvision.utils import make_grid, save_image
import classifier_lib
import random
import time

#----------------------------------------------------------------------------
# Proposed DiffRS sampler.

def isir_sampler(img_batch, device, save_type, time_min, time_max, vpsde, discriminator,
    net, labels=None,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    outdir=None, class_idx=None, batch_size=100, num_samples=50000, cnt_inner_steps=20, isir_batch=30
):

    S_churn_vec = torch.tensor([S_churn] * batch_size, device=device)
    S_churn_max = torch.tensor([np.sqrt(2) - 1] * batch_size, device=device)
    S_noise_vec = torch.tensor([S_noise] * batch_size, device=device)
    gamma_vec = torch.minimum(S_churn_vec / num_steps, S_churn_max)

    def sampling_loop(x_next, lst_idx, labels):
        t_cur = t_steps[lst_idx]
        t_next = t_steps[lst_idx+1]

        x_cur = x_next
        x_next_inner = None

        log_ratio_prev = classifier_lib.get_grad_log_ratio(discriminator, vpsde, x_cur, t_steps[lst_idx], \
                                                           net.img_resolution, time_min, time_max, labels, log_only=True).detach().cpu()
        for i in range(cnt_inner_steps):
          weights = []
          proposal_x_nexts = []
          if i == 0:
            for j in range(isir_batch):
              bool_gamma = (S_min <= t_cur) & (t_cur <= S_max)

              if bool_gamma.sum() != 0:
                  t_hat_temp = net.round_sigma(t_cur + gamma_vec * t_cur)[bool_gamma]
                  x_hat_temp = x_cur[bool_gamma] + (t_hat_temp ** 2 - t_cur[bool_gamma] ** 2).sqrt()[:, None, None, None] * S_noise_vec[bool_gamma, None, None,None] * torch.randn_like(x_cur[bool_gamma])

                  t_hat = t_cur
                  x_hat = x_cur

                  t_hat[bool_gamma] = t_hat_temp
                  x_hat[bool_gamma] = x_hat_temp
              else:
                  t_hat = t_cur
                  x_hat = x_cur

              # Euler step.
              denoised = net(x_hat, t_hat, labels).to(torch.float64)

              d_cur = (x_hat - denoised) / t_hat[:, None, None, None]
              x_next = x_hat + (t_next - t_hat)[:, None, None, None] * d_cur

              # Apply 2nd order correction.
              bool_2nd = (lst_idx < num_steps - 1)
              if bool_2nd.sum() != 0:
                  labels_ = labels[bool_2nd] if labels is not None else None
                  denoised = net(x_next[bool_2nd], t_next[bool_2nd], labels_).to(torch.float64)
                  
                  d_prime = (x_next[bool_2nd] - denoised) / t_next[bool_2nd][:, None, None, None]
                  x_next[bool_2nd] = x_hat[bool_2nd] + (t_next - t_hat)[bool_2nd][:, None, None, None] * (0.5 * d_cur[bool_2nd] + 0.5 * d_prime)
              proposal_x_nexts.append(x_next)
              log_ratio = classifier_lib.get_grad_log_ratio(discriminator, vpsde, x_next, t_steps[lst_idx+1], net.img_resolution, \
                                                            time_min, time_max, labels, log_only=True).detach().cpu()
              weights.append(np.exp(log_ratio-log_ratio_prev))
          else:
            proposal_x_nexts.append(x_next_inner)
            log_ratio = classifier_lib.get_grad_log_ratio(discriminator, vpsde, x_next_inner, t_steps[lst_idx+1], net.img_resolution, \
                                                          time_min, time_max, labels, log_only=True).detach().cpu()
            weights.append(np.exp(log_ratio-log_ratio_prev))
            for j in range(isir_batch-1):
              bool_gamma = (S_min <= t_cur) & (t_cur <= S_max)

              if bool_gamma.sum() != 0:
                  t_hat_temp = net.round_sigma(t_cur + gamma_vec * t_cur)[bool_gamma]
                  x_hat_temp = x_cur[bool_gamma] + (t_hat_temp ** 2 - t_cur[bool_gamma] ** 2).sqrt()[:, None, None, None] * S_noise_vec[bool_gamma, None, None,None] * torch.randn_like(x_cur[bool_gamma])

                  t_hat = t_cur
                  x_hat = x_cur

                  t_hat[bool_gamma] = t_hat_temp
                  x_hat[bool_gamma] = x_hat_temp
              else:
                  t_hat = t_cur
                  x_hat = x_cur

              # Euler step.
              denoised = net(x_hat, t_hat, labels).to(torch.float64)

              d_cur = (x_hat - denoised) / t_hat[:, None, None, None]
              x_next = x_hat + (t_next - t_hat)[:, None, None, None] * d_cur

              # Apply 2nd order correction.
              bool_2nd = (lst_idx < num_steps - 1)
              if bool_2nd.sum() != 0:
                  labels_ = labels[bool_2nd] if labels is not None else None
                  denoised = net(x_next[bool_2nd], t_next[bool_2nd], labels_).to(torch.float64)

                  d_prime = (x_next[bool_2nd] - denoised) / t_next[bool_2nd][:, None, None, None]
                  x_next[bool_2nd] = x_hat[bool_2nd] + (t_next - t_hat)[bool_2nd][:, None, None, None] * (0.5 * d_cur[bool_2nd] + 0.5 * d_prime)
              proposal_x_nexts.append(x_next)
              log_ratio = classifier_lib.get_grad_log_ratio(discriminator, vpsde, x_next, t_steps[lst_idx+1], net.img_resolution, \
                                                              time_min, time_max, labels, log_only=True).detach().cpu()
              weights.append(np.exp(log_ratio-log_ratio_prev))
          proposal_x_nexts = torch.stack(proposal_x_nexts, dim=0)
          weights = torch.stack(weights, dim=0).detach().cpu()
          weights = weights/weights.sum(dim=0)
          cat = torch.distributions.categorical.Categorical(probs=weights.T)
          x_next_inner_ind = cat.sample()
          x_next_inner = proposal_x_nexts[x_next_inner_ind, np.arange(batch_size)]        
          
        lst_idx = lst_idx + 1
        return x_next_inner, lst_idx

    def save_img(images, index, save_type="npz", batch_size=100):
        ## Save images.
        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        if save_type == "png":
            count = 0
            for image_np in images_np:
                image_path = os.path.join(outdir, f'{index*batch_size+count:06d}.png')
                count += 1
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)

        elif save_type == "npz":
            # r = np.random.randint(1000000)
            with tf.io.gfile.GFile(os.path.join(outdir, f"samples_{index}.npz"), "wb") as fout:
                io_buffer = io.BytesIO()
                if labels == None:
                    np.savez_compressed(io_buffer, samples=images_np)
                else:
                    np.savez_compressed(io_buffer, samples=images_np, label=class_labels.cpu().numpy())
                fout.write(io_buffer.getvalue())

            nrow = int(np.sqrt(images_np.shape[0]))
            image_grid = make_grid(torch.tensor(images_np).permute(0, 3, 1, 2) / 255., nrow, padding=2)
            with tf.io.gfile.GFile(os.path.join(outdir, f"sample_{index}.png"), "wb") as fout:
                save_image(image_grid, fout)

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    current_time = time.time()
    index = 0
    for _ in range(num_samples//batch_size):
      # Main sampling loop.
      lst_idx = torch.zeros((batch_size,),).long()
      x_next_inner = None


      for i in range(cnt_inner_steps):
        weights = []
        proposal_x_nexts = []
        if i == 0:
          for j in range(isir_batch):
            latents = torch.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            x_next = latents.to(torch.float64) * t_steps[0]
            proposal_x_nexts.append(x_next)
            log_ratio = classifier_lib.get_grad_log_ratio(discriminator, vpsde, x_next, t_steps[lst_idx+1], net.img_resolution, time_min, time_max, labels, log_only=True).detach().cpu()
            weights.append(torch.exp(log_ratio)) 
        else:
          proposal_x_nexts.append(x_next_inner)
          for j in range(isir_batch-1):
            latents = torch.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            x_next = latents.to(torch.float64) * t_steps[0]
            proposal_x_nexts.append(x_next)
            log_ratio = classifier_lib.get_grad_log_ratio(discriminator, vpsde, x_next, t_steps[lst_idx+1], net.img_resolution, time_min, time_max, labels, log_only=True).detach().cpu()
            weights.append(torch.exp(log_ratio)) 
        proposal_x_nexts = torch.stack(proposal_x_nexts, dim=0)
        weights = torch.stack(weights, dim=0).detach().cpu()
        weights = weights/weights.sum(dim=0)
        cat = torch.distributions.categorical.Categorical(probs=weights.T)
        x_next_inner_ind = cat.sample()
        x_next_inner = proposal_x_nexts[x_next_inner_ind, np.arange(batch_size)] 
    
      x_0 = x_next_inner
      for t in tqdm(range(num_steps - 1)):
        x_0, lst_idx = sampling_loop(x_0, lst_idx, labels)
      for cnt_save_img in range(batch_size//img_batch):
        l = cnt_save_img * img_batch
        r = l + img_batch
        save_img(x_0[l:r], index, save_type, batch_size)
        index += 1
   
    print(time.time()-current_time)


#----------------------------------------------------------------------------
@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'batch_size',     help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=100, show_default=True)

@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)


#---------------------------------------------------------------------------- Options for Discriminator-Guidance
## Sampling configureation
@click.option('--do_seed',                 help='Applying manual seed or not', metavar='INT',                       type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--seed',                    help='Seed number',                 metavar='INT',                       type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--num_samples',             help='Num samples',                 metavar='INT',                       type=click.IntRange(min=1), default=50000, show_default=True)
@click.option('--save_type',               help='png or npz',                  metavar='png|npz',                   type=click.Choice(['png', 'npz']), default='npz')
@click.option('--device',                  help='Device', metavar='STR',                                            type=str, default='cuda:0')
@click.option('--cnt_inner_steps',         help='Num inner steps at each time',metavar='INT',                       type=click.IntRange(min=1), default=20, show_default=True)
@click.option('--isir_batch',              help='Num samples for isir',        metavar='INT',                       type=click.IntRange(min=1), default=30, show_default=True)
@click.option('--img_batch',               help='Num samples for one image',        metavar='INT',             type=click.IntRange(min=1), default=500, show_default=True)

## DG configuration
@click.option('--time_min',                help='Minimum time[0,1] to apply DG', metavar='FLOAT',                   type=click.FloatRange(min=0., max=1.), default=0.01, show_default=True)
@click.option('--time_max',                help='Maximum time[0,1] to apply DG', metavar='FLOAT',                   type=click.FloatRange(min=0., max=1.), default=1.0, show_default=True)

## Discriminator checkpoint
@click.option('--pretrained_classifier_ckpt',help='Path of ADM classifier(latent extractor)',  metavar='STR',       type=str, default='checkpoints/ADM_classifier/32x32_classifier.pt', show_default=True)
@click.option('--discriminator_ckpt',      help='Path of discriminator',  metavar='STR',                            type=str, default='checkpoints/discriminator/cifar_uncond/discriminator_60.pt', show_default=True)
@click.option('--cond',                    help='Is it conditional discriminator?', metavar='INT',                  type=click.IntRange(min=0, max=1), default=0, show_default=True)



def main(pretrained_classifier_ckpt, discriminator_ckpt, cond, save_type, isir_batch, cnt_inner_steps, 
         batch_size, do_seed, seed, num_samples, network_pkl, 
         outdir, class_idx, device, img_batch, **sampler_kwargs):
    ## Load pretrained score network.
    print(f'Loading network from "{network_pkl}"...')
    with open(network_pkl, 'rb') as f:
        net = pickle.load(f)['ema'].to(device)

    ## Load discriminator
    if 'ffhq' in network_pkl:
        depth = 4
    else:
        depth = 2
    discriminator = classifier_lib.get_discriminator(pretrained_classifier_ckpt, discriminator_ckpt,
                                                     net.label_dim and cond, net.img_resolution, device,
                                                     depth=depth, enable_grad=False)
    print(discriminator)
    vpsde = classifier_lib.vpsde()

    ## Loop over batches.
    print(f'Generating {num_samples} images to "{outdir}"...')
    os.makedirs(outdir, exist_ok=True)


    ## Set seed
    if do_seed:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
   
    class_labels = None
    if net.label_dim:
        class_labels = torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[batch_size], device=device)]
    if class_idx is not None:
        class_labels[:, :] = 0
        class_labels[:, class_idx] = 1

    ## Generate images.
    sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
    isir_sampler(img_batch=img_batch, device=device, save_type=save_type, vpsde=vpsde, discriminator=discriminator,
                net=net, labels=class_labels,
                outdir=outdir, class_idx=class_idx, batch_size=batch_size, 
                num_samples=num_samples, cnt_inner_steps=cnt_inner_steps, isir_batch=isir_batch, **sampler_kwargs)

#----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
#----------------------------------------------------------------------------
