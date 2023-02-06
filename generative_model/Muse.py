import torch
from torch import nn
import torch.nn.functional as F
from muse_maskgit_pytorch import VQGanVAE, VQGanVAETrainer, MaskGit, MaskGitTransformer


from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP

def print_gpu() :
    print("enable devices")
    for i in range(torch.cuda.device_count()):
       print(torch.cuda.get_device_properties(i).name)


IMAGE_PATH = 'input'

if __name__ == '__main__':
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    vae=VQGanVAE(
        dim=256,
        vq_codebook_size=512
    )

    trainer = DP(VQGanVAETrainer(
        vae=vae,
        image_size=128,
        # you may want to start with small images, and then curriculum learn to larger ones, but because the vae is all convolution, it should generalize to 512 (as in paper) without training on it
        folder=IMAGE_PATH,
        batch_size=1,
        grad_accum_every=8,
        num_train_steps=50000
    ), device_ids = [0,1])



    print("number of parameter" ,sum([p.numel() for p in trainer.parameters() if p.requires_grad]))

    trainer.train()


    ##############


    # first instantiate your vae
    vae = VQGanVAE(
        dim=256,
        vq_codebook_size=512
    ).cuda()

    vae.load('/path/to/vae.pt')  # you will want to load the exponentially moving averaged VAE

    # then you plug the vae and transformer into your MaskGit as so

    # (1) create your transformer / attention network

    transformer = MaskGitTransformer(
        num_tokens=512,  # must be same as codebook size above
        seq_len=256,  # must be equivalent to fmap_size ** 2 in vae
        dim=512,  # model dimension
        depth=8,  # depth
        dim_head=64,  # attention head dimension
        heads=8,  # attention heads,
        ff_mult=4,  # feedforward expansion factor
        t5_name='t5-small',  # name of your T5
    )

    # (2) pass your trained VAE and the base transformer to MaskGit

    base_maskgit = MaskGit(
        vae=vae,  # vqgan vae
        transformer=transformer,  # transformer
        image_size=256,  # image size
        cond_drop_prob=0.25,  # conditional dropout, for classifier free guidance
    ).cuda()

    # ready your training text and images

    texts = [
        'a child screaming at finding a worm within a half-eaten apple',
        'lizard running across the desert on two feet',
        'waking up to a psychedelic landscape',
        'seashells sparkling in the shallow waters'
    ]

    images = torch.randn(4, 3, 256, 256).cuda()

    # feed it into your maskgit instance, with return_loss set to True

    loss = base_maskgit(
        images,
        texts=texts
    )

    loss.backward()

    # do this for a long time on much data
    # then...

    images = base_maskgit.generate(texts=[
        'a whale breaching from afar',
        'young girl blowing out candles on her birthday cake',
        'fireworks with blue and green sparkles'
    ], cond_scale=3.)  # conditioning scale for classifier free guidance

    images.shape  # (3, 3, 256, 256