import torch
from options.test_options import TestOptions
from models import create_model


class PrePostModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def forward(self, x):
        x = x.float()

        x.div_(255.0)

        x.add_(-0.5)
        x.div_(0.5)

        y = self.model(x)

        out = ((y * 0.5 + 0.5) * 255.0).round().to(torch.uint8)

        return out


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    if opt.eval:
        model.eval()

    if len(opt.gpu_ids) == 0:
        pre_post_model = PrePostModel(model.netG)
        filename = f"{opt.name}_cpu.pt"
    elif len(opt.gpu_ids) == 1:
        pre_post_model = PrePostModel(model.netG.module)
        filename = f"{opt.name}_cuda.pt"
    else:
        raise NotImplementedError

    x = torch.randint(0, 255, (1, 3, 256, 256), dtype=torch.uint8, device=model.device)

    traced_model = torch.jit.trace(pre_post_model, x)
    traced_model.save(filename)
