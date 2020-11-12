import shutil

import torch
import torchvision.transforms as transforms
import torch.distributed as dist


def get_image_transforms(input_size):
    return transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomCrop((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def load_checkpoint(device, filename='checkpoint.pth.tar'):
    return torch.load(filename, map_location=device)


def save_main_model(epoch, combined_model, best_loss, optimizer, is_best):
    state = {
        'epoch': epoch + 1,
        'state_dict': combined_model.state_dict(),
        'best_recall': best_loss,
        'optimizer': optimizer.state_dict(),
    }

    save_checkpoint(state, is_best)


def load_main_model_for_inference(combined_model, device, filename='model_best.pth.tar'):
    state = torch.load(filename, map_location=device)

    combined_model.load_state_dict(state['state_dict'])

    return combined_model


def l2norm(X, eps=1e-7):
    """L2-normalize columns of X"""
    norm = (torch.pow(X, 2).sum(dim=1, keepdim=True) + eps).sqrt()
    X = torch.div(X, norm + eps)
    return X

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()
