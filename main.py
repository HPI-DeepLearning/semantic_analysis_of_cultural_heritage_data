import argparse
import copy
import os
import time

import numpy as np
import pkbar
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from coco_dataset import CoCoDataset
from losses import cosine_sim, ContrastiveLoss, MMDLoss
from semart_dataset import SemArtDataset
from combined_model import CombinedModel
from text_preprocessing import build_vocab
from utils import save_main_model, load_checkpoint, synchronize
from validation import evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco', help='Coco dataset location', default='./data/coco')
    parser.add_argument('--semart', help='SemArt dataset location', default='./data/SemArt')
    parser.add_argument('--wpi', help='WPI dataset location', default='./data/WPI_Dataset')
    parser.add_argument('--resnet', help='Either "pretrained" or the location of the resnet checkpoint',
                        default='pretrained')
    parser.add_argument('-b', '--batch-size', help='The batch size to use', default=128, type=int)
    parser.add_argument('-e', '--epochs', help='The number of epochs to train', default=30, type=int)
    parser.add_argument('-lr', '--learning-rate', help='The learning rate of both optimizers', default=2e-4, type=float)
    parser.add_argument('--mmd-weight', help='The weighting of the MMD loss', default=1, type=float)
    parser.add_argument('--log-dir', help='Where to save tensorboard logs', default=None)
    parser.add_argument('--load-model', help='path to saved model file')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--reduction",  default='sum')

    return parser.parse_args()


def train(combined_model, supervised, unsupervised, optimizer, mmd_weight):
    batches_per_epoch = len(supervised)

    kbar = pkbar.Kbar(target=batches_per_epoch, width=8)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    combined_model.train()

    images_supervised = []
    text_supervised = []

    images_unsupervised = []
    text_unsupervised = []

    running_loss = 0.0
    running_loss_supervised = 0.0
    running_loss_unsupervised = 0.0

    print("Start training")

    for indx, (supervised_inputs, unsupervised_inputs) in enumerate(zip(supervised, unsupervised)):
        img_inputs = supervised_inputs[0].to(device)
        text_inputs = supervised_inputs[1].to(device)

        unsupervised_image_inputs = unsupervised_inputs[0].to(device)
        unsupervised_text_inputs = unsupervised_inputs[1].to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            text_embeddings_supervised, image_embeddings_supervised = combined_model(text_inputs, img_inputs)

            # unsupervised output
            text_embeddings_unsupervised, image_embeddings_unsupervised = combined_model(unsupervised_text_inputs, unsupervised_image_inputs)

            supervised_loss = criterion_supervised(text_embeddings_supervised, image_embeddings_supervised)
            unsupervised_loss = criterion_unsupervised(text_embeddings_unsupervised, image_embeddings_unsupervised)

            # Scale unsupervised loss with batch size
            unsupervised_loss = unsupervised_loss * mmd_weight

            loss = supervised_loss + unsupervised_loss

            loss.backward()
            optimizer.step()

            images_supervised.append(image_embeddings_supervised.detach().clone())
            text_supervised.append(text_embeddings_supervised.detach().clone())

            images_unsupervised.append(image_embeddings_unsupervised.detach().clone())
            text_unsupervised.append(text_embeddings_unsupervised.detach().clone())

        # statistics
        running_loss += loss.item()
        running_loss_supervised += supervised_loss
        running_loss_unsupervised += unsupervised_loss

        kbar.update(indx, values=[("loss", loss), ("supervised_loss", supervised_loss),
                                  ("unsupervised_loss", unsupervised_loss)])

    recall_supervised = evaluate(text_supervised, images_supervised)
    recall_unsupervised = evaluate(text_unsupervised, images_unsupervised)

    epoch_loss = (running_loss / len(supervised), running_loss_supervised / len(supervised),
                  running_loss_unsupervised / len(supervised))

    return epoch_loss, recall_supervised, recall_unsupervised


def validate(combined_model, unsupervised_val, retrievable_items):
    batches_per_epoch = len(unsupervised_val)
    kbar = pkbar.Kbar(target=batches_per_epoch, width=8)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    combined_model.eval()

    image_embeddings = []
    text_embeddings = []

    running_loss = 0.0
    print("Start validation")

    for indx, unsupervised_inputs in enumerate(unsupervised_val):
        unsupervised_image_inputs = unsupervised_inputs[0].to(device)
        unsupervised_text_inputs = unsupervised_inputs[1].to(device)

        with torch.set_grad_enabled(False):
            text_embeddings_unsupervised, image_embeddings_unsupervised = combined_model(unsupervised_text_inputs, unsupervised_image_inputs)

            unsupervised_loss = criterion_unsupervised(text_embeddings_unsupervised, image_embeddings_unsupervised)

            image_embeddings.append(image_embeddings_unsupervised.detach().clone())
            text_embeddings.append(text_embeddings_unsupervised.detach().clone())

        # statistics
        running_loss += unsupervised_loss
        kbar.update(indx, values=[("unsupervised_loss", unsupervised_loss)])

    recalls = []

    for item_count in retrievable_items:
        recalls.append((evaluate(text_embeddings, image_embeddings, 5, item_count), item_count))

    epoch_loss = running_loss / len(unsupervised_val)

    return epoch_loss, recalls


def update_learning_rate(optimizer, initial_lr, epoch):
    if epoch >= 15:
        new_lr = initial_lr * 0.1
    else:
        new_lr = initial_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def train_model(combined_model, supervised_train, supervised_val, unsupervised_train, unsupervised_val, wpi_data,
                optimizer, best_recall=0, start_epoch=0, learning_rate=2e-4, mmd_weight=1, log_dir=None, num_epochs=25):
    writer = SummaryWriter(log_dir)

    since = time.time()
    best_wts = copy.deepcopy(combined_model.state_dict())

    if start_epoch > 15:
        update_learning_rate(optimizer, learning_rate, 15)

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        train_loss, recall_supervised, recall_unsupervised_train = train(combined_model,
                                                                         supervised_train,
                                                                         unsupervised_train,
                                                                         optimizer,
                                                                         mmd_weight)

        if args.local_rank == 0:

            print('Overall Train Loss: {:.4f}'.format(train_loss[0]))
            print('Supervised Train Loss: {:.4f}'.format(train_loss[1]))
            print('Unsupervised Train Loss: {:.4f}'.format(train_loss[2]))
            print(f'Supervised text retrieval recall @ 5: {recall_supervised[0]}')
            print(f'Supervised image retrieval recall @ 5: {recall_supervised[1]}')

            writer.add_scalar('Loss/train_overall', train_loss[0], epoch)
            writer.add_scalar('Loss/train_supervised', train_loss[1], epoch)
            writer.add_scalar('Loss/train_unsupervised', train_loss[2], epoch)
            writer.add_scalar('Text_retrieval_r@5/train_supervised', recall_supervised[0], epoch)
            writer.add_scalar('Image_retrieval_r@5/train_supervised', recall_supervised[1], epoch)
            writer.add_scalar('Text_retrieval_r@5/train_unsupervised', recall_unsupervised_train[0], epoch)
            writer.add_scalar('Image_retrieval_r@5/train_unsupervised', recall_unsupervised_train[1], epoch)

            _, recalls_supervised_val = validate(combined_model,
                                                    supervised_val,
                                                    [None])

            print(f'Supervised text retrieval recall @ 5: {recalls_supervised_val[0][0][0]}')
            print(f'Supervised image retrieval recall @ 5: {recalls_supervised_val[0][0][1]}')
            writer.add_scalar(f'Text_retrieval_r@5/val_supervised', recalls_supervised_val[0][0][0], epoch)
            writer.add_scalar(f'Image_retrieval_r@5/val_supervised', recalls_supervised_val[0][0][1], epoch)

            val_unsupervised_loss, recalls_unsupervised_val = validate(combined_model,
                                                                   unsupervised_val,
                                                                   [100, 300, 500, 1000])

            print('Unsupervised validation Loss: {:.4f}'.format(val_unsupervised_loss))
            writer.add_scalar('Loss/val_unsupervised', val_unsupervised_loss, epoch)

            mean_recall_val = 0

            for recall in recalls_unsupervised_val:
                mean_recall_val += recall[0][0]
                mean_recall_val += recall[0][1]
                print(f'Unsupervised text retrieval recall @ 5: {recall[0][0]} (n = {recall[1]})')
                print(f'Unsupervised image retrieval recall @ 5: {recall[0][1]} (n = {recall[1]})')
                writer.add_scalar(f'Text_retrieval_r@5/val_unsupervised_n_{recall[1]}', recall[0][0], epoch)
                writer.add_scalar(f'Image_retrieval_r@5/val_unsupervised_n_{recall[1]}', recall[0][1], epoch)

            _, wpi_recall = validate(combined_model, wpi_data, [100])

            print(f'WPI eval text retrieval recall @ 5: {wpi_recall[0][0][0]}')
            print(f'WPI eval image retrieval recall @ 5: {wpi_recall[0][0][1]}')
            writer.add_scalar(f'Text_retrieval_r@5/WPI_eval', wpi_recall[0][0][0], epoch)
            writer.add_scalar(f'Image_retrieval_r@5/WPI_eval', wpi_recall[0][0][1], epoch)

            update_learning_rate(optimizer, learning_rate, epoch)

            mean_recall_val = mean_recall_val / (2 * len(recalls_unsupervised_val))

            # deep copy the model
            if mean_recall_val > best_recall:
                best_recall = mean_recall_val
                best_wts = copy.deepcopy(combined_model.state_dict())

            save_main_model(epoch, combined_model, best_recall, optimizer, mean_recall_val == best_recall)

    if args.local_rank == 0:
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val recall: {:4f}'.format(best_recall))

    # load best model weights
    combined_model.load_state_dict(best_wts)

    return combined_model


if __name__ == "__main__":
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_device(args.local_rank)
    else:
        device = torch.device("cpu")

    criterion_unsupervised = MMDLoss(1)
    criterion_supervised = ContrastiveLoss(margin=0.2,
                                           measure="cosine",
                                           reduction=args.reduction,
                                           max_violation=True)

    print("initializing dataloaders")
    print('=' * 30 + '\n')

    coco_data = CoCoDataset(args.coco, 'train', 224)
    coco_val = CoCoDataset(args.coco, 'val', 224)
    semart_data = SemArtDataset(semart_root=args.semart, split="train", desired_length=len(coco_data),
                                img_input_size=224)
    semart_val = SemArtDataset(semart_root=args.semart, split="val", img_input_size=224)
    wpi_data = SemArtDataset(semart_root=args.wpi, split="val", img_input_size=224)     # WPI has same format as SemArt

    vectorizer = build_vocab(semart_data.captions + coco_data.captions)
    coco_data.set_vectorizer(vectorizer)
    coco_val.set_vectorizer(vectorizer)
    semart_data.set_vectorizer(vectorizer)
    semart_val.set_vectorizer(vectorizer)
    wpi_data.set_vectorizer(vectorizer)

    coco_dataloader = torch.utils.data.DataLoader(dataset=coco_data,batch_size=args.batch_size, num_workers=4,
                                                  drop_last=True, sampler=torch.utils.data.distributed.DistributedSampler(dataset=coco_data,num_replicas= int(os.environ["WORLD_SIZE"]), rank= args.local_rank))
    coco_val_dataloader = torch.utils.data.DataLoader(coco_val, batch_size=args.batch_size,
                                                      num_workers=4, drop_last=True)
    semart_dataloader = DataLoader(dataset=semart_data, batch_size=args.batch_size, drop_last=True, num_workers=4, sampler=torch.utils.data.distributed.DistributedSampler(dataset=semart_data,num_replicas= int(os.environ["WORLD_SIZE"]), rank= args.local_rank))
    semart_val_dataloader = DataLoader(dataset=semart_val, batch_size=args.batch_size, drop_last=False, num_workers=4)

    wpi_dataloader = DataLoader(dataset=wpi_data, batch_size=args.batch_size, drop_last=False, num_workers=4)

    if int(os.environ["WORLD_SIZE"]) > 1:
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
    print("world size: {}".format(os.environ["WORLD_SIZE"]))
    print("rank: {}".format(args.local_rank))
    synchronize()

    if int(os.environ["WORLD_SIZE"]) > 1:
        combined_model = torch.nn.parallel.DistributedDataParallel(CombinedModel(len(vectorizer.vocabulary_), device, args.resnet, l2_norm=True), device_ids=[args.local_rank], output_device=args.local_rank).cuda()
    else:
        combined_model = CombinedModel(len(vectorizer.vocabulary_), device, args.resnet, l2_norm=True).to(device)

    params = combined_model.parameters()
    optimizer = optim.Adam(params=params, lr=args.learning_rate)

    start_epoch = 0

    if args.load_model:
        print(f"loading model from file {args.load_model}")
        state = load_checkpoint(device, args.load_model)
        combined_model.load_state_dict(state['state_dict'])
        combined_model.to(device)
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = state['epoch']
        best_recall = state['best_recall']
    else:
        best_recall = 0
        start_epoch = 0

    print("start training")
    train_model(combined_model, coco_dataloader, coco_val_dataloader, semart_dataloader,
                semart_val_dataloader, wpi_dataloader,
                optimizer, best_recall, start_epoch, args.learning_rate, args.mmd_weight, args.log_dir, args.epochs)
