import torch
import numpy as np

from losses import cosine_sim


def recall_at_k(prediction, k=1):
    recalls = 0
    for i in range(len(prediction)):
        if i in prediction[i][:k]:
            recalls += 1
    return recalls / len(prediction)


def predict(given, wanted, k):
    predictions = []
    cosine_batch_size = 5000  # Maybe make this into a separate parameter

    for i in range(0, given.shape[0], cosine_batch_size):
        similarities = cosine_sim(given[i:(i + cosine_batch_size)], wanted)

        top_k = torch.topk(similarities, k).indices.cpu().numpy()

        predictions.append(top_k)

    return np.concatenate(predictions)


def evaluate_recall_at_k(given, wanted, k):
    predictions = predict(given, wanted, k)

    return recall_at_k(predictions, k)


def evaluate(text_embeddings, image_embeddings, k=5, n=None):
    text = torch.cat(text_embeddings, dim=0)
    images = torch.cat(image_embeddings, dim=0)

    if n is not None:
        text = text[:n]
        images = images[:n]

    recall_text_retrieval = evaluate_recall_at_k(images, text, k)
    recall_image_retrieval = evaluate_recall_at_k(text, images, k)

    return recall_text_retrieval, recall_image_retrieval