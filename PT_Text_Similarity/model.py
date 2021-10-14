import numpy
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

sentences22 = [
    "List sentences here",
]

def getSentenceVectors(sentences):
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

    tokens = {'input_ids': [], 'attention_mask': []}

    for sentence in sentences:
        new_tokens = tokenizer.encode_plus(sentence, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
        tokens['input_ids'].append(new_tokens['input_ids'][0])
        tokens['attention_mask'].append(new_tokens['attention_mask'][0])

    tokens['input_ids'] = torch.stack(tokens['input_ids'])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])
    outputs = model(**tokens)
    embeddings = outputs.last_hidden_state
    attention = tokens['attention_mask']
    mask = attention.unsqueeze(-1).expand(embeddings.shape).float()
    mask_embeddings = embeddings * mask
    summed = torch.sum(mask_embeddings, 1)
    counts = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = summed / counts
    mean_pooled = mean_pooled.detach().numpy()
    return mean_pooled


def findTwoMostSimilar(sentences):
    mean_pooled = getSentenceVectors(sentences)
    similarity_vector = numpy.array([], dtype=float)
    comparison_list = []
    for sentence1 in range(len(mean_pooled) - 1):
        for sentence2 in range(len(mean_pooled)):
            if(sentence1 != sentence2 and sentence2 > sentence1):
                similarity = cosine_similarity(
                    [mean_pooled[sentence1]],
                    [mean_pooled[sentence2]]
                )
                comparison_list.append([sentence1, sentence2])
                similarity_vector = numpy.append(similarity_vector, similarity)

    max_similarity_score = numpy.amax(similarity_vector)
    max_similarity_idx = similarity_vector.tolist().index(max_similarity_score)
    print(f"Sentences {comparison_list[max_similarity_idx][0] + 1} & {comparison_list[max_similarity_idx][1] + 1} are the most similar")

def similarityScore(sentence1, sentence2):
    sentences = [sentence1, sentence2]
    mean_pooled = getSentenceVectors(sentences)
    score = cosine_similarity(
        [mean_pooled[0]],
        [mean_pooled[1]]
    )
    print(f"Similarity score: {score[0][0]}")

similarityScore("Three years later, the coffin was still full of Jello", "The person box was packed with jelly many dozens of months later")


