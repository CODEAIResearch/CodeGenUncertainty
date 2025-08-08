from lib2to3.pgen2 import token
import torch
import torch.nn.functional as F
import math


def tensor_to_python(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_python(v) for v in obj]
    else:
        return obj


def compute_max_sequence_prob(token_probs):
    log_token_probs = (token_probs).log()
    log_msp = log_token_probs.sum(dim=-1)  # (batch_size,)
    msp = log_msp.exp()  # back to normal probability

    UMSP = 1.0 - msp

    return UMSP

def compute_perpexity(token_probs):
    L = token_probs.shape[1]  # number of generated tokens
    log_msp = (token_probs + 1e-12).log().sum(dim=-1)
    # Length-normalized sequence probability (optional, if you want to report it)
    p_bar = (log_msp / L).exp()  # shape: (batch_size,)

    # Perplexity (UPerp)
    uperp = (-log_msp / L).exp()  # shape: (batch_size,)

    return uperp

def compute_mean_entropy(probs):
    token_entropies = []

    for probs in probs:
        # Entropy per token: -sum P log P
        entropy = -(probs * (probs + 1e-12).log()).sum(dim=-1)  # (batch_size,)
        token_entropies.append(entropy)

    # Stack to shape: (batch_size, num_new_tokens)
    token_entropies = torch.stack(token_entropies, dim=1)

    # Compute Mean Token Entropy (UHT)
    uht = token_entropies.mean(dim=-1)  # (batch_size,)

    return uht


from sentence_transformers import CrossEncoder
cross_encoder = CrossEncoder('cross-encoder/stsb-roberta-large')

def compute_TokenSAR(generated_tokens, tokenizer, input_tokens, token_probs):
    """
    Compute TokenSAR (Duan et al., 2024) robustly without asserts.
    Automatically aligns tokens and probabilities.

    Args:
        generated_tokens: (batch_size, num_new_tokens) tensor of token IDs.
        tokenizer: HuggingFace tokenizer.
        input_tokens: (batch_size, seq_len) tensor of input token IDs.
        token_probs: (batch_size, num_new_tokens) tensor of probabilities per generated token.

    Returns:
        tokensar: (batch_size,) tensor of TokenSAR scores.
    """
    device = token_probs.device
    cross_encoder = CrossEncoder('cross-encoder/stsb-roberta-large')

    # Assume batch_size == 1
    input_text = tokenizer.batch_decode(input_tokens, skip_special_tokens=True)[0]
    tokens_ids = generated_tokens.squeeze(0)

    # Convert IDs to tokens (no skip_special_tokens to preserve alignment)
    tokens = tokenizer.convert_ids_to_tokens(tokens_ids)

    # Manually remove any padding or special tokens to keep things clean (optional)
    clean_tokens = [t for t in tokens if t not in tokenizer.all_special_tokens]

    # If removing special tokens changed the count, we slice token_probs too
    if len(clean_tokens) != token_probs.shape[1]:
        token_probs = token_probs[:, :len(clean_tokens)]
    tokens = clean_tokens

    # Reconstruct the full sequence
    full_sequence = tokenizer.convert_tokens_to_string(tokens)

    # Build leave-one-out pairs
    pairs = []
    for i in range(len(tokens)):
        modified_tokens = tokens[:i] + tokens[i+1:]
        modified_sequence = tokenizer.convert_tokens_to_string(modified_tokens)
        pairs.append([full_sequence, modified_sequence])

    # Compute Cross-Encoder similarities (0-1)
    similarity_scores = cross_encoder.predict(pairs)
    similarity_scores = torch.tensor(similarity_scores, device=device)

    # Relevance = 1 - similarity
    relevance = 1.0 - similarity_scores
    relevance_norm = relevance / (relevance.sum() + 1e-12)

    # Log-probabilities
    log_probs = (token_probs + 1e-12).log().squeeze(0)

    # Ensure shapes align
    if relevance_norm.shape[0] != log_probs.shape[0]:
        min_len = min(relevance_norm.shape[0], log_probs.shape[0])
        relevance_norm = relevance_norm[:min_len]
        log_probs = log_probs[:min_len]

    # TokenSAR computation
    tokensar = -(relevance_norm * log_probs).sum()

    return tokensar


def compute_UPMI(model, tokenizer, input_tokens, generated_tokens, token_probs, device):
    """
    Compute UPMI (Pointwise Mutual Information) based uncertainty.
    UPMI compares conditional vs. unconditional token probabilities.

    Args:
        model: HuggingFace causal LM
        tokenizer: HuggingFace tokenizer
        input_tokens: (batch_size, seq_len) input prompt token IDs
        generated_tokens: (batch_size, num_new_tokens) generated token IDs (prompted)
        token_probs: (batch_size, num_new_tokens) per-token probabilities (prompted)
        device: CUDA/CPU device

    Returns:
        upmi: (batch_size,) tensor of UPMI scores
    """

    batch_size, num_new_tokens = generated_tokens.shape

    # -------------------------------
    # 1️Unconditional Generation
    # -------------------------------
    # Empty string or BOS only
    empty_input = tokenizer("", return_tensors="pt").to(device)

    outputs_uncond = model.generate(
        **empty_input,
        max_new_tokens=num_new_tokens,
        return_dict_in_generate=True,
        output_scores=True,
        output_hidden_states=False,
        do_sample=False,
        early_stopping=False,
        pad_token_id=tokenizer.pad_token_id,
          # Ensure padding isn't added unexpectedly
    )

    # Extract unconditional generated tokens (remove BOS if needed)
    unconditional_generated_tokens = outputs_uncond.sequences[:, empty_input["input_ids"].shape[-1]:]

    # -------------------------------
    # 2️ Compute Unconditional Token Probs
    # -------------------------------
    probs_per_step_uncond = [F.softmax(score, dim=-1) for score in outputs_uncond.scores]

    token_probs_uncond = [
        probs.gather(dim=-1, index=token.unsqueeze(-1)).squeeze(-1)
        for probs, token in zip(probs_per_step_uncond, unconditional_generated_tokens.T)
    ]
    token_probs_uncond = torch.stack(token_probs_uncond, dim=1)  # (batch_size, num_new_tokens)

    # -------------------------------
    # 3️ Compute UPMI
    # -------------------------------
    # Ensure shapes match
    assert token_probs.shape == token_probs_uncond.shape, \
        f"Mismatched shapes: conditional {token_probs.shape}, unconditional {token_probs_uncond.shape}"

    # log P(y_l | y_<l>) - log P(y_l | y_<l>, x)
    log_pmi = (token_probs_uncond + 1e-12).log() - (token_probs + 1e-12).log()

    # Average over tokens (length-normalized)
    upmi = log_pmi.mean(dim=-1)  # (batch_size,)

    return upmi


def compute_UCPMI(model, tokenizer, input_tokens, generated_tokens, token_probs, probs_per_step, device, tau=3.0, lambd=1.0):
    """
    Compute UCPMI (Conditional PMI) uncertainty score.
    Applies entropy-based selection as described in the metric.

    Args:
        model: HuggingFace causal LM
        tokenizer: HuggingFace tokenizer
        input_tokens: (batch_size, seq_len) input prompt token IDs
        generated_tokens: (batch_size, num_new_tokens) generated token IDs (prompted)
        token_probs: (batch_size, num_new_tokens) per-token probabilities (prompted)
        probs_per_step: list of (batch_size, vocab_size) per timestep probabilities (prompted)
        device: CUDA/CPU device
        tau: Entropy threshold τ
        lambd: Scaling parameter λ

    Returns:
        ucpmi: (batch_size,) tensor of UCPMI scores
    """

    batch_size, num_new_tokens = generated_tokens.shape

    # ------------------------------------
    # 1️ Unconditional Generation
    # ------------------------------------
    empty_input = tokenizer("", return_tensors="pt").to(device)
    outputs_uncond = model.generate(
        **empty_input,
        max_new_tokens=num_new_tokens,
        return_dict_in_generate=True,
        output_scores=True,
        output_hidden_states=False,
        do_sample=False,
        early_stopping=False,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Extract unconditional token probabilities
    unconditional_generated_tokens = outputs_uncond.sequences[:, empty_input["input_ids"].shape[-1]:]
    probs_per_step_uncond = [F.softmax(score, dim=-1) for score in outputs_uncond.scores]

    token_probs_uncond = [
        probs.gather(dim=-1, index=token.unsqueeze(-1)).squeeze(-1)
        for probs, token in zip(probs_per_step_uncond, unconditional_generated_tokens.T)
    ]
    token_probs_uncond = torch.stack(token_probs_uncond, dim=1)  # (batch_size, num_new_tokens)

    # ------------------------------------
    # 2️ Entropy of Prompted Distributions
    # ------------------------------------
    # probs_per_step: list of (batch_size, vocab_size)
    entropies = [
        -(probs * (probs + 1e-12).log()).sum(dim=-1)  # (batch_size,)
        for probs in probs_per_step
    ]
    entropies = torch.stack(entropies, dim=1)  # (batch_size, num_new_tokens)

    # ------------------------------------
    # 3️ Compute UCPMI
    # ------------------------------------
    log_p_cond = (token_probs + 1e-12).log()  # (batch_size, num_new_tokens)
    log_p_uncond = (token_probs_uncond + 1e-12).log()  # (batch_size, num_new_tokens)

    # - (1/L) * sum log P(yl | x)
    mean_log_p_cond = -log_p_cond.mean(dim=-1)  # (batch_size,)

    # (λ / L) * sum_{entropy >= τ} log P(yl | uncond)
    mask = (entropies >= tau).float()  # (batch_size, num_new_tokens)
    sum_log_p_uncond = (mask * log_p_uncond).sum(dim=-1)
    num_tokens_per_batch = torch.tensor([num_new_tokens], device=device, dtype=torch.float32)

    ucpmi = mean_log_p_cond + (lambd / num_tokens_per_batch) * sum_log_p_uncond

    return ucpmi


def compute_URD(probs_per_step, vocab_size, alpha=2.0):
    """
    Compute URD (Rényi Divergence from Uniform) uncertainty score.

    Args:
        probs_per_step: list of (batch_size, vocab_size) probabilities per generation step (prompted)
        vocab_size: Size of vocabulary N
        alpha: Rényi divergence order (> 0)

    Returns:
        urd: (batch_size,) tensor of URD scores
    """

    batch_size = probs_per_step[0].shape[0]
    num_new_tokens = len(probs_per_step)
    q_uniform = 1.0 / vocab_size  # q_i = 1 / N

    urd_per_step = []

    for probs in probs_per_step:
        # (batch_size, vocab_size)
        # Compute sum P_i^alpha * (1/N)^{1-alpha}
        sum_term = (probs ** alpha).sum(dim=-1) * (q_uniform ** (1 - alpha))  # (batch_size,)
        urd_step = (1.0 / (alpha - 1.0)) * torch.log(sum_term + 1e-12)  # (batch_size,)
        urd_per_step.append(urd_step)

    urd_per_step = torch.stack(urd_per_step, dim=1)  # (batch_size, num_new_tokens)
    urd = urd_per_step.mean(dim=-1)  # (batch_size,)

    return urd

def compute_UFR(probs_per_step, vocab_size):
    """
    Compute UFR (Fisher-Rao Distance from Uniform) uncertainty score.

    Args:
        probs_per_step: list of (batch_size, vocab_size) probabilities per generation step (prompted)
        vocab_size: Size of vocabulary N

    Returns:
        ufr: (batch_size,) tensor of UFR scores
    """

    batch_size = probs_per_step[0].shape[0]
    num_new_tokens = len(probs_per_step)

    # Uniform distribution sqrt(q_i)
    q_uniform_sqrt = (1.0 / vocab_size) ** 0.5

    ufr_per_step = []

    for probs in probs_per_step:
        # (batch_size, vocab_size)
        probs_sqrt = probs.sqrt()  # (batch_size, vocab_size)
        inner_product = probs_sqrt.sum(dim=-1) * q_uniform_sqrt  # (batch_size,)
        # Clamp to [0, 1] to avoid numerical issues
        inner_product = torch.clamp(inner_product, 0.0, 1.0)
        ufr_step = (2.0 / torch.pi) * torch.arccos(inner_product + 1e-12)  # (batch_size,)
        ufr_per_step.append(ufr_step)

    ufr_per_step = torch.stack(ufr_per_step, dim=1)  # (batch_size, num_new_tokens)
    ufr = ufr_per_step.mean(dim=-1)  # (batch_size,)

    return ufr

def compute_CCP(model, tokenizer, generated_tokens, probs_per_step, device):
    """
    Compute Claim-Conditioned Probability (CCP) via token-level perturbations and semantic similarity.

    Args:
        model: HuggingFace causal LM
        tokenizer: HuggingFace tokenizer
        generated_tokens: (batch_size, num_new_tokens)
        probs_per_step: list of (batch_size, vocab_size) probabilities
        device: CUDA/CPU

    Returns:
        ccp: (batch_size,) tensor of CCP scores
    """
    cross_encoder = CrossEncoder('cross-encoder/stsb-roberta-large')

    # Assume batch_size = 1
    tokens = tokenizer.convert_ids_to_tokens(generated_tokens.squeeze(0), skip_special_tokens=True)
    original_sequence = tokenizer.convert_tokens_to_string(tokens)

    perturbed_sequences = []

    for idx, (token, probs) in enumerate(zip(tokens, probs_per_step)):
        # Choose an alternative token (e.g., second most likely token)
        topk = torch.topk(probs.squeeze(0), k=2, dim=-1)
        # Skip if top-1 == original token; otherwise pick top-2
        orig_id = tokenizer.convert_tokens_to_ids(token)
        alternative_id = topk.indices[0].item() if topk.indices[0].item() != orig_id else topk.indices[1].item()
        alternative_token = tokenizer.convert_ids_to_tokens([alternative_id])[0]

        # Create perturbed sequence
        perturbed_tokens = tokens[:idx] + [alternative_token] + tokens[idx+1:]
        perturbed_sequence = tokenizer.convert_tokens_to_string(perturbed_tokens)
        perturbed_sequences.append([original_sequence, perturbed_sequence])

    # Similarity scores
    similarity_scores = cross_encoder.predict(perturbed_sequences)
    similarity_scores = torch.tensor(similarity_scores, device=device)

    # CCP: Mean of (1 - similarity)
    ccp = (1.0 - similarity_scores).mean().unsqueeze(0)  # (batch_size,)

    return ccp





#Sampling based 

def sample_sequences_with_probs(model, tokenizer, input_tokens, device, num_samples=10, max_new_tokens=768):
    """
    Samples sequences via stochastic decoding and collects token-level probabilities and log-probs.

    Args:
        model: HuggingFace causal LM
        tokenizer: HuggingFace tokenizer
        input_tokens: (batch_size, seq_len)
        device: CUDA/CPU
        num_samples: Number of samples to draw
        max_new_tokens: Max new tokens to generate

    Returns:
        all_generated_tokens: list of (batch_size, variable num_new_tokens) tensors
        all_token_probs: list of (batch_size, variable num_new_tokens) tensors
        all_log_probs: list of (batch_size,) tensors (raw sum log-probs, not normalized)
    """

    all_generated_tokens = []
    all_token_probs = []
    all_log_probs = []

    for _ in range(num_samples):
        outputs = model.generate(
            input_ids=input_tokens.to(device),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id,
        )

        generated_tokens = outputs.sequences[:, input_tokens.shape[-1]:]  # (batch_size, variable length)
        probs_per_step = [F.softmax(score, dim=-1) for score in outputs.scores]

        # Gather per-token probabilities
        token_probs = [
            probs.gather(dim=-1, index=token.unsqueeze(-1)).squeeze(-1)
            for probs, token in zip(probs_per_step, generated_tokens.T)
        ]
        token_probs = torch.stack(token_probs, dim=1)  # (batch_size, num_new_tokens)

        # Log sequence probability (no normalization here; leave to UHS step)
        log_p = (token_probs + 1e-12).log().sum(dim=-1)  # (batch_size,)
        log_p = log_p / generated_tokens.shape[1]

        all_generated_tokens.append(generated_tokens)
        all_token_probs.append(token_probs)
        all_log_probs.append(log_p)

    # Return lists, not stacked tensors
    return all_generated_tokens, all_token_probs, all_log_probs


def compute_USE(all_generated_tokens, all_log_probs, tokenizer, device, threshold=0.8):
    """
    Compute Semantic Entropy (USE) from sampled sequences using CrossEncoder.

    Args:
        all_generated_tokens: list of (batch_size, variable num_new_tokens) tensors
        all_log_probs: list of (batch_size,) tensors (log P(y | x))
        tokenizer: HuggingFace tokenizer
        device: CUDA/CPU
        threshold: Similarity threshold for semantic clustering (0-1)

    Returns:
        use: (batch_size,) tensor of USE scores
    """

    cross_encoder = CrossEncoder('cross-encoder/stsb-roberta-large', device=device)

    num_samples = len(all_generated_tokens)
    batch_size = all_generated_tokens[0].shape[0]

    all_decoded = []
    all_log_probs_tensor = torch.stack(all_log_probs, dim=0).T  # (batch_size, num_samples)

    # Decode to text (list of lists)
    for b in range(batch_size):
        texts = []
        for tokens in all_generated_tokens:
            decoded = tokenizer.decode(tokens[b], skip_special_tokens=True)
            texts.append(decoded)
        all_decoded.append(texts)

    use_scores = []

    for b in range(batch_size):
        sequences = all_decoded[b]
        log_probs = all_log_probs_tensor[b]  # (num_samples,)

        # Build all pairs (i < j)
        pairs = [(sequences[i], sequences[j]) for i in range(num_samples) for j in range(i + 1, num_samples)]
        similarities = cross_encoder.predict(pairs)

        # Build clusters via simple greedy thresholding
        clusters = []
        assigned = set()

        for i in range(num_samples):
            if i in assigned:
                continue
            cluster = [i]
            for j in range(num_samples):
                if i != j and (i < j):
                    idx = i * num_samples - i * (i + 1) // 2 + (j - i - 1)
                    if similarities[idx] >= threshold:
                        cluster.append(j)
                        assigned.add(j)
            assigned.add(i)
            clusters.append(cluster)

        # Compute USE
        probs = log_probs.exp()  # (num_samples,)
        use = 0.0

        for cluster in clusters:
            p_cluster = probs[cluster].sum()
            weight = len(cluster) / num_samples
            use += weight * torch.log(p_cluster + 1e-12)

        use = -use
        use_scores.append(use)

    use_scores = torch.stack(use_scores, dim=0)  # (batch_size,)

    return use_scores


def compute_USentSAR(all_generated_tokens, all_log_probs, tokenizer, device, temperature=1.0):
    """
    Compute SentenceSAR (USentSAR) using CrossEncoder similarity and sequence probabilities.

    Args:
        all_generated_tokens: list of (batch_size, variable num_new_tokens) tensors
        all_log_probs: list of (batch_size,) tensors (log P(y | x))
        tokenizer: HuggingFace tokenizer
        device: CUDA/CPU
        temperature: Temperature parameter t in SentenceSAR

    Returns:
        usentsar: (batch_size,) tensor of USentSAR scores
    """

    cross_encoder = CrossEncoder('cross-encoder/stsb-roberta-large', device=device)

    num_samples = len(all_generated_tokens)
    batch_size = all_generated_tokens[0].shape[0]

    all_decoded = []
    all_log_probs_tensor = torch.stack(all_log_probs, dim=0).T  # (batch_size, num_samples)

    # Decode to text (list of lists)
    for b in range(batch_size):
        texts = []
        for tokens in all_generated_tokens:
            decoded = tokenizer.decode(tokens[b], skip_special_tokens=True)
            texts.append(decoded)
        all_decoded.append(texts)

    usentsar_scores = []

    for b in range(batch_size):
        sequences = all_decoded[b]
        log_probs = all_log_probs_tensor[b]  # (num_samples,)
        probs = log_probs.exp()  # (num_samples,)

        # Pairwise similarities using CrossEncoder
        pairs = [(sequences[i], sequences[j]) for i in range(num_samples) for j in range(num_samples)]
        similarities = cross_encoder.predict(pairs)
        similarities = torch.tensor(similarities, device=device).view(num_samples, num_samples)

        # Compute RS for each sequence
        rs_values = []
        for j in range(num_samples):
            mask = torch.ones(num_samples, device=device, dtype=torch.bool)
            mask[j] = False  # Exclude j==j
            rs_j = (similarities[j][mask] * probs[mask]).sum()
            rs_values.append(rs_j)

        rs_values = torch.stack(rs_values)  # (num_samples,)

        # USentSAR computation
        adjusted_probs = probs + (rs_values / temperature)
        usentsar = -(adjusted_probs + 1e-12).log().mean()
        usentsar_scores.append(usentsar)

    usentsar_scores = torch.stack(usentsar_scores, dim=0)  # (batch_size,)

    return usentsar_scores


#blackbox 

from transformers import RobertaForSequenceClassification, RobertaTokenizer
# Load the pre-trained DeBERTa model and tokenizer for NLI
#nli_model = DebertaForSequenceClassification.from_pretrained("microsoft/deberta-v3-large")
nli_tokenizer = RobertaForSequenceClassification.from_pretrained("roberta-large-mnli")
nli_model = RobertaTokenizer.from_pretrained("roberta-large-mnli")


def nli_similarity(seq1, seq2):
    """Computes similarity using NLI (entailment vs contradiction)."""
    # Tokenize the sequences
    inputs = nli_tokenizer([seq1, seq2], padding=True, truncation=True, return_tensors="pt")

    # Get NLI logits (entailment and contradiction)
    outputs = nli_model(**inputs)
    entailment_prob = F.softmax(outputs.logits, dim=-1)[0][0].item()  # Entailment score
    contradiction_prob = F.softmax(outputs.logits, dim=-1)[0][1].item()  # Contradiction score
    
    # Return similarity based on entailment score (or 1 - contradiction for contrast)
    return entailment_prob

def count_semantic_sets(responses, similarity_func):
    """Counts the number of semantic sets by comparing pairs of responses."""
    num_responses = len(responses)
    semantic_sets = []

    # Compare each pair of responses
    for i in range(num_responses):
        for j in range(i + 1, num_responses):
            similarity = similarity_func(responses[i], responses[j])
            
            if similarity > 0.7:  # If similarity threshold is met
            # Check if either response already belongs to a semantic set
                found = False
                for semantic_set in semantic_sets:
                    # If either response is in an existing set, merge the responses
                    if responses[i] in semantic_set or responses[j] in semantic_set:
                        semantic_set.add(responses[i])  # Add response[i] to the set
                        semantic_set.add(responses[j])  # Add response[j] to the set
                        found = True
                        break  # Exit loop once a match is found

                if not found:
                # If no matching set was found, create a new set
                    semantic_sets.append({responses[i], responses[j]})
            else:
                # Treat as distinct sets, only append a new set with the two responses
                semantic_sets.append({responses[i], responses[j]})

    # Return the number of distinct semantic sets
    return len(semantic_sets)


import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Assuming `sample_sequences_with_probs` is already defined

def compute_similarity_matrix(generated_tokens, tokenizer, model, device):
    """
    Computes a similarity matrix based on the cosine similarity of generated sequences.
    This function computes the token embeddings first, then calculates similarity.
    For causal language models like GPT-2, we use the logits as embeddings.
    """
    num_samples = len(generated_tokens)
    similarity_matrix = np.zeros((num_samples, num_samples))

    # Get token embeddings from the model for each generated sequence
    embeddings = []
    for i in range(num_samples):
        # Decode token IDs to text (ensure you're passing a tensor or list of IDs)
        generated_token_ids = generated_tokens[i].squeeze(0).cpu().numpy()  # Flatten to 1D array if needed
        
        # Decode token IDs to text
        decoded_response = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

        # Tokenize and get embeddings
        inputs = tokenizer(decoded_response, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=False, output_hidden_states=False)

        # For Causal Language Models, logits are returned. We use the logits as embeddings.
        logits = outputs.logits.squeeze(0).to(torch.float32).cpu().numpy()  # Convert to float32 before moving to CPU
        embeddings.append(logits.mean(axis=0))  # Average over the sequence length

    # Now compute pairwise cosine similarity between embeddings
    for i in range(num_samples):
        for j in range(i, num_samples):
            # Compute similarity between two generated sequences based on logits
            similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity  # Since the matrix is symmetric

    return similarity_matrix

def compute_laplacian(similarity_matrix):
    """
    Computes the Graph Laplacian L = I - D^(-1/2) S D^(-1/2) from the similarity matrix.
    """
    # Degree matrix D (diagonal matrix)
    D = np.diag(np.sum(similarity_matrix, axis=1))

    # Normalize similarity matrix by D^(-1/2)
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))

    # Compute the Laplacian matrix
    L = np.eye(len(similarity_matrix)) - D_inv_sqrt @ similarity_matrix @ D_inv_sqrt
    return L

def compute_uncertainty(L):
    """
    Computes the uncertainty measure U_Eig as the sum of the eigenvalues of the Laplacian.
    """
    # Eigenvalues of the Laplacian
    eigenvalues = np.linalg.eigvals(L)

    # Uncertainty measure: sum of eigenvalues (1 - eigenvalue) if > 0
    uncertainty = np.sum(np.maximum(0, 1 - eigenvalues))
    return uncertainty


from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize


def compute_lexical_similarity(generated_tokens, tokenizer, model, device, use_rouge=True):
    """
    Computes lexical similarity based on ROUGE or BLEU scores between pairs of generated responses.
    The function calculates the average score between all pairs.
    
    Args:
    - generated_tokens: List of generated token sequences.
    - tokenizer: HuggingFace tokenizer used to decode tokens.
    - model: HuggingFace model used for generating sequences.
    - device: Device (CUDA or CPU).
    - use_rouge: Boolean flag to use ROUGE or BLEU. Defaults to True for ROUGE.
    
    Returns:
    - average_lexical_similarity: The average lexical similarity score.
    """
    # Get the generated sequences
    decoded_responses = [tokenizer.decode(tokens[0], skip_special_tokens=True) for tokens in generated_tokens]
    
    total_similarity = 0
    num_pairs = 0

    # Initialize ROUGE scorer
    if use_rouge:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Iterate over all pairs of generated responses
    for i in range(len(decoded_responses)):
        for j in range(i + 1, len(decoded_responses)):
            response_1 = decoded_responses[i]
            response_2 = decoded_responses[j]
            
            if use_rouge:
                # Compute ROUGE scores between two responses
                scores = scorer.score(response_1, response_2)
                total_similarity += (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
            else:
                # Compute BLEU score between two responses
                reference = word_tokenize(response_1.lower())
                candidate = word_tokenize(response_2.lower())
                total_similarity += sentence_bleu([reference], candidate)
                
            num_pairs += 1

    # Calculate the average similarity score
    average_lexical_similarity = total_similarity / num_pairs
    return average_lexical_similarity



def compute_BB_SE(all_generated_tokens, all_log_probs, tokenizer, device, temperature=0.5):
    """
    Compute BB Semantic Entropy using CrossEncoder similarity.

    Args:
        all_generated_tokens: list of (batch_size, variable num_new_tokens) tensors
        all_log_probs: list of (batch_size,) tensors (log P(y | x))
        tokenizer: HuggingFace tokenizer
        device: CUDA/CPU
        temperature: Softmax temperature for weighting similarities

    Returns:
        bbse: (batch_size,) tensor of BB Semantic Entropy scores
    """

    cross_encoder = CrossEncoder('cross-encoder/stsb-roberta-large', device=device)

    num_samples = len(all_generated_tokens)
    batch_size = all_generated_tokens[0].shape[0]

    all_decoded = []
    all_log_probs_tensor = torch.stack(all_log_probs, dim=0).T  # (batch_size, num_samples)

    for b in range(batch_size):
        texts = []
        for tokens in all_generated_tokens:
            decoded = tokenizer.decode(tokens[b], skip_special_tokens=True)
            texts.append(decoded)
        all_decoded.append(texts)

    bbse_scores = []

    for b in range(batch_size):
        sequences = all_decoded[b]
        log_probs = all_log_probs_tensor[b]  # (num_samples,)

        # Build all pairs (i, j) for similarity matrix
        pairs = [(sequences[i], sequences[j]) for i in range(num_samples) for j in range(num_samples)]
        similarities = cross_encoder.predict(pairs)
        similarities = torch.tensor(similarities, device=device).view(num_samples, num_samples)  # (num_samples, num_samples)

        # Row-wise softmax to get bootstrap-like weights
        weights = torch.softmax(similarities / temperature, dim=1)  # (num_samples, num_samples)

        # Average across rows to produce final weight for each sequence
        avg_weights = weights.mean(dim=0)  # (num_samples,)

        # BBSE computation
        bbse = -(avg_weights * log_probs).sum()
        bbse_scores.append(bbse)

    bbse_scores = torch.stack(bbse_scores, dim=0)  # (batch_size,)
    return bbse_scores

def compute_label_probability(all_generated_tokens, tokenizer, device, threshold=0.8):
    """
    Compute Label Probability as fraction of dominant semantic cluster.

    Args:
        all_generated_tokens: list of (batch_size, variable num_new_tokens) tensors
        tokenizer: HuggingFace tokenizer
        device: CUDA/CPU
        threshold: Similarity threshold for clustering (0-1)

    Returns:
        label_probs: (batch_size,) tensor of dominant label probability
    """

    cross_encoder = CrossEncoder('cross-encoder/stsb-roberta-large', device=device)

    num_samples = len(all_generated_tokens)
    batch_size = all_generated_tokens[0].shape[0]

    all_decoded = []

    for b in range(batch_size):
        texts = []
        for tokens in all_generated_tokens:
            decoded = tokenizer.decode(tokens[b], skip_special_tokens=True)
            texts.append(decoded)
        all_decoded.append(texts)

    label_probs = []

    for b in range(batch_size):
        sequences = all_decoded[b]

        pairs = [(sequences[i], sequences[j]) for i in range(num_samples) for j in range(i + 1, num_samples)]
        similarities = cross_encoder.predict(pairs)
        similarities = torch.tensor(similarities, device=device)

        # Greedy clustering based on similarity threshold
        clusters = []
        assigned = set()

        for i in range(num_samples):
            if i in assigned:
                continue
            cluster = [i]
            for j in range(num_samples):
                if i != j and i < j:
                    idx = i * num_samples - i * (i + 1) // 2 + (j - i - 1)
                    if similarities[idx] >= threshold:
                        cluster.append(j)
                        assigned.add(j)
            assigned.add(i)
            clusters.append(cluster)

        # Compute Label Probability
        largest_cluster_size = max(len(cluster) for cluster in clusters)
        label_prob = largest_cluster_size / num_samples
        label_probs.append(torch.tensor(label_prob, device=device))

    label_probs = torch.stack(label_probs, dim=0)
    return label_probs



def compute_info_metrics(outputs, input_len, tokenizer, input_tokens, model, alpha=2, tau=1.5, lambd=1.0):
    #extract generated token
    generated_tokens = outputs.sequences[:, input_len:]

    #Convert logits to probabilities
    probs_per_step = [F.softmax(score, dim=-1) for score in outputs.scores]

    #Gather the probabilities assigned to the generated tokens at each step
    token_probs = [
    probs.gather(dim=-1, index=token.unsqueeze(-1)).squeeze(-1)  # (batch_size,)
    for probs, token in zip(probs_per_step, generated_tokens.T)
]
    #save each token probs
    token_probs = torch.stack(token_probs, dim=1)  # (batch_size, num_new_tokens)

    device = token_probs.device

    

    UMSP = compute_max_sequence_prob(token_probs)

    uperp = compute_perpexity(token_probs)

    entropy = compute_mean_entropy(probs_per_step)

    tokenSAR = compute_TokenSAR(generated_tokens,tokenizer,input_tokens, token_probs)

    upmi = compute_UPMI(model, tokenizer, input_tokens, generated_tokens, token_probs, device)

    ucpmi = compute_UCPMI(model, tokenizer, input_tokens, generated_tokens, token_probs, probs_per_step, device, tau=3.0, lambd=1.0)

    vocab_size=model.config.vocab_size
    urd = compute_URD(probs_per_step, vocab_size, alpha=2.0)

    ufr = compute_UFR(probs_per_step, vocab_size)

    ccp = compute_CCP(model, tokenizer, generated_tokens, probs_per_step, device)


    #sampling based 

    all_generated_tokens, all_token_probs, all_log_probs = sample_sequences_with_probs(model, tokenizer, input_tokens, device)
    #log_p = log_p / generated_tokens.shape[1]
    call_log_probs = torch.stack(all_log_probs, dim=0)
    uhs = -call_log_probs.mean(dim=0)

    use = compute_USE(all_generated_tokens, all_log_probs, tokenizer, device, threshold=0.8)

    usentsar = compute_USentSAR(all_generated_tokens, all_log_probs, tokenizer, device, temperature=1.0)


    #blackbox


    # Flatten the list of token IDs and convert to text (this is a placeholder, you'll need a method to decode tokens back to text)
    generated_sequences = [tokenizer.decode(tokens) for tokens in all_generated_tokens[0]]  # Assume batch_size = 1

    # Calculate the number of distinct semantic sets
    semantic_sets_count = count_semantic_sets(generated_sequences, nli_similarity)  # Using NLI similarity

    # Compute similarity matrix for the generated sequences
    similarity_matrix = compute_similarity_matrix(all_generated_tokens, tokenizer, model, device)
    # Compute the Laplacian matrix
    laplacian = compute_laplacian(similarity_matrix)
    # Compute the uncertainty measure
    eigen_uncertainty = compute_uncertainty(laplacian)

    lexical_similarity = compute_lexical_similarity(all_generated_tokens, tokenizer, model, device, use_rouge=True)

    bb_label_probs = compute_label_probability(all_generated_tokens, tokenizer, device, threshold=0.8)
    bbse = compute_BB_SE(all_generated_tokens, all_log_probs, tokenizer, device, temperature=0.5)

    





    
    


    metrics = {
        'MSP': UMSP,
        'UPerp': uperp,
        'CCP': ccp,
        'Entropy': entropy,
        'TokenSAR': tokenSAR,
        'UPMI': upmi,
        'UCPMI': ucpmi,
        'URD': urd,
        'UFR': ufr,
        'UHS':uhs,
        'USE': use,
        'USENTSAR':usentsar,
        'SEMANTIC_SET': semantic_sets_count,
        'EIGEN_UNCERTAINTY': eigen_uncertainty,
        'LEXICAL_SIM': lexical_similarity,
        'BB_Label_Prob': bb_label_probs,
        'BB_SEMANIC_Entropy': bbse





    }

    metrics = tensor_to_python(metrics)
    return metrics


"""def compute_info_metrics1(outputs, input_len, tokenizer, alpha=2, tau=1.5, lambd=1.0):
    scores = outputs.scores  # List[tensor] of shape [B, vocab_size]
    sequences = outputs.sequences
    gen_tokens = sequences[:, input_len:]  # Only newly generated tokens

    batch_size, L = gen_tokens.shape
    V = scores[0].shape[-1]  # vocab size

    log_probs, probs, entropy = [], [], []
    for i, s in enumerate(scores):
        lp = F.log_softmax(s, dim=-1)
        p = lp.exp()
        h = -(p * lp).sum(-1)

        log_probs.append(lp[torch.arange(batch_size), gen_tokens[:, i]])
        probs.append(p)
        entropy.append(h)

    log_probs = torch.stack(log_probs, dim=1)   # [B, L]
    probs = torch.stack(probs, dim=1)           # [B, L, V]
    entropy = torch.stack(entropy, dim=1)       # [B, L]

    # 1. Maximum Sequence Probability (MSP)
    P_yx = log_probs.sum(dim=1).exp()           # [B]
    MSP = 1 - P_yx

    # 2. Perplexity (UPerp)
    UPerp = torch.exp(-log_probs.sum(dim=1) / L)

    # 3. Length-normalized log prob (P̄)
    P_bar = torch.exp(log_probs.sum(dim=1) / L)

    # 4. Mean token entropy (UHT)
    UHT = entropy.mean(dim=1)

    # 5. TokenSAR — using dummy relevance (1)
    R = torch.ones_like(log_probs)
    R_ = R / R.sum(dim=1, keepdim=True)
    TokenSAR = -(R_ * log_probs).sum(dim=1)

    # 6. PMI — with dummy unconditional (uniform)
    log_p_uncond = torch.full_like(log_probs, -math.log(V))
    PMI = (log_p_uncond - log_probs).mean(dim=1)

    # 7. CPMI — dummy entropy threshold + uniform unconditional
    entropy_mask = (entropy >= tau).float()
    CPMI = (-log_probs.sum(1) + lambd * (log_p_uncond * entropy_mask).sum(1)) / L

    # 8. Rényi divergence (URD)
    q = torch.full((V,), 1 / V, device=probs.device)
    q = q.unsqueeze(0).unsqueeze(0)             # [1, 1, V]
    URD = ((probs ** alpha) * (q ** (1 - alpha))).sum(-1).log() / (alpha - 1)
    URD = URD.mean(dim=1)

    # 9. Fisher-Rao (UFR)
    cos = (probs.sqrt() * q.sqrt()).sum(-1).clamp(0, 1)
    UFR = (2 / math.pi) * torch.acos(cos)
    UFR = UFR.mean(dim=1)

    metrics = {
        'MSP': MSP,
        'UPerp': UPerp,
        'P_bar': P_bar,
        'UHT': UHT,
        'TokenSAR': TokenSAR,
        'PMI': PMI,
        'CPMI': CPMI,
        'URD': URD,
        'UFR': UFR
    }

    metrics = tensor_to_python(metrics)
    return metrics"""