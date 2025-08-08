import torch
import torch.nn.functional as F

def compute_prefill_uncertainty_metrics(model, tokenizer, prompt, device, alpha=2.0, tau=3.0, lambd=1.0):
    # Tokenize prompt
    input_tokens = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = input_tokens["input_ids"]

    # Forward pass to get prefill logits
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=False,
            output_attentions=False,
            use_cache=False,
        )
        logits = outputs.logits  # (batch_size, seq_len, vocab_size)

    # Shift tokens for next-token prediction
    probs_per_step = F.softmax(logits[:, :-1, :], dim=-1)   # (batch_size, seq_len-1, vocab_size)
    target_tokens = input_ids[:, 1:]                        # (batch_size, seq_len-1)

    # Gather token probabilities for the actual next tokens
    token_probs = probs_per_step.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)  # (batch_size, seq_len-1)
    

    # ---- Compute Metrics ----
    UMSP = compute_max_sequence_prob(token_probs)
    UPerp = compute_perplexity(token_probs)
    Entropy = compute_mean_entropy(probs_per_step.permute(1, 0, 2))  # (seq_len-1, batch_size, vocab_size)
    vocab_size = model.config.vocab_size
    URD = compute_URD(probs_per_step.permute(1, 0, 2), vocab_size, alpha)
    UFR = compute_UFR(probs_per_step.permute(1, 0, 2), vocab_size)

        # ---- Additional Metrics: TokenSAR, UPMI, UCPMI, CCP ----
    generated_tokens = input_ids[:, 1:]  # The next tokens being predicted

    TokenSAR = compute_TokenSAR_prefill(generated_tokens, tokenizer, input_ids, token_probs)
    UPMI = compute_UPMI_prefill(model, tokenizer, generated_tokens, token_probs, device)
    #UCPMI = compute_UCPMI_prefill(model, tokenizer, generated_tokens, token_probs, probs_per_step, device, tau, lambd)
    #probs_per_step_list = [p.squeeze(0) for p in probs_per_step]
    #CCP = compute_CCP_prefill(model, tokenizer, generated_tokens, probs_per_step_list, device)

    metrics =  {
        "UMSP": UMSP,
        "UPerp": UPerp,
        "Entropy": Entropy,
        "URD": URD,
        "UFR": UFR,
        "TokenSAR": TokenSAR,
        "UPMI": UPMI,
        #"UCPMI": UCPMI,
        #"CCP": CCP,
    }

    metrics = tensor_to_python(metrics)
    return metrics

def tensor_to_python(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_python(v) for v in obj]
    else:
        return obj

# ---- Existing Helper Functions (Unchanged from You) ----
def compute_max_sequence_prob(token_probs):
    log_token_probs = (token_probs + 1e-12).log()
    log_msp = log_token_probs.sum(dim=-1)
    msp = log_msp.exp()
    UMSP = 1.0 - msp
    return UMSP

def compute_perplexity(token_probs):
    L = token_probs.shape[1]
    log_msp = (token_probs + 1e-12).log().sum(dim=-1)
    uperp = (-log_msp / L).exp()
    return uperp

def compute_mean_entropy(probs):
    token_entropies = []
    for probs_step in probs:
        entropy = -(probs_step * (probs_step + 1e-12).log()).sum(dim=-1)
        token_entropies.append(entropy)
    token_entropies = torch.stack(token_entropies, dim=1)
    uht = token_entropies.mean(dim=-1)
    return uht

def compute_URD(probs_per_step, vocab_size, alpha=2.0):
    q_uniform = 1.0 / vocab_size
    urd_per_step = []
    for probs in probs_per_step:
        sum_term = (probs ** alpha).sum(dim=-1) * (q_uniform ** (1 - alpha))
        urd_step = (1.0 / (alpha - 1.0)) * torch.log(sum_term + 1e-12)
        urd_per_step.append(urd_step)
    urd_per_step = torch.stack(urd_per_step, dim=1)
    urd = urd_per_step.mean(dim=-1)
    return urd

def compute_UFR(probs_per_step, vocab_size):
    q_uniform_sqrt = (1.0 / vocab_size) ** 0.5
    ufr_per_step = []
    for probs in probs_per_step:
        probs_sqrt = probs.sqrt()
        inner_product = probs_sqrt.sum(dim=-1) * q_uniform_sqrt
        inner_product = torch.clamp(inner_product, 0.0, 1.0)
        ufr_step = (2.0 / torch.pi) * torch.arccos(inner_product + 1e-12)
        ufr_per_step.append(ufr_step)
    ufr_per_step = torch.stack(ufr_per_step, dim=1)
    ufr = ufr_per_step.mean(dim=-1)
    return ufr


from sentence_transformers import CrossEncoder
cross_encoder = CrossEncoder('cross-encoder/stsb-roberta-large')

def compute_TokenSAR_prefill(generated_tokens, tokenizer, input_tokens, token_probs):
    device = token_probs.device
    input_text = tokenizer.batch_decode(input_tokens, skip_special_tokens=True)[0]
    tokens_ids = generated_tokens.squeeze(0)
    tokens = tokenizer.convert_ids_to_tokens(tokens_ids)
    clean_tokens = [t for t in tokens if t not in tokenizer.all_special_tokens]
    if len(clean_tokens) != token_probs.shape[1]:
        token_probs = token_probs[:, :len(clean_tokens)]
    tokens = clean_tokens
    full_sequence = tokenizer.convert_tokens_to_string(tokens)
    pairs = []
    for i in range(len(tokens)):
        modified_tokens = tokens[:i] + tokens[i+1:]
        modified_sequence = tokenizer.convert_tokens_to_string(modified_tokens)
        pairs.append([full_sequence, modified_sequence])
    similarity_scores = cross_encoder.predict(pairs)
    similarity_scores = torch.tensor(similarity_scores, device=device)
    relevance = 1.0 - similarity_scores
    relevance_norm = relevance / (relevance.sum() + 1e-12)
    log_probs = (token_probs + 1e-12).log().squeeze(0)
    min_len = min(relevance_norm.shape[0], log_probs.shape[0])
    tokensar = -(relevance_norm[:min_len] * log_probs[:min_len]).sum()
    return tokensar

def compute_UPMI_prefill(model, tokenizer, generated_tokens, token_probs, device):
    batch_size, num_new_tokens = generated_tokens.shape
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
    unconditional_generated_tokens = outputs_uncond.sequences[:, empty_input["input_ids"].shape[-1]:]
    probs_per_step_uncond = [F.softmax(score, dim=-1) for score in outputs_uncond.scores]
    token_probs_uncond = [
        probs.gather(dim=-1, index=token.unsqueeze(-1)).squeeze(-1)
        for probs, token in zip(probs_per_step_uncond, unconditional_generated_tokens.T)
    ]
    token_probs_uncond = torch.stack(token_probs_uncond, dim=1)
    log_pmi = (token_probs_uncond + 1e-12).log() - (token_probs + 1e-12).log()
    upmi = log_pmi.mean(dim=-1)
    return upmi

def compute_UCPMI_prefill(model, tokenizer, generated_tokens, token_probs, probs_per_step, device, tau=3.0, lambd=1.0):
    batch_size, num_new_tokens = generated_tokens.shape
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
    unconditional_generated_tokens = outputs_uncond.sequences[:, empty_input["input_ids"].shape[-1]:]
    probs_per_step_uncond = [F.softmax(score, dim=-1) for score in outputs_uncond.scores]
    token_probs_uncond = [
        probs.gather(dim=-1, index=token.unsqueeze(-1)).squeeze(-1)
        for probs, token in zip(probs_per_step_uncond, unconditional_generated_tokens.T)
    ]
    token_probs_uncond = torch.stack(token_probs_uncond, dim=1)
    entropies = [
        -(probs * (probs + 1e-12).log()).sum(dim=-1)
        for probs in probs_per_step
    ]
    entropies = torch.stack(entropies, dim=1)
    log_p_cond = (token_probs + 1e-12).log()
    log_p_uncond = (token_probs_uncond + 1e-12).log()
    mean_log_p_cond = -log_p_cond.mean(dim=-1)
    mask = (entropies >= tau).float()
    sum_log_p_uncond = (mask * log_p_uncond).sum(dim=-1)
    num_tokens_per_batch = torch.tensor([num_new_tokens], device=device, dtype=torch.float32)
    ucpmi = mean_log_p_cond + (lambd / num_tokens_per_batch) * sum_log_p_uncond
    return ucpmi

def compute_CCP_prefill(model, tokenizer, generated_tokens, probs_per_step, device):
    tokens = tokenizer.convert_ids_to_tokens(generated_tokens.squeeze(0), skip_special_tokens=True)
    original_sequence = tokenizer.convert_tokens_to_string(tokens)
    perturbed_sequences = []

    for idx, (token, probs) in enumerate(zip(tokens, probs_per_step)):
        probs = probs.squeeze()  # Ensure shape (vocab_size,)
        if probs.dim() != 1:
            raise RuntimeError(f"Expected 1D probs but got {probs.shape}")

        topk = torch.topk(probs, k=2, dim=-1)
        top1_id = topk.indices[0].item()
        top2_id = topk.indices[1].item()

        orig_id = tokenizer.convert_tokens_to_ids(token)
        alternative_id = top1_id if top1_id != orig_id else top2_id
        alternative_token = tokenizer.convert_ids_to_tokens([alternative_id])[0]

        perturbed_tokens = tokens[:idx] + [alternative_token] + tokens[idx+1:]
        perturbed_sequence = tokenizer.convert_tokens_to_string(perturbed_tokens)
        perturbed_sequences.append([original_sequence, perturbed_sequence])

    similarity_scores = cross_encoder.predict(perturbed_sequences)
    similarity_scores = torch.tensor(similarity_scores, device=device)
    ccp = (1.0 - similarity_scores).mean().unsqueeze(0)
    return ccp
