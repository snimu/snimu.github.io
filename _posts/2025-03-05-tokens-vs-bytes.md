---
layout: post
title: "Tokens vs. Bytes"
date: 2025-03-07
comments: true
---

Compared to bytes, tokens have two **advantages**: 1) They lead to shorter sequence lengths; 2) Their embeddings contain trainset-wide statistics on the specific combination of bytes that they consist of. They also have two **disadvantages**: 1) They are poorly legible; 2) They encourage memorization.

## Advantages

### Tokens lead to shorter sequence lengths

This point is obvious, and the main purpose of tokens. If you have a larger vocabulary, you are more likely to stumble upon tokens containing more bytes, which leads to shorter sequence lengths while covering the same information, which is very useful. Let's move on.

### Tokens contain trainset-wide statistics

What is my evidence for this claim?

The first clue is that, beyond shortening the sequence length, having a larger vocabulary improves LLM eval scores. This must be due to the additional parameters in some way, but there is something strange about that: Most tokens can be split up into other tokens (ultimately, bytes), and are therefore redundant. So why do their embeddings, which are placed before the rest of the model and thus don't interact with the other tokens in any sequence, lead to improved LLM performance? The only answer I can think of is that during training, they *do* interact with the embeddings of other tokens, via the backward pass. ["Embeddings are in the middle of the model"](https://snimu.github.io/2024/12/24/embeddings-are-in-the-middle-of-the-model.html), and thus the embedding layer can learn global, trainset-wide statistics about the specific combination of bytes that each token consists of, relative to the other combinations of bytes (tokens) in the same sequence. In a sense, [it does statically what batch-norm does dynamically](https://snimu.github.io/2024/11/14/tokenization-and-batchnorm.html).

A second piece of evidence is that the [Byte Latent Transformer](https://arxiv.org/abs/2412.09871) replicates this approach by dynamically adding n-gram embeddings to each byte embedding (see [my article on the Byte Latent Transformer](https://snimu.github.io/2024/12/31/on-the-byte-latent-transformer.html)). To be precise: for every possible n-gram (for several values of n), they create an entry in the vocabulary, and for every byte, they add the embedding of the n-gram ending in the byte to the embedding of the byte itself. This is obviously extremely similar to tokens, except in a more sliding-window fashion. According to the authors themselves, this is crucial for the model to perform as well as it does.

## Disadvantages

### Tokens are poorly legible

What do I mean by that? An LLM cannot inherently tell how a token is spelled from the embedding of the token itself, because that doesn't give access to the bytes that make up the token. Instead, it has to memorize what bytes the token consists of.

While LLMs can and do do this, it means that for many tasks that require a byte-level view at the inputs, the models must learn to internally deconstruct each token into its parts, and then apply some transformation to those. That wastes model capacity during inference, requires update steps during training which otherwise could have been used to learn different things, and is still error-prone. A non-exhaustive list of such tasks is:

- Math: doing any kind of math is extremely difficult if you don't have each and every digit in each of the numbers you're working with available at the input. Tokenizers often group three digits into one token. To perform many mathematical algorithms&mdash;like simple addition with a carry term&mdash;this requires models to internally recall the digits of the numbers, and *then* perform the addition, which is wasteful and error-prone.
- Character-level tasks like those from [the CUTE benchmark](https://arxiv.org/abs/2409.15452): "Spelling", "Inverse Spelling", "Contains", "Similarity", "Insertion", "Deletion", "Subsitution", and "Swapping".
- Rhyming and poetry. While the exact pronounciation of words cannot always be known even given their spelling (especially in English), knowing the spelling helps a lot. Again, tokens require memorizing the bytes of the words to achieve the same as a byte-level model.

This is evidenced by the Byte Latent Transformer, where working in the space of bytes improves performance on the CUTE benchmark. It is also evidenced by the Mixture of Tokenizers (MoT), a model that uses both bytes and tokens. In my experiments on forward addition with a small transformer, the MoT performed significantly better than the token-based models, as seen [here](https://snimu.github.io/2025/01/28/mixture-of-tokenizers-math.html); at least on equations that are long and thus diverse enough for memorization to be impossible.

### Tokens encourage memorization

Above, I've listed many tasks for which token-based models have to memorize the bytes in the tokens. On the one hand, this makes those tasks harder even if the bytes are successfully memorized, because the model has to recall the bytes internally for the tasks. On the other hand, I posit that it also encourages the models to learn by memorization.

The reason why I believe this is that if a model works with tokens, it *must* first memorize which tokens mean the same thing; "t" "o" "k" "e" "n" is the same as "token". This is because if the same text is written slightly differently, the same words can be tokenized differently. If the model is forced to memorize the tokens already, it will be more likely to also memorize other patterns.

In other words: I believe that what the model learns early in training strongly determines the entire training run. Therefore, encouraging memorization early on is like consciously picking an anti-lottery-ticket initialization at the start of training!

And yes, of course this isn't the main reason why models memorize instead of generalize, but I suspect that it plays a small part.

While this disadvantage is pretty speculative, evidence for it comes, again, from my MoT work [here](https://snimu.github.io/2025/01/28/mixture-of-tokenizers-math.html). For very short equations, where every equation is seen during training, the token-based model outperforms the MoT. But in regimes where memorization is not possible, the byte-based MoT has several times higher answer accuracy than the baseline, while having a similar training loss. And before you say that this is due to added parameters: I've removed a full transformer layer (attention + MLP) from the MoT, so that it always has fewer parameters than the baseline. In other words, parameter counts might explain why the baseline is better at memorization, but not why the MoT is better at generalization.

## Citation

```bibtex
@article{snimu2025tokensvsbytes,
  title={Tokens vs. Bytes},
  author={Sebastian Nicolas Müller},
  year={2025},
  month={3},
  url={https://snimu.github.io/2025/03/07/tokens-vs-bytes.html}
}
```
