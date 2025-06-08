---
layout: post
title: "Model stacking doesn't work"
date: 2025-06-07
---

In my article about [Model Stacking](https://snimu.github.io/2025/02/18/model-stacking.html), I proposed a method for decentralized pre-training of models.

I have since performed extensive experiments that show that it doesn't work. I present these experiments below. Note: I've obtained these results several months ago and only got around to writing them down now, so I unfortunately don't remember every nuance of my motivations, and the writing will be fairly short.

## The method

The basic insight is that tying a language head (lm-head) to the token embedding layers&mdash;a somewhat common practice for saving parameters&mdash;should force the model to work in the same embedding space at the input and output. This would allow us to do the following:

1. Train model 1 using tied embeddings and lm-head
2. Train a differently initialized model, model 2, but with the same (now frozen) embeddings and lm-head
3. For inference, remove the lm-head from model 1 and the embeddings from model 2, and stack the two, which, according the the original idea, should work well together because their embedding spaces are aligned

In principle, this could be scaled to stack many model atop each other.

## Experiments

You can find my code here: [https://github.com/snimu/model-stack](https://github.com/snimu/model-stack)

It's based on [this old modded-nanogpt speedrun](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/101024_Muon/train_gpt2.py), because 1) it uses tied embedding and unembedding weights, and 2) it's still a fairly simple model, which is probably good for stacking the models.

As my experiments, I simply followed the receipt from the previsous section, then compared the validation losses of the individual models and the stacked model.

### Model variations

When doing the experiments, I varied three things: the norms, how many of the layers are stacked, and model looping.

Norming the activations at the in- and output is important, because the loss on the final outputs includes a softmax, which means that the residual magnitude can differ between in- and output even if the embedding and lm-head weights are tied. I independently test norming the token embeddings (`norm_wte`), the inputs to the lm-head (`norm_lm_head`), and the residual activations in the transformer blocks between the two (`norm_inter_model`).

Not stacking all layers of the models is based on the observation that early and late layers are used to move embeddings into and out of the abstract space of actual thinking, to and from the concrete space of individual tokens. If we want to stack models, we should stack them in such a way that they share the abstract thoughts, not the concrete token-embeddings.

Looping the model like in [COCONUT](https://arxiv.org/abs/2412.06769) might teach the model to make use of its own outputs. If I do this to both models, and both work in the same embedding space at both in- and outputs, then model 2 might learn to make use of the outputs of model 1 at its input. To train these models with looping in mind during pre-training, I use the techniques described in my article [COCONUT: parallel pre-training](https://snimu.github.io/2025/01/12/COCONUT-parallel-pretraining.html) (Spoilers: this makes everything worse, so the method doesn't work).

As a baseline, I always stack the same model twice (when I don't, `unique_model` is true).

## Results

These are the column names:

- `val_loss_stack`: The validation loss of the model stack
- `val_loss_mean`: The mean of the validation losses of the models that are being stacked
- `use_first_layer`: Whether or not the first layer of the second model in the stack is used
- `use_last_layer`: Whether or not the last layer of the first model in the stack is used
- `norm_wte`: Did we use an embedding norm (in model training and the model stack)?
- `norm_lm_head`: Same but for the last transformer block, right before the lm head
- `norm_inter_model`: Should be called "intra-model" but whatever; it means norming the residual between the transformer blocks
- `unique_model`: If true, the two models in the model stack were trained from a different seed; if false, they are both the same model
- `coconut_every`: How many normal steps before each COCONUT-parallel-style update step?

### Full results

Here are the full results:

|   val_loss_stack | use_first_layer   | use_last_layer   | norm_wte   | norm_lm_head   | norm_inter_model   | unique_model   |   mean_val_loss |
|-----------------:|:------------------|:-----------------|:-----------|:---------------|:-------------------|:---------------|----------------:|
|            10.57 | False             | False            | layer_norm | layer_norm     | layer_norm         | False          |            3.31 |
|            10.02 | False             | False            | layer_norm | layer_norm     | layer_norm         | True           |            3.30 |
|             8.28 | False             | False            | none       | rms_norm       | none               | False          |            3.30 |
|             6.48 | False             | False            | none       | rms_norm       | none               | True           |            3.28 |
|            12.85 | False             | False            | rms_norm   | rms_norm       | rms_norm           | False          |            3.31 |
|            10.02 | False             | False            | rms_norm   | rms_norm       | rms_norm           | True           |            3.30 |
|             9.37 | False             | True             | layer_norm | layer_norm     | layer_norm         | False          |            3.31 |
|             9.97 | False             | True             | layer_norm | layer_norm     | layer_norm         | True           |            3.30 |
|             7.48 | False             | True             | none       | rms_norm       | none               | False          |            3.30 |
|             7.01 | False             | True             | none       | rms_norm       | none               | True           |            3.28 |
|            14.78 | False             | True             | rms_norm   | rms_norm       | rms_norm           | False          |            3.31 |
|            10.40 | False             | True             | rms_norm   | rms_norm       | rms_norm           | True           |            3.30 |
|            10.68 | True              | False            | layer_norm | layer_norm     | layer_norm         | False          |            3.31 |
|            10.42 | True              | False            | layer_norm | layer_norm     | layer_norm         | True           |            3.30 |
|             8.28 | True              | False            | none       | rms_norm       | none               | False          |            3.30 |
|             8.31 | True              | False            | none       | rms_norm       | none               | True           |            3.28 |
|            11.25 | True              | False            | rms_norm   | rms_norm       | rms_norm           | False          |            3.31 |
|            10.35 | True              | False            | rms_norm   | rms_norm       | rms_norm           | True           |            3.30 |
|            10.37 | True              | True             | layer_norm | layer_norm     | layer_norm         | False          |            3.31 |
|             9.84 | True              | True             | layer_norm | layer_norm     | layer_norm         | True           |            3.30 |
|             6.92 | True              | True             | none       | rms_norm       | none               | False          |            3.30 |
|             7.65 | True              | True             | none       | rms_norm       | none               | True           |            3.28 |
|            11.44 | True              | True             | rms_norm   | rms_norm       | rms_norm           | False          |            3.31 |
|             9.97 | True              | True             | rms_norm   | rms_norm       | rms_norm           | True           |            3.30 |

The most important results is that the model stack is always worse than the individual models it's made up from, and by a significant margin.

### Results by layer use

|   val_loss_stack |   mean_val_loss | use_first_layer   | use_last_layer   | unique_model   |
|-----------------:|----------------:|:------------------|:-----------------|:---------------|
|             9.79 |            3.30 | False             | False            | True           |
|             9.79 |            3.30 | True              | False            | True           |
|             9.60 |            3.30 | False             | True             | True           |
|             9.60 |            3.30 | True              | True             | True           |
|             9.79 |            3.30 | False             | False            | False          |
|             9.79 |            3.30 | True              | False            | False          |
|             9.60 |            3.30 | False             | True             | False          |
|             9.60 |            3.30 | True              | True             | False          |

Clearly, whether the first layer of the second model is used or not makes no difference, but whether or not the last layer of the first model is used does: not using it noticably improves the performance of the model stack. This is independent of whether or not the model being stacked is unique.

### Results by norm

|   val_loss_stack |   mean_val_loss | norm_wte   | norm_lm_head   | norm_inter_model   | unique_model   |
|-----------------:|----------------:|:-----------|:---------------|:-------------------|:---------------|
|            10.06 |            3.30 | layer_norm | layer_norm     | layer_norm         | True           |
|            10.19 |            3.30 | rms_norm   | rms_norm       | rms_norm           | True           |
|             7.36 |            3.28 | none       | rms_norm       | none               | True           |
|            10.24 |            3.31 | layer_norm | layer_norm     | layer_norm         | False          |
|            12.58 |            3.31 | rms_norm   | rms_norm       | rms_norm           | False          |
|             7.74 |            3.30 | none       | rms_norm       | none               | False          |

The first thing that jumps out at me is that here, using two different models does make a difference: it's significantly better than stacking the same model on top of itself.

Other than that, using not `norm_lm_head` and no `norm_wte` is by far the best setting, which I remember surprising me a lot, because I expected it to be important that the activations at the model in- and outputs are aligned, the more the better. Accordingly, I expected Layer Norm to outperform RMS Norm, whic it did by a small margin.

### Results with parallel COCONUT

I did these with the best setting discovered before: no `norm_lm_head` and no `norm_wte`, and two different models in the model stack. In this case, I performed two to four runs per setting, and will simply show the mean validation losses below.

|   val_loss_stack |   mean_val_loss | use_first_layer   | use_last_layer   |   coconut_every |
|-----------------:|----------------:|:------------------|:-----------------|----------------:|
|             8.09 |            3.32 | True              | True             |          100.00 |
|             8.00 |            3.32 | True              | False            |          100.00 |
|             8.85 |            3.32 | True              | True             |           10.00 |
|             9.66 |            3.32 | True              | False            |           10.00 |

Some observations:

- Using model looping in the way I've proposed increases the validation loss of the trained model (that's another hypothesis disproven)
- Using model looping more often during training doesn't worsen the trained model significantly, but it does worsen the stacked model
- Using the last layer of the first model makes a larger difference when `coconut_every` is lower

All in all, this trick seems to make things worse rather than beter.

## Addendum

*Edit 2025-06-08:* I've found the notes for my early experimental runs which contain additional information, including mention of one ephemeral run that I could never reproduce where the model stack had a validation loss of 3.0 while the individual models had a loss of ~3.3; I remember now that I was *extremely* frustrated about losing that.

The experiments presented above were my final ablations done after everything noted below, so they contain a lot of the same, but the above and below sections don't fully overlap.

*Warning: these are my raw running notes and I swear a lot, I'm just copy-pasting them in.*

### 2025-02-18

Code: [https://github.com/snimu/model-stack](https://github.com/snimu/model-stack)

I based my code on [this old modded-nanogpt speedrun](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/101024_Muon/train_gpt2.py), because 1) it uses tied embedding and unembedding weights, and 2) it's still a fairly simple model, which is probably good for stacking the models.

I trained the first model on 3.1M tokens [CORRECTION: 3.25 *billion* tokens, I don't know what I thought...] of fineweb, took its embedding weights, froze them, and trained a second model with them on the same data. Then, I stacked them. I either removed the last transformer block from the first model when stacked ("Layer removed") or not ("Layer kept").

As a first baseline, I also trained two models with different embedding weights and stacked them, to see what would happen.

Here are the final validation losses for the individual models, and their stack:

| Layer removed | Shared embeddings | Model 1 val loss | Model 2 val loss | Stack val loss |
| ------------- | ----------------- | ---------------- | ---------------- | -------------- |
| No            | No                | 3.28             | 3.28             | 7.26           |
| No            | Yes               | 3.28             | 3.31             | 6.30           |
| Yes           | No                | 3.28             | 3.28             | 6.66           |
| Yes           | Yes               | 3.28             | 3.31             | 5.90           |

The three takeaways are:

1. So far, this approach has failed
   - The model stack is not better than the individual models
2. Sharing the embedding weights helps noticably with stackings
   - This is promising; I might be on a good track, and things like training for more tokens might solve the issue
   - However, the chances of success are still very small
3. Removing the first layer of model 2 helps in both cases
   - Hypothesis 1: The [logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) is right that the first layer is what turns the input embeddings into next-token predictions; all other layers are only there to refine the predictions. If this is true, the problem of aligning the predicted-token-positions between models is solved
   - Hypothesis 2: Removing a layer from model 2 removes one transformation of the output of  model 1. With every layer that is removed, the performance increases.

As a second baseline, I only trained a single model and stacked it with itself (of course, it inherently can only use shared embeddings). Here are the results:

| Layer removed | Model 1 val loss | Stack val loss |
| ------------- | ---------------- | -------------- |
| No            | 3.28             | 7.28           |
| Yes           | 3.28             | 6.28           |

This is best compared to the models with shared embeddings. In this case, stacking the same model twice leads to significantly worse results than stacking two different models. This is a hint that removing the first layer of model 2 helps because it's what turns the input embeddings into next-token predictions; otherwise, why would model 2 hurt performance less than model 1, when applied to the outputs of model 1?

Next steps:

- Shuffle data
  - By default, the models train on the same data *in the same order*, and only their initialization is different.
  - Changing this could make a difference, but I don't think it will
- Train for longer
  - I don't know if the embedding weights from model 1 are even trained sufficiently after 8M tokens [EDIT: again, it's 3.25 billion tokens]
  - Training for longer &mdash; for example, for 1B, or 10B tokens &mdash; might help [EDIT: agian, it's already 3.25 billion tokens]
- Larger models
  - It's possible that the model size is a limiting factor for the ability to make use of latents
  - If so, scaling will help
- Remove last layer of model 1
  - Removing the first layer of model 2 helps
  - Just to see what happens, would removing the last layer of model 1 help, too?
- Remove more layers of model 2
  - Distinguish between the two hypotheses above

### 2025-02-23

I have done two additional things:

1. Trained for 5x as many tokens as before, so 16.25 billion tokens
2. Also tried cutting off the last layer of model 1

Here are the results:

|   val_loss_stack |   val_loss_1 |   val_loss_2 | use_first_layer   | use_last_layer   | same_model_twice   |
|------------------|--------------|--------------|-------------------|------------------|--------------------|
|         11.3679  |      3.14785 |      3.20779 | False             | False            | False              |
|         14.7551  |      3.14785 |      3.20779 | True              | False            | False              |
|          8.91999 |      3.14785 |      3.20779 | False             | True             | False              |
|         12.2524  |      3.14785 |      3.20779 | True              | True             | False              |
|         10.0821  |      3.14785 |      3.14785 | False             | False            | True               |
|         16.0815  |      3.14785 |      3.14785 | True              | False            | True               |
|         10.547   |      3.14785 |      3.14785 | False             | True             | True               |
|         16.5895  |      3.14785 |      3.14785 | True              | True             | True               |

A few observations:

1. Training for longer makes model stacking worse, not better.
2. Removing the last layer of model 1 makes model stacking way worse.
3. Removing the first layer of model 2 is still really good.

The first point makes me really pessimistic about this method. I'll be working on other things for now.

### 2025-02-25

I've just noticed that I didn't use a norm between the two models. Maybe that would help?

God fucking damn it. I've had one run where the stacked loss is 3.0... and the individual models are 3.3..., where `use_first_layer is False`, `use_last_layer is False`, and `use_norm is True`. But then I've noticed that I'm only using 4 out of my 8 GPUs, changed that, re-trained while I'm at it, and now the results look like this:

|   val_loss_stack |   val_loss_1 |   val_loss_2 | use_first_layer   | use_last_layer   | use_norm   | same_model_twice   |
|------------------|--------------|--------------|-------------------|------------------|------------|--------------------|
|         13.2456  |      3.28372 |      3.31274 | False             | False            | True       | False              |
|         10.7768  |      3.28372 |      3.31274 | True              | False            | True       | False              |
|         11.4209  |      3.28372 |      3.31274 | False             | True             | True       | False              |
|         11.7097  |      3.28372 |      3.31274 | True              | True             | True       | False              |
|         11.2224  |      3.28372 |      3.28372 | False             | False            | True       | True               |
|          8.59775 |      3.28372 |      3.28372 | True              | False            | True       | True               |
|         10.455   |      3.28372 |      3.28372 | False             | True             | True       | True               |
|          8.53166 |      3.28372 |      3.28372 | True              | True             | True       | True               |
|          7.20448 |      3.28372 |      3.31274 | False             | False            | False      | False              |
|          7.05133 |      3.28372 |      3.31274 | True              | False            | False      | False              |
|          6.2493  |      3.28372 |      3.31274 | False             | True             | False      | False              |
|          6.66394 |      3.28372 |      3.31274 | True              | True             | False      | False              |
|          5.87195 |      3.28372 |      3.28372 | False             | False            | False      | True               |
|          8.05615 |      3.28372 |      3.28372 | True              | False            | False      | True               |
|          6.49195 |      3.28372 |      3.28372 | False             | True             | False      | True               |
|          7.59556 |      3.28372 |      3.28372 | True              | True             | False      | True               |

Clearly, norming is much worse than not norming. WTF?

So let's do a few sanity checks:

1. Are the wte and lm_head weights tied?
    - Yes, they are, very clearly
2. Do the individual models actually reach the loss they reached in training?
    - Yes, they do

Okay, so I clearly fucked up somewhere. I've been going at this over and over again, training and validating with slight changes to the code, but the results always suck.

OMG I just noticed that in this old version of nanogpt, the GPT model doesn't use rms_norm on the embeddings, no wonder norming is worse. I'll try with that soon.

### 2025-02-26

I have now implemented the following changes:

- If "--use-norm" is set during training, the GPT will use rms_norm on the embeddings
- If "--use-norm" is set during model stacking, the GPT will use rms_norm on the embeddings, between every model it consists of, and before the lm_head

Question: Should I use layer-norm between wte & blocks, and blocks & lm_head? That way, their residual streams would be even more similar.

---

*This is where my notes stopped. I think right afterwards, I ran the ablations presented in [Results](#results) but never presented them because everything was done a bit sloppily and I was unsure whether or not I would continue the work.*

## Conclusion

I won't be pursuing this idea further.

## Citation

```bibtex
@misc{snimu2025modelstackingdoesntwork,
    title={Model stacking doesn't work},
    author={Sebastian M\"uller},
    year={2025},
    month={6},
    url={https://snimu.github.io/2025/06/07/model-stacking.html}
}
```
