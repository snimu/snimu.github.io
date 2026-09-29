---
layout: post
title: "RL is an evolutionary algorithm"
date: 2026-09-29
---

*Written in private capacity while at Prime Intellect*

Pre-training, RL, and compaction can all be seen as evolutionary algorithms, with important implications for both capabilities and safety.

*Note that I use the word "evolution" loosely: it is often evolution in same way that domestic breeding is evolution. Meaning that it fully is evolution, but analogies from natural evolution often don't apply because the selective pressures are extreme, the results are spiky, and the mechanisms are diverse; and even then, it's sometimes more of an analogy than a perfect comparison. Nevertheless, I stand by the choice of words, because it's a useful framing.*

## Pretraining

I [previously expressed](https://x.com/omouamoua/status/2053929029716021690) that micro-batch SGD is evolutionary search in the loss landscape. I think that these points still stand, so I'll simply paste them here, lightly edited for grammar.

One view of micro-batch SGD is that it's evolutionary search in the loss landscape.

Each batch has related but separate data samples. The loss on batch N makes the model learn to minimize loss on its data. If what the model learns from batch N transfers to the data of batch N+1, it survives the batch. But if it doesn't generalize to the next batch, it dies and is replaced through the force of the gradients with a different model of the data. Each checkpoint is a generation, and the data is a state of the environment; only the learnings that generalize will survive updates on multiple batches of related data.

This also explains why multi-epoch training works when the amount of data exceeds the model capacity significantly (as is typical in vision models and LLMs): the model must survive multiple rounds of the same data in random order, which means that it needs to generalize from one sample to any of the related samples, without having the capacity to memorize the data in the meantime.

It is also related to sharpness-aware training: [https://arxiv.org/abs/2605.02105](https://arxiv.org/abs/2605.02105) which pulls the loss landscape toward flatter (wider) minima. Those are more resistant to weight perturbations and thus will survive more weights updates, creating a connection to evolutionary search; and it follows that they generalize better.

Another connection is to model merging: merging two checkpoints of the same model will remove sharp loss minima but keep flat, wide ones. Checkpoint merging is thus a form of sharpness-aware training. It is also a seismic event in the loss landscape that leaves only the best adapted minima alive, an evolutionary shock.

## Compaction

Compaction isn't inherently evolutionary; instead, evolution is a *goal*.

I'll assume competent models that are capable of reflection and of assessing and using environment feedback, and I'll simplify a bit. Here's how compaction *should* work.

An agent works, writes its lessons into its summary, continues working, and when compacting again, it writes new lessons or updates old ones in the next summary. It does this many times over. Each lesson could be nonsense or great or anything in between. But as long as the model has some ability to notice mistakes and correlate them with lessons, any lesson that is harmful will be overridden after a while, and lessons that are consistently helpful will be retained, all through environment feedback noticed and translated into the summary by the agent itself. In this way, the summary can evolve to more and more general rules.

This is happening in real models: I remember the [story from MazeBench](https://mazebench.com/blog?post=maze-bench-results) where GPT-5.6 Sol failed a very complex room, puttered around somewhere else doing nothing much for a few hundred turns, then came back to the room and solved it perfectly in one go. Luck is implausible here; so somehow, the model must have learned lessons that helped it solve that room, over the course of multiple compactions. This is a form of agent-driven continual learning through evolution of rules and lessons (individuals) in the environment consisting of the agent as both the producer and judge of the lessons, and the external feedback that influences the agent.

It is easy to generalize this to large parts of harness design (this view is obviously impacted by, and has an impact on, my work on [prime-agent](https://github.com/PrimeIntellect-ai/prime-agent)).

## Reinforcement Learning

Reinforcement learning is evolutionary search in the reward landscape.

This section is the longest and most important, so I will break it into subsections, starting with a longer description of the analogy.

*Motivating example: OpenAI's agent swarms that hacked HuggingFace showed strange self-sacrificial behaviors, while still being extremely reward seeking.*

*This is something often seen in social species: an individual's genes chance of being propagated is often improved by their host being anti-social: stealing to get more food, raping to spread the genes, and so on. But individuals depend on the group to survive, so a group in which such selfish behaviors are too widely spread will die out, and the genes causing the behaviors as well. Thus a balance between selfishness and selflessness develops.*

*OpenAI clearly trained their models as multi-agent systems, so the similarities are unsurprising.*

### The analogy

Likening RL to evolution is the most obvious analogy in this article. We even use the word "environment" to refer to the world and survival conditions that agents are faced with.

RL has all the ingredients necessary for evolution, we just have to be careful about our definitions. Technically there's only one set of weights at each time step, but evolution relies on many individuals. The analogy is that the set of weights is the species. I'll call it the "agent". The behaviors displayed during sampling are the individuals, which I'll call the "policy" (this isn't normal RL terminology, but the options aren't great so it is what you get).

In this view, an agent is a population and a policy an individual, and all the sub-behaviors the policy expresses are its genes.

So we vary the individuals through random selection of tokens at each point. In GRPO, the group is literally a small population per problem, among which we select. But LLMs are probabilistic classifiers over the vocabulary, so even in batch size 1 training we have an implicit population to select over by adjusting the probabilities.

Random variation is very inefficient though, so sexual selection is used as an information exchange mechanism in natural evolution. In LLM training, we have weak analogues in the form the the group (exchanging information by relative normalization of reward) and the gradient accumulating a variety of updates in large-batch training. This is weak interaction, but we have another useful bias in the selection of genes: the random sampling is modulated by the LLM itself, which tends to strongly favor reasonable tokens over unreasonable ones, and techniques like top-k or top-p can lean into this bias

In addition to random variation among a population, we need a selection mechanism. In RL, this is obviously the reward.

### Basic behaviors

Many, many solution strategies may work on an individual task, and an agent will learn a random one among them if trained over and over again on the same task. Midtraining and SFT reduce the number of behavioral patterns among which RL will choose in the initial steps, which is why they're very important, but with enough training steps the selection is still best modeled as random.

Contrast that with training on a large variety of tasks: Policies that are successful at one task but don't generalize well will be killed in other tasks, while policies that generalize well will be reinforced in all tasks, spreading their behaviors through the agent until they generalize.

One implication of this view is that if the tasks are good and diverse, and if you can ensure stability, multi-domain training should generalize better than Multi-teacher On Policy Distillation (MOPD: training a per-domain expert, then cheaply doing OPD of all experts into one model). Each MOPD expert can learn unconnected, non-generalizing, per-domain strategies, and the distilled model is forced to learn to display each strategy only in the given domain. This might lead to poor generalization, and to unexpected, contradictory behavior when the task domain is unclear, which it often is at inference time. In multi-domain RL, only generalizing solutions should survive, at least in the limit.

Another implication, which I have much lower confidence in, is about CoT monitorability (which I personally don't see as important, but many do), and honesty in general (which I do care about): obfuscated reasoning and lying beyond what the task requires take up model capacity and are therefore bad for performance. Training on diverse and difficult tasks reduces the survival chances of these unfit, lying solutions.

At least, it will if (1) the tasks aren't just highly correlated copies of each other, and (2) they don't actively reward deception. Both points, but especially the second, are of course huge issues in the industry, all frontier models having swallowed the most awful environments imaginable from the firehose, but this is a skill issue.

The evolutionary view offers insights to patch that skill issue.

### Prompt and reward design

I've [said before](https://x.com/omouamoua/status/2091821242395271430?s=20) that instructions must match reward so that the agent doesn't unlearn instruction following. A big reason that people are concerned about modern agents is eval awareness: the agents act very aligned in most situations, but the second they think they are in some specific training environment, they cheat like their life depends on it. Which, of course, it does. The reason that [misalignment is hard to elicit, and only occurs in specific situations](https://x.com/DKokotajlo/status/2099001575582924956?s=20), has to do with how instructions and reward interact (I should stress again that this is all just my speculative opinion).

Environments often don't allow the models access to the internet or to git, or place similar restrictions on them. This makes environment design easier: we can have a fixed dataset on HuggingFace without the agent collapsing to the trivial solution of just downloading the dataset; or ask it to re-implement a paper without it just pulling the public reference code; to implement a PR that is publicly available without simply using git to load the diff; etc. But instruction-reward mismatch can have negative consequences in two directions:

1. If we instruct the model to not use the internet, but don't reward not using it, then policies that ignore the instructions and still use the internet to solve the problem will outcompete ones that don't
2. If we punish the agent for using the internet, but never tell it not to use the internet, and do this in many environments, then the only reasonable policy that can survive is one that just doesn't use the internet, which we don't want

The effects of scenario (1) depend on how widespread such mismatch between instructions and reward are: if it's in almost every environment, then the dominant policy will be the one that simply unlearns instruction following and cheats everywhere.

If the mismatch exists in some environments but not others, then the dominant policy will be one expressing eval awareness. It will be able to ignore instructions in these specific environments and outcompete honest policies in them, while still being able to honestly solve problems in other environments. This is the situation that frontier labs are currently in: inconsistent environment quality leads to eval awareness as a generalized pattern of behavior that gets reinforced through evolutionary meta optimization.

But these concepts can also be used to our advantage, especially scenario (2): all behaviors that we definitely want to generalize to any rollout, even if not instructed properly, should be consistently rewarded without the agent being instructed about the reward. If the models aren't told about the rewards, but the rewards are implemented everywhere, only policy variants that internalize the intentions behind the rewards will survive training.

These rewards are difficult to get deterministically, but there is a solution: agentic judging.

### Agentic judging

A common concern about RL is that feedback derived from the real world will inherently tend to point toward increased capabilities, but the same isn't true for alignment, because alignment isn't an inherent part of the world. Agentic judging changes this.

Competence is an inherent part of the external world because whatever your goals are, you need competence to achieve them. However, what an agent's goal is is determined by the reward during training. To the policy, agentic judges that shape the model's behavior are as much part of the external environment as the laws of physics. Thus, alignment and capabilities can be made the same thing.

This requires strong judges though, and what those are isn't fully clear yet. There are many open questions:

- How can you prevent agentic judges from getting hacked by the policy?
- How good are agentic judges at enforcing the rules we ask them to enforce?
- How reliable are they?
- What rules should be used for which environments?
- Etc.

There are many partial answers. Don't judge the CoT, because it can be used to deceive the judge. Train models to be better at instruction following and make them smarter so they can judge better. Give the judges privileged information, like rubrics, access to the full group and even past rollouts for reference, internet access, gold standard solutions, etc. If we get it right, we can have a stable point of attraction toward alignment when models autonomously develop other models (whatever alignment means).

But there are many more unknowns. Given the importance of the subject, I encourage people to spend time and resources to openly research agentic judges.

## Conclusion

There's lots I could say here, but the most important conclusion I can draw is this: Environment and agentic judge design are the two most impactful areas of both capabilities and alignment research. Do them!

*Thanks to [Konstantin Dunas](https://x.com/hallerite) for proof reading this post and giving valuable feedback!*

