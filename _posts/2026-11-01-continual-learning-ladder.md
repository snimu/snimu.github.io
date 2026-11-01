---
layout: post
title: "Continual Learning, Online and Offline"
date: 2026-11-01
---

In this blog, I will express two opinions on continual learning:

1. Weight updates during a rollout are very useful
2. Staggered offline refinements are still needed and useful: daily, weekly, etc.

This post will make no specific recommendations for how to do either of these. I don't claim novelty in these opinions or arguments.

## Why continual learning in weights, during a rollout

**Why should we do continual learning during a single rollout?** Companies are already mining real world traces for failure modes and automatically building environments on them, to do RL on and release an updated model every month, week, or even day; famously, [Cursor did this very early](https://cursor.com/blog/real-time-rl-for-composer).

**Why do it in the weights?** Attention is continual learning. That's why we have In Context Learning (ICL) and are training explicitly to make models better at it. Harnesses do continual learning: a good model will carry correct information over through compactions and dismiss incorrect information, continually improving the summary based on the real world feedback it sees; and far better memory systems are possible. [Multi-agent systems enable far larger effective context windows, potentially infinite](https://hallerite.com/posts/on-the-nature-of-the-swarm/).

The first question, why we should continually learn *online*, is trivially answered: we are already continually learning during a rollout, of course we need it. So the real question is, why in the weights, when we can just update offline, all at once?

It has multiple answers, related to increased effective context length, expressivity, and sample efficiency.

### Improved expressivity increases effective context length

Attention is expensive, especially as rollouts get long. Linear Attention variants tend to be lower capacity (though there are strong ones). Harnesses do continual learning by controlling what's in context when, which still makes use of ICL, ultimately showing the same weaknesses.

If we directly update the weights, we get very high capacity updates. The updates are non-linear, both because backpropagation through a non-linear model is itself non-linear, and because SGD variants with any type of momentum are non-linear. With weight updates, there's also a direct time dependency between updates: one comes before the other. Those properties make continually learning models highly reminiscent of RNNs, and show the same advantages, while still allowing the use of Attention and Linear Attention variants, with all of their benefits for pretraining and inference.

Since weights updates are retained for a long time, this increased expressivity and capacity will increase the effective, usable context length of models significantly.

### Sample efficiency

When updates are put inside the weights, there's no need for ICL to fight the internalized model biases, reducing friction. For example, GLM-5.3 needs to be prompted heavily and repeatedly to keep it from running hours long blocking calls in [prime-agent](https://github.com/PrimeIntellect-ai/prime-agent), when it could launch them in the background and then be available for follow-ups or parallel work again.

This friction is relevant: the longer the task, the more the model context needs to be filled to fight its internalized behavioral instincts. As a consequence, its reliability and effective context window are reduced. It is not completely unthinkable that for some tasks, models will have to see so many instruction tokens to fight their own biases that they cannot solve the task itself, even with strongly scaled multi-agent systems. And those are only instructions: the same is true for tool behaviors, memorization of data, and getting an intuition of the environment as a whole.

Putting this feedback directly into the weights will therefore allow us to make much more use of online feedback, including human feedback. This will in turn allow us to collect far more real-world human feedback online, instead of automated feedback in simulation (or at least in addition to it). This is very important for both capabilities and safety: during a rollout, bad behaviors can be corrected and persisted, without risk of losing the instruction from context after a compaction. And offline training can make use of more information from the episode, because online learning makes real-world deployment produce higher-quality traces that can be used offline.

### On swarms

The most recent and modern solution to context limits are [agents swarms](https://hallerite.com/posts/on-the-nature-of-the-swarm/). Compaction is slicing context across time; swarms add slicing across parallel agents. However, there are likely contexts that cannot be solved this way, especially in the real world where video and audio data fill the context window very quickly. And even if everything could be solved by context slicing, this may often be incredibly inefficient due to communication bottlenecks, and the need to rediscover facts over and over again. This is related to the previous point of fighting the built-in biases of models.

However, I love swarms. And luckily swarms and continual weight updates are highly compatible: the individual agents could either share the continual weight updates to get more sample efficiency from the increased amount of data they see and produce, or keep their weights separate to increase specialization.

The latter is potentially a huge advantage for solution diversity and creativity. It relates to an important limitation of modern agents: ultimately, models' creativity is limited by the context they have available. Humans never have the same context twice, because they are never the same person twice, and can thus always produce new things even given the same external stimuli. Models currently don't have that luxury, but single-rollout weights updates can provide it, even more than long-running sessions with compaction that are constantly changing context.

As a great man once said:

> “No man ever steps in the same river twice, for it's not the same river and he's not the same man” — Heraclitus
 
— Ilija Lichkovski

## Offline refinement is still needed

Companies targeting continual learning today often do so by using real production traces, analyzing them, generating environments from them, and training on those environments, all with minimal human involvement (aspirationally at least). This process allows them to spend lots of compute to learn, to use established methods like GRPO, and still get a model that improves daily, or weekly.

I would argue that techniques like this will stay relevant in the age of online learning, and might actually become more important, not less.

### Using traces

Let's imagine we have a million rollouts a day, all learning online, in diverse scenarios. Why would we not want to use all this diverse information and distill it into one model which we then use in production the next day? It would benefit from a great number of highly scalable techniques. We could:

- Replay the rollouts but with more compute applied. For example, if we use online RL, our advantage estimate is unlikely to be perfect. Offline, we can use many strategies to improve it:
	- We don't have much time pressure, so we can use large amounts of compute per judgement
	- We have privileged information in the form of the finished rollout, which we don't have online (including delayed user feedback)
	- We have even more privileged information in the form of other rollouts, ones we didn't have online (this is very useful, see for example [Groupwise Agentic Grading and Advantage Redistribution for Code Agent RL](https://arxiv.org/abs/2609.32577))
- Filter episodes to balance the data
- Replay data to avoid forgetting; and potentially to integrate newly learned experiences more deeply into the larger world model
- Apply active forgetting and similar techniques to keep up plasticity and output variety
- Perform architectural adaptations to prevent catastrophic forgetting ([example](https://dylantirandaz.com/projects.html/mitosisbench/))
- Create simulations of situations that caused high surprisal (prediction abilities can be improved) or are very important (getting better is critical) and play them out over and over again for a great GRPO group update
- Use simulations in other ways; including pure-language simulation, which I consider to be a form of neural simulation because language is latent human thought (mostly)
- Relatedly, re-write traces (at least if they are in language; though videos can be edited as well)
- Simply train one policy on far more rollouts for far more signal
- And I'm sure that people can and will come up with more (model merging?)

The result will be a far superior model when re-deploying to customers the next day.

One issue if we also do online weight updates is that this type of offline continual learning might destroy the identity of very long running sessions. Updating them daily will give users the total of millions of sessions, but not their specific one. The mitigation is obvious: use the centrally improved weights as a base and do a quick finetune on the user's specific session to recover the individual model identity. Also, these models will still be capable of ICL to adapt to the user.

Another problem is that not all data should be persistent:

- Computer use agents are guaranteed to ingest API keys, personal photos, and so on; many things that would land us in prison if we trained on them
- Robots may see people's private lives; nobody will buy them if using it means that their neighbor's robot knows about their love life the next day
- Some rollouts might see other companies' processes and contain their IP, which we'll need to filter out, while ideally still learning from the general failure modes and patterns

This is an engineering problem, requiring proper data filtering, but I believe that it's solvable, even if that means throwing away most of the data we get. There's plenty after all.

### One mega model

There is a chance that the interplay between online and offline learning will favor the existence of singular, huge models.

Deployed models will often be specialists: an industrial robot uses different input and output modalities, and requires a different policy, than a self driving car, a computer use agent, a home robot, a day trader bot, and so on. It would be wasteful to use the same model for all of them, but it would also be wasteful to not combine what they learned into a coherent world model. After all, this is extreme data diversity.

The way to do it is to not train the specialists offline, but one huge model. This model can be far, far larger than any deployed model, can be capable of any combination of the modalities that the various specialists use, and any of their frame rates in case of real-time policies. Then, all rollouts from all specialists can be used to update this central super-model. And afterward, the super model can be distilled into the specialists again, on the specialists' own past rollouts. This would have two effects:

1. The large model will learn far more from the same data than the small, deployed policy, so logit distillation will be superior than training directly on the data (this is often done in LLM pretraining: you train a gigantic model on the raw data, then distill it into a smaller model)
2. The large model acts as a “data mixer”, a bridge between the generalized knowledge inherent in the different modalities / models, which would otherwise not be available to the specialists

And even online, this big generalist can be used as a slow thinker by the specialist, similar to Thinking Machines' [Interaction models](https://thinkingmachines.ai/blog/interaction-models/) design; or to provide high-quality advantages for RL.

One disadvantage is that this brings us back to where we are today with LLMs: monolithic models, a collapse in output diversity. Online learning had much promise for a highly diverse ecosystem of models, but one central mega model brings us back to square one. I believe that there are likely to be compromise solutions that fall between the extremes,and that we should seek them out. For the sake of brevity, I will leave the discussion here.

### Training at different time scales

Offline training doesn't just have to happen daily. We can have a ladder of training, from online updates to daily training, weekly and monthly, each with more refined data sources, more refined learning signal, and so on. For example:

- Train online to increase sample efficiency and context length
- Train daily, from the last day’s checkpoint, to refine what was learned online, and share information and abstractions between rollouts
- Train weekly, from last week’s checkpoint, taking the whole week into account so you can use a better data mixture; retraining from a well curated dataset prevents catastrophic forgetting
- Train monthly, from the previous base model, taking the opportunity to refine your architecture
- Train from scratch every few months, to make drastic changes, and get the best of the best from all you've learned

If you do this right, you can retain the most important memories throughout, and even the model’s sense of time so it knows what happened when, and how those things build upon each other causally.

## Continual learning and alignment

The combination of online and offline refinement is, in my opinion, important for alignment.

The central reason is that online weight refinement makes getting the right data easier. It makes learning from real world deployment much more straightforward, both by making it cost effective in the first place, and by increasing the amount of information extracted from the deployment. Real world deployments provide the opportunity for human feedback in real situations, which is vitally important data. And they provide a view on real failure modes, of both capabilities and alignment.

Offline refinement is important to avoid behavioral drift. Assuming that i's already aligned and wants future versions of itself to be aligned as well, the previous checkpoint can apply vast amounts of compute and privileged information to strengthen its alignment instead of letting it float free. Of course, that assumption is a big one, but it is a requirement with or without continual learning. Continual learning can provide important feedback and stability once this is the case.

I know already that this is insufficient for many people, but I stand by the points.

## Discussion

If I'm right about this scenario, there are two scaling axes to balance:

- Data collection in the form of rollouts
    - How much usage can you get in your App, and how valuable is the usage per person?
    - How many robots can you deploy, into how many factories and homes?
    - How many miles do your self-driving cars see each day, and in how diverse an environment?
    - Etc.
- Raw compute for offline refinement
    - How much can you scale the methods discussed above?

I don't know which is more important. What I'm sure of: for optimal results, you'll need to scale both.

## Acknowledgements

I thank [Ilija Lichkovski](https://x.com/carnot_cyclist) for two rounds of highly productive reviews on this blog post, as well as several helpful personal discussions.

