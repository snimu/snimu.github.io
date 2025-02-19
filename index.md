---
layout: home
title: "snimu's blog"
---

Hi :) I'm Sebastian. At my job, I work on LLM agents, and in my free time, I train language models and write about AI.

## Blog posts

{% for post in site.posts %}
  * [{{ post.title }}]({{ post.url }}) - {{ post.date | date: "%B %d, %Y" }}
{% endfor %}