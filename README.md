# AustenGPT

This repo implements a simple GPT, based on Andrej Karpathy's [NanoGPT](https://github.com/karpathy/nanoGPT/tree/master).

It will be trained on Jane Austen's texts to generate more Jane Austen!

## GPT-walkthrough

A good starting point is the [gpt-walkthrough.ipynb](gpt-walkthrough.ipynb) notebook, which contains notes from Andrej Karpathy's [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=9) video. This notebook starts from a simple bigram model, then progressively introduces self-attention, multi-head self-attention, as well as optimizations such as LayerNorm. It also includes pretty detailed notes on each of the steps.

The GPT in the notebook is a **character-level model** trained on **Jane Austen's *Emma***. By the end of the walkthrough, you can generate text like:

```
justicular
imaginell the Frank
Colong.
Hence, you nother best, by the agreably extralled, and
seceshamed Jane Fairfax at all only thoughth, it was scruptions, and _weather passionable that I am pain to you this declination?”—and worth him spiritéing servaking away, and she did I should never
most desome once bethlunted it drequest anxion of
farther, who concew any recoxcepte,
in steased, one brought,” createst-Miss Bates’s were slight conshaded
what is quite
from any body’s one bring? who half in the great he kind; and
solution, that to
Mr. Elton issurple, his middles than understanding. He
had
only living us. She few the little, you short placle some
days could her own used him; and all there
is quite for so.
Colongut sometingsfore her
most to be disappoitations what is her
father’s enjictely did spodering the nobody that she disgue, Jane, if he us, suffulnuicate Box to convise present it thought another. Mr. Cole, which she than Harreated to
her worthing
advarer.”

“She safely, and c
```

## AustenGPT

The [austen-gpt](./austen-gpt/) directory contains an implementation of AustenGPT.

For a more detailed description of its implementation, see [austen-gpt/README.md](./austen-gpt/README.md).