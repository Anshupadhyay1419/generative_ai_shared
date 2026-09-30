# LSTM translation with concatenate attention

- [Student lab](seq2seq_attention_lab.ipynb)
- [Instructor solution](seq2seq_attention_solution.ipynb)
- [Notebook builder](build_attention_lab.py)

This lab adapts the local `../EncodeDecoderTranslator.ipynb` baseline to German-to-English translation with the [Multi30k dataset](https://huggingface.co/datasets/bentrevett/multi30k). It uses encoder output states and a previous-decoder-state query as taught in [concatenate attention](../../seq2seq_learning/concatenate_attn.md). The [existing LSTM notebook](../../seq2seq_learning/Seq2Seq_LSTM.ipynb) supplies fixed-context and teacher-forcing background; the [tweet classifier](../../seq2seq_learning/disaster_tweets_with_attention.ipynb) supplies a sequence-to-one contrast. The source notebooks have not been modified.

## Teaching alignment

| Element | Alignment |
| --- | --- |
| Module | Module 2: attention as a bridge from recurrent encoder–decoders to Transformer attention |
| CO1 | Implement and trace scoring, masking, context formation, and decoder updates |
| CO2 | Analyze fixed context versus attention, greedy versus beam, and perplexity versus BLEU |
| CO3 | Design a controlled translation experiment and diagnose failures |
| Prerequisites | Introductory Python/PyTorch, LSTMs, padding, cross-entropy, and teacher forcing |

The two-hour core is data flow, baseline execution, concatenate-attention implementation, and shape/mask/gradient checks. Beam search, full GPU training, and the metric comparison are suitable for the following week. The notebook defaults to a tiny smoke corpus with full training disabled, so a top-to-bottom run does not download data or claim translation quality. Set the documented switches for a full run.

The full run needs `torch`, `datasets`, `matplotlib`, and `sacrebleu`. Multi30k data download requires access to its Hugging Face repository or a locally attached copy. The dataset card has no explicit license field; consult the [upstream repository](https://github.com/multi30k/dataset) before redistributing data. Do not commit downloads or model checkpoints. The solution contains expected reasoning but no invented training scores.

After editing the builder, run `python build_attention_lab.py` and validate both generated notebooks. Runtime checks completed locally in smoke mode on CPU with PyTorch 2.4.0; full dataset download and GPU training have not been run here.
