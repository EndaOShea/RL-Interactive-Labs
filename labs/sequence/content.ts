import { LabContent } from '../../catalog/types';

// Co-located theory + lifecycle content for the Sequence-Models labs (RNN, LSTM,
// seq2seq), rendered in each lab's Context tab via LabContext. These are the
// recurrent analogue of the Deep-Learning ResNet lab: how a signal — here a
// gradient flowing back through TIME — survives or dies as depth grows.

export const RNN_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Recurrence: a network unrolled through time',
      body: 'A recurrent network processes a sequence one step at a time, carrying a hidden state h that summarises everything seen so far: h_t = tanh(W_hh·h_{t-1} + W_xh·x_t + b). The SAME weights are reused at every timestep, so an RNN of length T behaves like a feed-forward net T layers deep that shares parameters. That weight sharing is what lets it handle variable-length inputs, but it also means a single recurrent matrix W_hh is applied over and over.',
      details: [
        { label: 'Hidden state h_t', text: 'A fixed-width memory updated each step; it is the only channel through which the past reaches the future.' },
        { label: 'Shared W_hh', text: 'One recurrent matrix is reused at every timestep — the unrolled net is deep but parameter-tied.' },
        { label: 'tanh squashing', text: 'Keeps activations bounded in (−1, 1); its derivative tanh′ ≤ 1 will matter for the gradient.' },
      ],
    },
    {
      heading: 'BPTT & the vanishing / exploding gradient',
      body: 'Training uses Backpropagation Through Time (BPTT): the loss gradient is chained backward across every timestep. The Jacobian that carries a gradient from step t back to step t−k is a PRODUCT, ∂h_t/∂h_{t−k} = Π diag(tanh′)·W_hh. Repeated multiplication by the same matrix makes that product behave like (factor)^k: if the effective factor (the spectral radius of W_hh times the average tanh′) is below 1 the gradient VANISHES exponentially, and above 1 it EXPLODES. Either way the network struggles to learn long-range dependencies — the motivation for gating and gradient clipping.',
      details: [
        { label: 'Product of Jacobians', text: 'Gradient over k steps ≈ (tanh′·ρ)^k where ρ is the spectral radius of W_hh — exponential in the lag.' },
        { label: 'Vanishing (<1)', text: 'Early timesteps receive almost no signal, so long-range dependencies are never learned.' },
        { label: 'Exploding (>1)', text: 'The gradient blows up; training destabilises. Gradient clipping caps the norm to keep steps sane.' },
      ],
    },
    {
      heading: 'Why this matters & how it is mitigated',
      body: 'The vanishing-gradient problem is the recurrent version of the depth problem the ResNet lab tackles with skip connections. Here the fixes are: clip the gradient norm to tame explosions, initialise W_hh orthogonally (spectral radius ≈ 1) to delay vanishing, and — most importantly — add gating (LSTM/GRU) that gives the gradient a near-identity path through time. Attention later removes the bottleneck entirely by letting every output look directly at every input.',
      details: [
        { label: 'Gradient clipping', text: 'Rescale the gradient when its norm exceeds a threshold — cheap, standard, fixes explosion.' },
        { label: 'Orthogonal init', text: 'A spectral radius near 1 keeps the per-step factor close to 1, slowing the exponential decay.' },
        { label: 'Gating → LSTM', text: 'A gated cell carries the gradient on a near-identity path; the next lab shows exactly how.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'Long-range dependencies are hard', description: 'Because the gradient is a product over time, a vanilla RNN can rarely connect an output to an input many steps earlier — the signal has decayed to nothing.', recommendation: 'Use gated cells (LSTM/GRU) or attention for tasks with long-range structure; reserve plain RNNs for short windows.' },
    { category: 'METHODOLOGY', title: 'Stabilise BPTT', description: 'Exploding gradients (spectral radius > 1) destabilise training and can produce NaNs after a single bad step.', recommendation: 'Clip the global gradient norm, use orthogonal/identity initialisation, and truncate BPTT to a bounded window for very long sequences.' },
  ],
};

export const LSTM_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Gated memory: the LSTM cell',
      body: 'An LSTM augments the hidden state with a separate cell state c that acts as long-term memory, controlled by three gates. The forget gate f decides what to erase, the input gate i decides what to write, and the output gate o decides what to read out: c_t = f⊙c_{t-1} + i⊙g and h_t = o⊙tanh(c_t), where g is the candidate update. Each gate is a sigmoid in (0, 1), so the network LEARNS, per dimension and per step, how much to keep, write, and expose.',
      details: [
        { label: 'Cell state c', text: 'A protected long-term memory updated additively, separate from the exposed hidden state h.' },
        { label: 'Forget / input / output', text: 'Three learned sigmoid gates control erase, write, and read of the cell — soft, differentiable switches.' },
        { label: 'Candidate g', text: 'A tanh proposal for new content; the input gate decides how much of it actually enters the cell.' },
      ],
    },
    {
      heading: 'The constant error carousel',
      body: 'The reason LSTMs learn long-range dependencies is the gradient path along the cell state. Because c_t = f⊙c_{t-1} + i⊙g, the Jacobian of the carry is ∂c_t/∂c_{t-1} ≈ diag(f). When the forget gate f ≈ 1 the gradient is multiplied by ≈ 1 at every step, so it survives across many timesteps instead of decaying like the vanilla RNN — Hochreiter & Schmidhuber called this the "constant error carousel". Overlay the LSTM gradient curve on the RNN one and the difference is stark: a near-flat highway versus an exponential cliff.',
      details: [
        { label: '∂c_t/∂c_{t−1} ≈ diag(f)', text: 'The carry is additive, so the gradient factor per step is the forget gate, not a dense matrix.' },
        { label: 'f ≈ 1 → flat gradient', text: 'A near-1 forget gate gives a factor ≈ 1^k — the gradient highway that beats vanishing.' },
        { label: 'Forget-gate bias', text: 'Initialising the forget bias high (≈ +1 to +2) opens the carousel from the start and speeds learning.' },
      ],
    },
    {
      heading: 'What made long-range sequence learning practical',
      body: 'Gating turns memory into something the network controls rather than something that passively decays. By learning what to keep, write, and read, an LSTM can latch a value early in a sequence and surface it many steps later — copy tasks, language modelling, speech, translation. This was the dominant approach to sequence learning for years, until attention let models read all positions directly. The same gating idea reappears in GRUs (a 2-gate simplification) and, in spirit, in the residual/highway connections of deep nets.',
      details: [
        { label: 'Learned retention', text: 'The cell holds a value as long as the forget gate stays near 1, releasing it when the task demands.' },
        { label: 'GRU', text: 'A lighter gated cell (update + reset gates, no separate cell state) with similar long-range behaviour.' },
        { label: 'Bridge to attention', text: 'Gating mitigates the bottleneck; attention removes it by giving direct access to every encoder state.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'Gates are the memory controller', description: 'Long-range retention depends on the forget gate staying open; if it learns to close too eagerly, the cell still forgets.', recommendation: 'Initialise the forget-gate bias positive so the carousel starts open, and monitor mean gate activations during training.' },
    { category: 'DEPLOYMENT', title: 'Cost vs Transformers', description: 'LSTMs process tokens strictly sequentially, so they cannot parallelise across time the way attention can — a throughput limit on long sequences.', recommendation: 'Use LSTMs/GRUs for streaming or low-latency settings and small data; prefer attention-based models when sequences are long and compute allows.' },
  ],
};

export const SEQ2SEQ_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Encoder → context vector → decoder',
      body: 'A sequence-to-sequence model maps one sequence to another (translation, summarisation, dialogue). An encoder RNN reads the whole input and compresses it into a single fixed-width CONTEXT VECTOR — its final hidden state. A decoder RNN is then initialised from that vector and generates the output one token at a time. The entire meaning of the input must therefore pass through one fixed-size vector, no matter how long the input is.',
      details: [
        { label: 'Encoder', text: 'Reads the input left-to-right and folds it into its last hidden state — the context vector.' },
        { label: 'Context vector', text: 'A single fixed-width summary; the only thing the decoder sees of the input.' },
        { label: 'Decoder', text: 'Generates the output sequence conditioned on the context vector (and its own prior outputs).' },
      ],
    },
    {
      heading: 'The information bottleneck',
      body: 'A fixed d-dimensional vector has finite capacity — on the order of d·(bits per dimension) bits. The input demands roughly L·log2(V) bits for length L over a vocabulary of size V. As the input grows for a fixed context dimension, demand outstrips capacity and information must be dropped. Crucially the context vector is the encoder\'s LAST state, so the EARLY tokens — seen first and overwritten most — fade fastest: per-position reconstruction accuracy decays at the start of long inputs. This lab is an ANALYTIC illustration of that capacity-versus-demand trade-off, not a trained network.',
      details: [
        { label: 'Capacity ≈ d·bits/dim', text: 'A wider context vector holds more, but capacity is fixed once chosen — it cannot grow with the input.' },
        { label: 'Demand ≈ L·log2(V)', text: 'Longer inputs and larger vocabularies carry more bits; eventually they exceed any fixed capacity.' },
        { label: 'Early tokens fade', text: 'Because the summary is the final hidden state, the start of a long sequence is forgotten first.' },
      ],
    },
    {
      heading: 'Attention removes the bottleneck',
      body: 'The fix that reshaped the field: instead of squeezing everything through one vector, ATTENTION lets the decoder read ALL the encoder hidden states and, at each output step, form a weighted combination focused on the relevant input positions. There is no single bottleneck, so long inputs no longer lose their start, and alignment becomes learnable. This is the direct conceptual bridge to the platform\'s LLM / Attention lab — and ultimately to the Transformer, which is attention without any recurrence at all.',
      details: [
        { label: 'Read all states', text: 'The decoder attends over every encoder position, not just the final summary — no fixed bottleneck.' },
        { label: 'Learned alignment', text: 'Attention weights show which input tokens drive each output token; long-range links are direct, not chained.' },
        { label: 'Toward Transformers', text: 'Dropping recurrence and keeping only attention gives the Transformer — see the Attention lab next.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'One vector cannot hold everything', description: 'A fixed-width context vector is a hard capacity limit; translation quality of vanilla seq2seq drops sharply as input length grows.', recommendation: 'Use attention (or a Transformer) so the decoder accesses all encoder states; reserve plain encoder-decoder vectors for short inputs.' },
    { category: 'VERIFICATION', title: 'This is an analytic illustration', description: 'The fidelity curves here come from a capacity-vs-demand model, not from training a real seq2seq network, so treat them as intuition for the bottleneck rather than measured accuracy.', recommendation: 'To see real numbers, train an encoder-decoder with and without attention and compare BLEU/accuracy as input length increases.' },
  ],
};
