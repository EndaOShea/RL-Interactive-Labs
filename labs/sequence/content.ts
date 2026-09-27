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
      body: 'Training uses Backpropagation Through Time (BPTT): the loss gradient is chained backward across every timestep. The Jacobian that carries a gradient from step t back to step t−k is a PRODUCT, ∂h_t/∂h_{t−k} = Π diag(1 − h_j²)·W_hh, one factor per step (tanh′ = 1 − h²). Its size obeys ‖J_k‖ ≤ Π max tanh′ · ‖W_hh‖₂^k, so a largest singular value ‖W_hh‖₂ ≤ 1 guarantees the gradient can only shrink; ‖W_hh‖₂ > 1 is needed for it to EXPLODE but does not guarantee it — saturated units (small tanh′) can still make it VANISH. In this lab W_hh = ρ·Q with Q orthogonal, so ρ is at once the spectral radius and ‖W_hh‖₂, and the exact product is recomputed from the stored hidden states after every step.',
      details: [
        { label: 'Product of Jacobians', text: 'The lab plots the exact ‖∂h_t/∂h_{t−k}‖ against the lag and, dashed, the bound Π max tanh′·ρᵏ.' },
        { label: 'Vanishing', text: 'Per-step factor ‖J_K‖^(1/K) below 0.9: early timesteps receive almost no signal, so long-range dependencies are never learned.' },
        { label: 'Exploding', text: 'Per-step factor above 1.1: the gradient blows up and training destabilises. Gradient clipping caps the norm to keep steps sane.' },
      ],
    },
    {
      heading: 'Why this matters & how it is mitigated',
      body: 'The vanishing-gradient problem is the recurrent version of the depth problem the ResNet lab tackles with skip connections. Here the fixes are: clip the gradient norm to tame explosions, initialise W_hh orthogonally so the matrix itself neither shrinks nor stretches any direction, and — most importantly — add gating (LSTM/GRU) that gives the gradient a near-identity path through time. Orthogonal initialisation alone is not enough: at ρ = 1 the lab\'s gradient still vanishes over the sequence, because tanh′ < 1 at every step. Attention later removes the bottleneck entirely by letting every output look directly at every input.',
      details: [
        { label: 'Gradient clipping', text: 'Rescale the gradient when its norm exceeds a threshold — cheap, standard, fixes explosion.' },
        { label: 'Orthogonal init', text: 'Every singular value of an orthogonal matrix is 1, so the only shrinkage left is tanh′; near-critical needs tanh′·ρ ≈ 1.' },
        { label: 'Gating → LSTM', text: 'A gated cell carries the gradient on a near-identity path; the next lab shows exactly how.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'Long-range dependencies are hard', description: 'Because the gradient is a product over time, a vanilla RNN can rarely connect an output to an input many steps earlier — the signal has decayed to nothing.', recommendation: 'Use gated cells (LSTM/GRU) or attention for tasks with long-range structure; reserve plain RNNs for short windows.' },
    { category: 'METHODOLOGY', title: 'Stabilise BPTT', description: 'Exploding gradients (a per-step factor above 1, which needs ‖W_hh‖₂ > 1) destabilise training and can produce NaNs after a single bad step.', recommendation: 'Clip the global gradient norm, use orthogonal/identity initialisation, and truncate BPTT to a bounded window for very long sequences.' },
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
      body: 'The reason LSTMs learn long-range dependencies is the gradient path along the cell state. Because c_t = f⊙c_{t-1} + i⊙g, the Jacobian along the carry itself is ∂c_t/∂c_{t-1} = diag(f) (further paths run through h and the gates). When the forget gate f ≈ 1 the gradient is multiplied by ≈ 1 at every step, so it survives across many timesteps instead of decaying like the vanilla RNN — Hochreiter & Schmidhuber called this the "constant error carousel". The lab multiplies the real forget gates along the carry and overlays the exact gradient of a vanilla RNN (orthogonal W_hh, ρ = 1) fed the same input: with f near 1 the carry path stays near 1, but with f ≈ 0.73 (bias +1) it falls below even that RNN.',
      details: [
        { label: '∂c_t/∂c_{t−1} = diag(f)', text: 'Along the carry the gradient factor per step is the forget gate itself, not a dense matrix; over k steps it is Π diag(f).' },
        { label: 'f ≈ 1 → flat gradient', text: 'A near-1 forget gate gives a factor ≈ 1 per step — the gradient highway that beats vanishing.' },
        { label: 'Forget-gate bias', text: 'A positive forget bias (commonly +1) starts f above ½ so the carousel starts partly open; training then pushes f toward 1 where memory is needed. In this untrained lab the bias is the only control: +3 gives a mean f ≈ 0.95, +4 ≈ 0.98.' },
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
    { category: 'CONCEPT', title: 'Gates are the memory controller', description: 'Long-range retention depends on the forget gate staying open; if it learns to close too eagerly, the cell still forgets.', recommendation: 'Initialise the forget-gate bias positive (commonly +1) so the carousel starts partly open, and monitor mean gate activations during training.' },
    { category: 'DEPLOYMENT', title: 'Cost vs Transformers', description: 'LSTMs process tokens strictly sequentially, so they cannot parallelise across time the way attention can — a throughput limit on long sequences.', recommendation: 'Use LSTMs/GRUs for streaming or low-latency settings and small data; prefer attention-based models when sequences are long and compute allows.' },
  ],
};

export const SEQ2SEQ_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Encoder → context vector → decoder',
      body: 'A sequence-to-sequence model maps one sequence to another (translation, summarisation, dialogue). An encoder RNN reads the whole input and compresses it into a single fixed-width CONTEXT VECTOR — its final hidden state. A decoder RNN is then initialised from that vector and generates the output one token at a time. The entire meaning of the input must therefore pass through one fixed-size vector, no matter how long the input is. To measure exactly what that vector holds, this lab replaces the decoder with a linear readout per output position.',
      details: [
        { label: 'Encoder', text: 'Reads the input left-to-right and folds it into its last hidden state — the context vector.' },
        { label: 'Context vector', text: 'A single fixed-width summary; the only thing the decoder sees of the input.' },
        { label: 'Decoder', text: 'Generates the output sequence conditioned on the context vector (and its own prior outputs).' },
      ],
    },
    {
      heading: 'The information bottleneck',
      body: 'A fixed d-dimensional vector must hold all L tokens. For a LINEAR readout of one-hot tokens over a vocabulary of V, each token needs roughly V − 1 independent directions, so d dimensions leave room for only a few tokens and the rest interfere. Crucially the context vector is the encoder\'s LAST state, so the EARLY tokens — whose trace is multiplied by diag(1 − h²)·W_hh at every later step — fade first. This lab measures it: a seeded, untrained encoder (V = 4), and a ridge linear probe per position fitted on 300 random sequences and scored on 200 held-out ones. At d = 4 the probe recovers about 1.7 tokens\' worth from a 12-token input; at d = 12, about 3.7 of 18.',
      details: [
        { label: 'Linear room ≈ d/(V−1) tokens', text: 'A wider context holds more, but its width is fixed once chosen — it cannot grow with the input.' },
        { label: 'Demand grows with L', text: 'Every extra token needs its own directions; eventually the input exceeds any fixed width.' },
        { label: 'Early tokens fade', text: 'Because the summary is the final hidden state, ‖∂h_L/∂x_p‖ shrinks with distance from the end and the start is forgotten first.' },
      ],
    },
    {
      heading: 'Attention removes the bottleneck',
      body: 'The fix that reshaped the field: instead of squeezing everything through one vector, ATTENTION lets the decoder read ALL the encoder hidden states and, at each output step, form a weighted combination focused on the relevant input positions. There is no single bottleneck, so long inputs no longer lose their start, and alignment becomes learnable. The lab\'s dashed curve reads each position from its own state h_p — attention with the ideal alignment — and fits the same probe. This is the direct conceptual bridge to the platform\'s LLM / Attention lab — and ultimately to the Transformer, which is attention without any recurrence at all.',
      details: [
        { label: 'Read all states', text: 'The decoder attends over every encoder position, not just the final summary — no fixed bottleneck.' },
        { label: 'Learned alignment', text: 'Attention weights show which input tokens drive each output token; long-range links are direct, not chained.' },
        { label: 'Toward Transformers', text: 'Dropping recurrence and keeping only attention gives the Transformer — see the Attention lab next.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'One vector cannot hold everything', description: 'A fixed-width context vector is a hard capacity limit; translation quality of vanilla seq2seq drops sharply as input length grows.', recommendation: 'Use attention (or a Transformer) so the decoder accesses all encoder states; reserve plain encoder-decoder vectors for short inputs.' },
    { category: 'VERIFICATION', title: 'What this lab measures', description: 'The curves are measured held-out accuracies of a linear probe on an untrained, seeded encoder. A trained encoder packs information better and a non-linear decoder can read more, so the numbers show what the vector linearly retains, not a trained seq2seq model\'s accuracy.', recommendation: 'To see trained numbers, train an encoder-decoder with and without attention and compare BLEU/accuracy as input length increases.' },
  ],
};
