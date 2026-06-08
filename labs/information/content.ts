import { LabContent } from '../../catalog/types';

// Co-located theory + lifecycle content for the Information Theory labs,
// rendered in each lab's Context tab via LabContext. Connected to ML throughout:
// cross-entropy loss, label smoothing, KL in VI / distillation / RLHF, entropy
// as the compression limit.

export const ENTROPY_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Surprise & entropy',
      body: 'Information theory measures uncertainty. The surprise (or "information content") of an outcome is −log p(x): a near-certain event carries almost no information, while a rare one carries a lot — observing it tells you something. Entropy H(p) = −Σ p(x) log p(x) is just the expected surprise, the average number of bits (log base 2) or nats (natural log) needed to pin down one draw from the distribution.',
      details: [
        { label: 'surprise −log p', text: 'Monotone decreasing in p: p=1 → 0 surprise, p→0 → ∞. Halving a probability adds exactly one bit of surprise.' },
        { label: 'H = E[surprise]', text: 'The probability-weighted average of −log p over all outcomes — the irreducible uncertainty of the source.' },
        { label: 'bits vs nats', text: 'Same quantity, different log base: log₂ gives bits, ln gives nats (H_nats = H_bits·ln 2). ML losses use nats; coding uses bits.' },
      ],
    },
    {
      heading: 'Maximum & minimum entropy',
      body: 'Entropy is maximised by the uniform distribution — when every outcome is equally likely there is nothing to predict, and H = log N (log of the number of outcomes). It falls as the distribution concentrates, reaching exactly 0 when one outcome has probability 1 (a certain source carries no information). Loading the die, or pushing a coin toward heads, always lowers H below its uniform ceiling.',
      details: [
        { label: 'uniform = max', text: 'H = log N for N equally-likely outcomes; the most uncertain, least compressible source.' },
        { label: 'certainty = 0', text: 'If one p=1 and the rest 0, every draw is known in advance, so H = 0 and no bits are needed.' },
        { label: 'fair coin = 1 bit', text: 'Two equiprobable outcomes give exactly 1 bit — the canonical unit of information.' },
      ],
    },
    {
      heading: 'Why entropy underlies ML',
      body: 'Entropy is the lower bound on lossless compression (Shannon): you cannot encode a source in fewer than H bits/symbol on average. In ML it is everywhere — the entropy of a model\'s predictive distribution measures its confidence, maximum-entropy modelling justifies softmax, and an "entropy bonus" in RL/policy-gradient methods rewards keeping the policy uncertain so the agent keeps exploring.',
      details: [
        { label: 'compression bound', text: 'No code beats H bits/symbol on average — entropy is the fundamental limit (see the Source Coding lab).' },
        { label: 'predictive entropy', text: 'A low-entropy softmax output = a confident prediction; high entropy = the model is unsure. Used for active learning and OOD detection.' },
        { label: 'entropy bonus (RL)', text: 'Adding +β·H(π) to the objective discourages premature collapse to a deterministic policy, sustaining exploration.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'Units matter', description: 'Bits and nats differ by a factor of ln 2; mixing them silently scales every entropy, cross-entropy and KL value.', recommendation: 'Fix one base for a project — deep-learning losses are in nats (natural log); state the base whenever you report an entropy.' },
    { category: 'DATA', title: 'Estimating entropy from samples', description: 'With finite data the plug-in estimate from observed frequencies is biased low — rare outcomes you never saw look impossible.', recommendation: 'Use bias-corrected estimators (Miller–Madow) or smoothing for small samples; the sampled average surprise only approaches H as n grows.' },
  ],
};

export const KL_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Cross-entropy = the classification loss',
      body: 'Given a true distribution p and a model distribution q over the same outcomes, the cross-entropy H(p,q) = −Σ p log q is the average number of bits needed to encode draws from p using a code built for q. When p is a one-hot label and q is the softmax output, H(p,q) = −log q(true class) — exactly the categorical cross-entropy loss that trains virtually every classifier. Minimising it w.r.t. the model is maximum-likelihood estimation.',
      details: [
        { label: 'H(p,q) = −Σ p log q', text: 'Coding p with q\'s code. Always ≥ H(p), with equality only when q = p.' },
        { label: 'one-hot p', text: 'For a hard label, cross-entropy collapses to −log q(y) — the negative log-likelihood of the correct class.' },
        { label: 'MLE link', text: 'Minimising cross-entropy over data = maximising likelihood = matching q to the empirical p.' },
      ],
    },
    {
      heading: 'KL divergence & its asymmetry',
      body: 'The Kullback–Leibler divergence KL(p‖q) = Σ p log(p/q) = H(p,q) − H(p) is the EXTRA cost of using q instead of the true p — the gap between cross-entropy and the irreducible floor H(p). It is ≥ 0 (Gibbs\' inequality) and zero only when q = p, but it is NOT symmetric: KL(p‖q) ≠ KL(q‖p). Forward KL (p‖q) is mass-covering — q must put probability everywhere p does — while reverse KL (q‖p) is mode-seeking, letting q ignore minor modes. The direction you pick changes the answer.',
      details: [
        { label: 'KL = H(p,q) − H(p)', text: 'Cross-entropy minus entropy: the avoidable bits. Training drives this to 0 even though H(p) cannot be reduced.' },
        { label: 'KL ≥ 0', text: 'Gibbs\' inequality: q can never beat the true distribution\'s own code. Equality iff q = p everywhere.' },
        { label: 'forward vs reverse', text: 'Forward (p‖q) spreads q to cover all of p; reverse (q‖p), used in variational inference, picks a single mode.' },
      ],
    },
    {
      heading: 'KL across modern ML',
      body: 'KL divergence is a workhorse far beyond classification. Variational inference and VAEs add a KL(q‖prior) regulariser to keep the learned posterior close to a prior. Knowledge distillation trains a student to match a teacher\'s soft distribution via KL. RLHF / PPO add a KL penalty to keep a fine-tuned policy near the reference model so it does not drift. And label smoothing replaces the hard one-hot p with a softened target, which is exactly cross-entropy against a mixed distribution — trading a little bias for better calibration.',
      details: [
        { label: 'VI / VAE', text: 'ELBO = reconstruction − KL(q(z|x)‖p(z)); the KL term regularises the latent posterior toward the prior.' },
        { label: 'distillation', text: 'Student minimises KL to the teacher\'s temperature-softened logits, transferring "dark knowledge" between classes.' },
        { label: 'RLHF KL penalty', text: 'PPO subtracts β·KL(π‖π_ref) so the aligned model stays close to the pretrained reference and avoids reward hacking.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'METHODOLOGY', title: 'Pick the right KL direction', description: 'Forward and reverse KL give different optima — mass-covering vs mode-seeking — so the wrong direction yields a model that under- or over-spreads.', recommendation: 'Use forward KL / cross-entropy for supervised fitting to data; use reverse KL when approximating an intractable posterior (VI), and state which you mean.' },
    { category: 'VERIFICATION', title: 'Numerical floor under log q', description: 'Cross-entropy and KL blow up when q assigns ~0 probability to an outcome that p has — a single log 0 makes the loss infinite or NaN.', recommendation: 'Add a small ε inside the log or work in log-space (log-softmax); label smoothing also keeps every q strictly positive.' },
  ],
};

export const SOURCE_CODING_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Prefix codes & Huffman',
      body: 'A prefix (instantaneous) code assigns each symbol a binary string such that no codeword is a prefix of another, so a stream decodes uniquely with no separators. Huffman\'s algorithm builds the optimal one: repeatedly merge the two least-probable nodes into a parent until a single tree remains, then read 0/1 off the edges to each leaf. Frequent symbols sit near the root and get short codes; rare symbols sit deep and get long ones.',
      details: [
        { label: 'prefix-free', text: 'No codeword prefixes another → instantaneous decoding, and Kraft\'s inequality Σ 2^(−lᵢ) ≤ 1 holds.' },
        { label: 'greedy merge', text: 'Combine the two smallest probabilities each step; this greedy choice is provably optimal for symbol codes.' },
        { label: 'variable length', text: 'Common symbols get fewer bits, rare ones more — the average shrinks below a fixed-length code.' },
      ],
    },
    {
      heading: 'The entropy bound: H ≤ L < H+1',
      body: 'The average code length L = Σ pᵢ·lᵢ cannot go below the entropy H(p) — Shannon\'s source-coding theorem. Huffman is optimal among prefix codes and provably satisfies H ≤ L < H + 1: it is within one bit of the limit per symbol. The efficiency H/L tells you how close you are; it equals 1 exactly when every probability is a power of ½ (so codeword lengths −log₂ pᵢ are integers). The slack vs a fixed-length code, ⌈log₂ N⌉ bits/symbol, is the compression you gain by exploiting a skewed distribution.',
      details: [
        { label: 'L ≥ H', text: 'No uniquely-decodable code averages fewer than H bits/symbol — entropy is the fundamental compression limit.' },
        { label: 'within 1 bit', text: 'Huffman achieves L < H + 1; the gap comes from rounding ideal lengths −log₂ pᵢ up to integers.' },
        { label: 'block coding', text: 'Coding blocks of k symbols at once drives L/k → H; arithmetic coding reaches the limit without integer-length waste.' },
      ],
    },
    {
      heading: 'From coding to ML',
      body: 'Source coding is the operational meaning of entropy, and it ties directly back to learning. The cross-entropy loss is literally the extra bits you pay to encode the data using the model\'s distribution instead of the true one — the "minimum description length" view casts learning as compression. Modern neural compressors (and the bits-back / latent-variable codes behind them) replace the Huffman table with a learned probability model and arithmetic coding, but the bound L ≥ H is the same Shannon limit.',
      details: [
        { label: 'MDL', text: 'Minimum Description Length: the best model is the one that compresses the data most — generalisation as compression.' },
        { label: 'cross-entropy = bits', text: 'Training loss H(p,q) is the average bits/symbol a model-based code would spend; lowering it = better compression.' },
        { label: 'arithmetic coding', text: 'Reaches the entropy limit fractionally (no integer-length waste) and pairs with learned models for state-of-the-art compression.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'Entropy is the floor, not a target to beat', description: 'Teams sometimes expect a coder to drop below H; that is impossible for a lossless code on a known source — only changing the model (better probabilities) lowers H itself.', recommendation: 'To compress further, improve the probability model (context, larger blocks) rather than the coder; measure efficiency H/L to see remaining headroom.' },
    { category: 'DEPLOYMENT', title: 'Distribution drift breaks the code', description: 'A Huffman table built on one distribution is suboptimal — sometimes expansive — if the real symbol frequencies differ, e.g. English-tuned codes on other text.', recommendation: 'Use adaptive / online models that update frequencies as data streams, or retrain the code per source; monitor the realised L against H in production.' },
  ],
};
