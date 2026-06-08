import { LabContent } from '../../catalog/types';

// Co-located theory + lifecycle content for the Stochastic & Bayesian Models
// labs, rendered in each lab's Context tab. Same depth/voice as the other areas.

export const BNN_CONTENT: LabContent = {
  sections: [
    {
      heading: 'From a point estimate to a distribution over models',
      body: 'A standard network learns one set of weights — a single function — and reports a prediction with no sense of its own reliability. A Bayesian neural network instead keeps a distribution over weights: a prior p(w), a posterior p(w|D) ∝ p(D|w)p(w) after data, and a predictive distribution p(y|x,D) = ∫ p(y|x,w)p(w|D) dw that averages over every plausible network. The spread of that predictive distribution is the model’s uncertainty.',
      details: [
        { label: 'Prior → posterior', text: 'Bayes updates a prior over weights into a posterior given the data — the same rule as the Bayes lab, lifted to weight space.' },
        { label: 'Predictive average', text: 'Predictions integrate over all weight settings, so disagreement between them becomes uncertainty.' },
        { label: 'Epistemic vs aleatoric', text: 'Epistemic (model) uncertainty shrinks with data; aleatoric (noise) uncertainty does not.' },
      ],
    },
    {
      heading: 'Why the band widens away from the data',
      body: 'Near training points, every plausible function must pass close to the observations, so the sampled functions agree and the predictive band is tight. In gaps between clusters and beyond the data’s range, many very different functions fit equally well, so the samples fan out and the band balloons. This is the property a point estimate cannot express — and exactly what you want for safe extrapolation, active learning and out-of-distribution detection.',
      details: [
        { label: 'Interpolation', text: 'Where data constrains the fit, functions converge and uncertainty is small.' },
        { label: 'Extrapolation', text: 'Outside the data the posterior reverts toward the prior — wide uncertainty.' },
        { label: 'Honest failure', text: 'A BNN can say “I don’t know here”, which a single network never does.' },
      ],
    },
    {
      heading: 'Practical approximations',
      body: 'The exact posterior is intractable for real networks, so practitioners approximate it. Mean-field variational inference (Bayes-by-Backprop) fits a Gaussian q(w)=N(μ,σ) by maximising the ELBO. MC-Dropout keeps dropout ON at test time and treats each masked forward pass as a posterior sample. Deep ensembles train several networks from different initialisations and use their disagreement. This lab uses a fixed random feature layer with a Bayesian linear output, so all four give the SAME well-defined predictive distribution — and you can watch dropout, ensembling and weight-sampling approximate it.',
      details: [
        { label: 'Variational / Bayes-by-Backprop', text: 'A learned Gaussian over weights; sample weights, average predictions.' },
        { label: 'MC-Dropout', text: 'Dropout at inference ≈ approximate Bayesian inference (Gal & Ghahramani).' },
        { label: 'Deep ensembles', text: 'Several independently-trained nets; cheap, strong uncertainty from disagreement.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'DEPLOYMENT', title: 'Calibrated uncertainty matters', description: 'Over-confident point predictions are dangerous in medicine, finance and control where being wrong silently is costly.', recommendation: 'Report predictive intervals; check calibration; abstain or defer when epistemic uncertainty is high.' },
    { category: 'CONCEPT', title: 'Uncertainty ≠ accuracy', description: 'A model can be confidently wrong if the prior or likelihood is misspecified.', recommendation: 'Validate uncertainty on held-out and out-of-distribution data, not just in-distribution error.' },
  ],
};

export const GP_CONTENT: LabContent = {
  sections: [
    {
      heading: 'A distribution over functions',
      body: 'A Gaussian process is a prior directly over functions: any finite set of points has a joint Gaussian distribution whose covariance is given by a kernel k(x,x′). You never choose weights — you choose how points co-vary. The kernel’s lengthscale sets how quickly the function can wiggle, and its signal variance sets the amplitude. Sampling from the prior draws whole random functions of that character.',
      details: [
        { label: 'Mean + kernel', text: 'A GP is fully specified by a mean function (here 0) and a covariance kernel k(x,x′).' },
        { label: 'Lengthscale ℓ', text: 'Small ℓ → wiggly functions with short-range correlation; large ℓ → smooth.' },
        { label: 'Kernel choice', text: 'RBF is infinitely smooth, Matérn-3/2 is rougher, periodic repeats — encoding prior beliefs.' },
      ],
    },
    {
      heading: 'Conditioning: the posterior in closed form',
      body: 'Observing data conditions the Gaussian. With training inputs X, targets y and noise σ²ₙ, the posterior at test points X∗ is again Gaussian with mean K∗(K+σ²ₙI)⁻¹y and covariance K∗∗ − K∗(K+σ²ₙI)⁻¹K∗ᵀ. No optimisation is needed — just linear algebra. The predictive standard deviation collapses to the noise level at observed points and grows between and beyond them, giving the GP its signature uncertainty band.',
      details: [
        { label: 'Posterior mean', text: 'A smooth interpolation of the data, weighted by kernel similarity.' },
        { label: 'Posterior variance', text: 'Small near data, large in gaps — uncertainty you get for free.' },
        { label: 'Exact inference', text: 'Closed-form for regression — no training loop, unlike a neural network.' },
      ],
    },
    {
      heading: 'Cost, kernels and the link to neural nets',
      body: 'GPs are the gold standard for uncertainty on small data and underpin Bayesian optimisation, but inference costs O(n³) in the number of points because of the matrix inverse, so they need sparse approximations to scale. The kernel encodes every assumption — periodicity, smoothness, trends — and tuning its hyperparameters (lengthscale, variance, noise) by marginal likelihood is the GP’s version of learning. A single-layer neural network with infinitely many random features converges to a GP, the bridge to the Bayesian-NN lab.',
      details: [
        { label: 'O(n³) cost', text: 'The (K+σ²I)⁻¹ inverse limits exact GPs to ~thousands of points.' },
        { label: 'Hyperparameters', text: 'ℓ, σ_f, σ_n are tuned by maximising the marginal likelihood (Occam balance).' },
        { label: 'Wide-net limit', text: 'An infinitely-wide random network is a GP — uncertainty without weights.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'METHODOLOGY', title: 'The kernel is the model', description: 'A mismatched kernel gives confident, wrong extrapolations (e.g. RBF on a periodic signal).', recommendation: 'Choose the kernel from domain knowledge; combine kernels; check posterior samples look plausible.' },
    { category: 'DEPLOYMENT', title: 'Scaling cost', description: 'Exact GP inference is cubic in the data and infeasible for large sets.', recommendation: 'Use sparse / inducing-point approximations or switch to scalable Bayesian deep models.' },
  ],
};

export const HMM_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Hidden state behind noisy observations',
      body: 'A hidden Markov model assumes an unobserved state that evolves as a Markov chain — the next state depends only on the current one — and emits an observation at each step from a state-dependent distribution. You see only the emissions and must infer the hidden states. The classic example: an occasionally-dishonest casino switching between a fair and a loaded die, where you observe the rolls but not which die is in play.',
      details: [
        { label: 'Transition A', text: 'P(state_t | state_{t-1}) — the Markov dynamics of the hidden state.' },
        { label: 'Emission B', text: 'P(observation | state) — how each hidden state generates what you see.' },
        { label: 'Initial π', text: 'The distribution over the first hidden state.' },
      ],
    },
    {
      heading: 'Filtering, smoothing and the most likely path',
      body: 'The forward algorithm computes the filtered belief P(state_t | observations up to t) by alternating a prediction step (multiply by A) with an update step (multiply by the emission likelihood) and renormalising — recursive Bayesian estimation in discrete state space. Combining a forward and a backward pass gives the smoothed posterior P(state_t | the whole sequence). Viterbi, a max-product version of the same recursion, returns the single most likely state path.',
      details: [
        { label: 'Forward (filtering)', text: 'Online belief over the current state as each observation arrives.' },
        { label: 'Forward–backward (smoothing)', text: 'Posterior over each state using past AND future observations.' },
        { label: 'Viterbi', text: 'Dynamic programming for the single most probable hidden-state sequence.' },
      ],
    },
    {
      heading: 'Where HMMs are used',
      body: 'Before deep sequence models, HMMs were the backbone of speech recognition, part-of-speech tagging and gene finding, and they remain the textbook model for noisy time series with discrete latent structure. They are trained by Baum–Welch (the EM algorithm) when the states are unknown. Conceptually they sit between the Bayes lab (recursive updating) and the Sequence-Models area (latent state over time) — a fully probabilistic cousin of the RNN.',
      details: [
        { label: 'Baum–Welch (EM)', text: 'Learns A, B, π from observations alone when the hidden states are unlabelled.' },
        { label: 'Scaling', text: 'Forward probabilities are renormalised each step to avoid numerical underflow.' },
        { label: 'Relation to RNNs', text: 'Both carry latent state through time; the HMM’s state is discrete and its inference exact.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'The Markov assumption', description: 'Assuming the future depends only on the present can be too strong for long-range structure.', recommendation: 'Check the assumption; use higher-order or hierarchical HMMs, or RNNs/transformers, when memory is longer.' },
    { category: 'VERIFICATION', title: 'Filtering vs smoothing', description: 'Online filtering is noisier than the smoothed posterior because it cannot use future evidence.', recommendation: 'Use smoothing for retrospective analysis; reserve filtering for real-time decisions where future data is unavailable.' },
  ],
};
