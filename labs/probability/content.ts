import { LabContent } from '../../catalog/types';

// Co-located theory + lifecycle content for the Probability & Bayesian labs,
// rendered in each lab's Context tab via LabContext.

export const BAYES_CONTENT: LabContent = {
  sections: [
    {
      heading: "Bayes' theorem & posterior beliefs",
      body: 'Bayes’ theorem inverts a conditional: it turns the likelihood P(E|H) — how probable the evidence is if a hypothesis holds — into the posterior P(H|E) we actually care about. The rule is P(H|E) = P(E|H)·P(H) / P(E), where the prior P(H) is our belief before seeing data and P(E) = Σ P(E|Hᵢ)P(Hᵢ) is the total probability of the evidence (the normaliser). The posterior is just the prior re-weighted by how well each hypothesis explains what we saw.',
      details: [
        { label: 'Prior P(H)', text: 'Belief before evidence. In a diagnostic test this is the disease prevalence — the base rate.' },
        { label: 'Likelihood P(E|H)', text: 'How probable the evidence is under the hypothesis. For a test: sensitivity P(+|D) and specificity P(−|¬D).' },
        { label: 'Posterior P(H|E)', text: 'Updated belief after evidence — prior × likelihood, renormalised over all hypotheses.' },
      ],
    },
    {
      heading: 'The base-rate fallacy',
      body: 'A test that is 99% accurate sounds conclusive, yet for a rare disease most positive results are still false alarms. The reason is the base rate: if only 1 in 1000 people is sick, then in a population of 100,000 the ~100 true positives are swamped by the ~1000 false positives drawn from the huge healthy majority. The posterior P(D|+) can be well under 10% even with a very good test — ignoring the prior is the classic error.',
      details: [
        { label: 'Rare disease', text: 'When P(D) is tiny, false positives from the large healthy group dominate the few true positives.' },
        { label: 'Why accuracy misleads', text: 'Headline accuracy hides the split between sensitivity and specificity and ignores prevalence entirely.' },
        { label: 'Fix', text: 'Always carry the prior; report P(D|+) (precision), not just the test’s sensitivity/specificity.' },
      ],
    },
    {
      heading: 'Sequential updating & conjugacy',
      body: 'Beliefs update one observation at a time: today’s posterior is tomorrow’s prior. For a Bernoulli process (a biased coin), a Beta(α, β) prior is conjugate — the posterior is again a Beta, with heads bumping α and tails bumping β. The posterior mean α/(α+β) tracks the empirical success rate, while the distribution tightens as evidence accumulates, shrinking the credible interval around the true rate.',
      details: [
        { label: 'Conjugate prior', text: 'Beta–Bernoulli: heads → α+1, tails → β+1. The posterior stays Beta, so updates are exact and cheap.' },
        { label: 'Posterior mean', text: 'α/(α+β) — a prior-smoothed success rate; the prior acts like pseudo-counts.' },
        { label: 'Credible interval', text: 'A range holding (say) 90% of posterior mass — it narrows as the data grows, unlike a frequentist CI.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'Mind the base rate', description: 'High test accuracy on a rare condition still yields mostly false positives; intuition that ignores the prior is reliably wrong.', recommendation: 'Report precision P(D|+) for the operating prevalence, and re-check it whenever the deployment population’s base rate shifts.' },
    { category: 'METHODOLOGY', title: 'Choosing a prior', description: 'A strong prior dominates when data is scarce and can entrench bias; a too-flat prior wastes prior knowledge.', recommendation: 'Use weakly-informative priors, show prior sensitivity, and let enough data wash the prior out where it matters.' },
  ],
};

export const DISTRIBUTIONS_CONTENT: LabContent = {
  sections: [
    {
      heading: 'PMFs, PDFs, mean & variance',
      body: 'A random variable’s law is described by a probability mass function (discrete — pmf(k) gives an actual probability) or a probability density function (continuous — pdf(x) gives a density that integrates to 1). The mean E[X] is the distribution’s centre of mass and the variance Var(X)=E[(X−μ)²] its spread. Entropy measures uncertainty: it is maximal for a uniform law and shrinks as the distribution concentrates.',
      details: [
        { label: 'PMF vs PDF', text: 'Discrete pmf sums to 1 and reads off probabilities; continuous pdf integrates to 1 and gives densities, not probabilities.' },
        { label: 'Mean & variance', text: 'E[X] is the balance point; Var(X) is the average squared distance from it — the scale of fluctuations.' },
        { label: 'Entropy', text: 'Expected surprise −Σ p log p (or −∫ f log f); larger = flatter/less predictable.' },
      ],
    },
    {
      heading: 'The common families',
      body: 'A handful of families cover most modelling: Bernoulli (a single yes/no), Binomial (count of successes in n trials), Poisson (counts of rare events at a rate λ), Geometric (trials until the first success), Uniform (no preference on an interval), Normal (the bell curve from many small additive effects), Exponential (waiting time between Poisson events), and Beta (a flexible distribution over a probability in [0,1]). Each has a parameter or two that set its location and shape.',
      details: [
        { label: 'Discrete counts', text: 'Bernoulli/Binomial (fixed n), Poisson (rate λ), Geometric (first-success trial index).' },
        { label: 'Continuous', text: 'Uniform (flat), Normal (μ,σ), Exponential (memoryless waits), Beta (shapes α,β on [0,1]).' },
        { label: 'Memorylessness', text: 'Geometric (discrete) and Exponential (continuous) forget the past: P(X>s+t | X>s)=P(X>t).' },
      ],
    },
    {
      heading: 'How the families connect (and the LLN)',
      body: 'The families are a web, not a list. A sum of n Bernoulli(p) trials is Binomial(n,p); as n→∞ with np fixed, Binomial → Poisson(λ=np) (the rare-event limit); and by the Central Limit Theorem both Binomial and Poisson approach a Normal once their counts are large. Drawing samples makes this concrete: by the Law of Large Numbers, an empirical histogram of i.i.d. draws converges to the true pmf/pdf as the sample size grows.',
      details: [
        { label: 'Bernoulli → Binomial', text: 'A Binomial is the sum of n independent Bernoulli trials.' },
        { label: 'Binomial → Poisson', text: 'Many trials, tiny success prob, fixed mean λ=np — the rare-event limit.' },
        { label: 'CLT → Normal', text: 'Sums/averages of many independent pieces become Normal, which is why the bell curve is everywhere.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'DATA', title: 'Pick the family from the mechanism', description: 'Fitting the wrong family (e.g. Normal to a heavy-tailed or count variable) gives biased estimates and bad tail risk.', recommendation: 'Choose the distribution from how the data is generated — counts → Poisson/NegBin, waits → Exponential, bounded rates → Beta — then check with a QQ-plot.' },
    { category: 'CONCEPT', title: 'Finite samples ≠ the true law', description: 'Small samples wobble far from the analytic curve; reading too much into them over-fits noise.', recommendation: 'Lean on the LLN/CLT — collect enough data, report uncertainty, and prefer the parametric form when the mechanism justifies it.' },
  ],
};

export const MCMC_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Why MCMC: sampling intractable targets',
      body: 'We often know a distribution only up to a constant — a Bayesian posterior ∝ prior × likelihood whose normaliser is an unsolvable integral. Markov Chain Monte Carlo builds a random walk whose stationary distribution IS that target, so the states it visits become samples we can average. Because the chain only ever compares densities at two points, the unknown normalising constant cancels and never has to be computed.',
      details: [
        { label: 'Unnormalised target', text: 'We can evaluate π(x) up to a constant but not integrate it — exactly the Bayesian posterior situation.' },
        { label: 'Stationary distribution', text: 'The chain is designed so that, once mixed, the fraction of time at each x matches π(x).' },
        { label: 'Ratios only', text: 'π(x′)/π(x) drops the normaliser — the key trick that makes the method tractable.' },
      ],
    },
    {
      heading: 'The Metropolis–Hastings rule',
      body: 'From the current state x, propose x′ = x + Normal(0, σ). Accept it with probability min(1, π(x′)/π(x)); otherwise stay at x (and still record x as the next sample). Uphill moves to higher density are always taken; downhill moves are taken sometimes, which is what lets the chain explore the whole distribution and cross low-density valleys between modes rather than getting stuck on one peak.',
      details: [
        { label: 'Symmetric proposal', text: 'x′ = x + Normal(0,σ) is symmetric, so the Hastings correction is 1 and accept = min(1, π(x′)/π(x)).' },
        { label: 'Accept / reject', text: 'Always climb; sometimes descend. Rejections repeat the current state — they are real samples, not skips.' },
        { label: 'Detailed balance', text: 'The accept rule enforces π(x)T(x→x′)=π(x′)T(x′→x), which makes π stationary.' },
      ],
    },
    {
      heading: 'Step size, mixing & burn-in',
      body: 'The proposal width σ is the central tuning knob. Too small and almost every proposal is accepted but the chain crawls, exploring slowly with highly correlated samples (poor mixing). Too large and most proposals land in near-zero density and are rejected, so the chain stalls in place. A healthy acceptance rate sits roughly in the 20–50% range. The early samples also depend on where you started, so a burn-in prefix is discarded before averaging.',
      details: [
        { label: 'σ too small', text: 'High acceptance but tiny steps — a slow random walk with strong autocorrelation between samples.' },
        { label: 'σ too large', text: 'Big jumps into low density are mostly rejected; the chain sticks and effective sample size collapses.' },
        { label: 'Burn-in', text: 'Discard the initial transient while the chain forgets its start before using the samples.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'VERIFICATION', title: 'Check convergence, do not assume it', description: 'A chain can look settled yet have missed entire modes; naive averages then misstate the target.', recommendation: 'Run multiple chains from dispersed starts, inspect trace plots and R-hat, and report effective sample size — not just the iteration count.' },
    { category: 'METHODOLOGY', title: 'Tune the proposal', description: 'Acceptance near 0% or 100% both signal a badly-scaled proposal and wasted computation.', recommendation: 'Target ~20–50% acceptance (adapt σ during burn-in), and prefer gradient-based samplers (HMC/NUTS) in higher dimensions.' },
  ],
};
