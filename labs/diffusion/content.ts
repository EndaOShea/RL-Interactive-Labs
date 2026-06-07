import { LabContent } from '../../catalog/types';

// Co-located theory + lifecycle content for the Diffusion labs (rendered in each
// lab's Context tab via LabContext).

export const FORWARD_REVERSE_CONTENT: LabContent = {
  sections: [
    {
      heading: 'The forward process is a fixed noising chain',
      body: 'Diffusion models define a forward process q that gradually corrupts data into noise over T steps. Each step adds a little Gaussian noise: q(xₜ | xₜ₋₁) = N(√(1−βₜ)·xₜ₋₁, βₜ·I). This Markov chain has NO learned parameters — the βₜ schedule is fixed in advance. Run enough steps and any structured distribution dissolves into a standard Gaussian.',
      details: [
        { label: 'Markov chain', text: 'xₜ depends only on xₜ₋₁. Noise accumulates monotonically; structure is destroyed step by step.' },
        { label: 'βₜ (beta)', text: 'The per-step variance. Small βₜ early (gentle), larger later. The schedule sets the pace of destruction.' },
        { label: 'Endpoint', text: 'After T steps x_T ≈ N(0, I) — pure noise, independent of the data. The cloud here collapses to a blob.' },
      ],
    },
    {
      heading: 'The reparameterised marginal lets us jump to any t',
      body: 'Because each step is Gaussian, the chain has a closed-form marginal: xₜ = √(ᾱₜ)·x₀ + √(1−ᾱₜ)·ε, with ᾱₜ = ∏ᵢ₌₁ᵗ (1−βᵢ) and ε ~ N(0,I). We never simulate the chain step-by-step — we sample t and noise the clean sample x₀ directly. This lab cheats by storing each point\'s x₀ and a FIXED ε, so scrubbing t shows the EXACT marginals (forward dissolve, then reverse re-formation) without training anything.',
      details: [
        { label: 'ᾱₜ (alpha-bar)', text: 'Cumulative signal-retention. ᾱ₀≈1 (all signal), ᾱ_T≈0 (all noise). √(ᾱₜ) scales the data, √(1−ᾱₜ) scales the noise.' },
        { label: 'Reparam trick', text: 'Writing xₜ as a deterministic function of x₀ and ε makes the marginal sampleable and the loss differentiable.' },
        { label: 'Learned reverse', text: 'A real model has no x₀ at sample time. It trains a net εθ(xₜ,t) to PREDICT the noise, then steps xₜ→xₜ₋₁. Here we reuse the stored ε instead — analytic, not learned.' },
      ],
    },
    {
      heading: 'Samplers & guidance shape the reverse pass',
      body: 'The forward chain is fixed, but you choose HOW to run it backward. DDPM samples ancestrally (stochastic, many steps); DDIM follows a deterministic ODE that hits the same marginals in a handful of strided steps. Classifier-free guidance then steers each step toward the conditioning by extrapolating ε̂ = (1+w)·ε_cond − w·ε_uncond — trading diversity for fidelity. This lab models all three: switch sampler, set the DDIM stride count, and dial the guidance scale w.',
      details: [
        { label: 'DDIM stride', text: 'With S DDIM steps each move skips ≈T/S chain-steps. Predict x₀̂, then re-noise straight to t−1 — deterministic, so the trajectory is a smooth ODE path.' },
        { label: 'w · CFG scale', text: 'w=0 is unguided. Larger w pulls samples harder onto their class — here the reverse cloud contracts toward each cluster centroid in proportion to w·√ᾱₜ.' },
        { label: 'Diversity trade', text: 'Few DDIM steps + high w = fast, crisp, but mode-collapsed. Many DDPM steps + low w = slow, diverse, softer. The right operating point depends on the use case.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'ε-prediction vs score', description: 'The network usually predicts the added noise ε, which is equivalent (up to scaling) to the score ∇ₓ log q(xₜ) — the direction toward higher data density.', recommendation: 'Think of reverse sampling as repeatedly nudging the sample up the data-density gradient while removing a bit of noise each step.' },
    { category: 'DEPLOYMENT', title: 'DDPM vs DDIM sampling', description: 'DDPM is ancestral/stochastic and needs many steps (~1000): each reverse step samples xₜ₋₁ ~ N(μθ, σₜ²) with fresh noise. DDIM is a DETERMINISTIC re-interpretation of the SAME marginals — it predicts x₀̂ = (xₜ−√(1−ᾱₜ)ε̂)/√ᾱₜ then re-noises to t−1 with no added randomness, so 20–50 strided steps match 1000-step DDPM quality.', recommendation: 'Use DDIM (or a higher-order ODE solver like DPM-Solver) when latency matters; reserve full DDPM when you want maximum sample diversity. Toggle the sampler and watch the same cloud re-form in far fewer DDIM strides.' },
    { category: 'METHODOLOGY', title: 'Classifier-free guidance (CFG)', description: 'Train one network on both conditional and unconditional inputs (dropping the label sometimes). At sample time combine them: ε̂ = (1+w)·ε_cond − w·ε_uncond. The guidance scale w extrapolates the prediction away from the unconditional, sharpening class identity.', recommendation: 'Raise w for crisper, more on-prompt samples; but high w trades away diversity and can over-saturate. Sweep w here — the reverse cloud tightens onto its class centroids as w grows.' },
  ],
};

export const NOISE_SCHEDULE_CONTENT: LabContent = {
  sections: [
    {
      heading: 'The schedule controls how fast information dies',
      body: 'The noise schedule {βₜ} (or equivalently {ᾱₜ}) decides how quickly the forward process destroys signal. A linear β schedule ramps variance uniformly; a cosine schedule keeps ᾱₜ near 1 for longer, then drops it smoothly to 0. The curve of ᾱₜ over t is the single most important design choice in a diffusion model.',
      details: [
        { label: 'β linear', text: 'βₜ rises linearly from ~1e-4 to ~0.02. Simple, but destroys structure quickly near the end and wastes steps where the image is already near-noise.' },
        { label: 'ᾱₜ cosine', text: 'Defined via ᾱₜ = cos²((t/T+s)/(1+s)·π/2). Information is removed more evenly across t, so more steps do useful work.' },
        { label: 'αₜ', text: 'αₜ = 1−βₜ is the per-step signal-retention; ᾱₜ is its cumulative product.' },
      ],
    },
    {
      heading: 'Signal-to-noise ratio (SNR) ties it together',
      body: 'SNR(t) = ᾱₜ / (1−ᾱₜ) measures how much clean signal survives relative to noise at step t. It starts huge (pure signal), crosses 1 somewhere in the middle, and approaches 0 at t=T. Plotting log-SNR makes the schedule\'s behaviour legible: the loss weighting and the difficulty of denoising at each t both track SNR.',
      details: [
        { label: 'SNR = 1', text: 'The crossover where signal and noise have equal power — the hardest region to denoise.' },
        { label: 'log-SNR', text: 'Many modern formulations parameterise the schedule directly in log-SNR space, which decouples it from a fixed T.' },
        { label: 'Loss weighting', text: 'How much each t contributes to the training loss is effectively a function of SNR; reweighting it changes sample quality.' },
      ],
    },
    {
      heading: 'Sigmoid / EDM schedules and the resolution shift',
      body: 'Beyond linear and cosine, a sigmoid schedule defines ᾱₜ = σ(logSNRₜ) with log-SNR swept linearly from high to low — symmetric in log-SNR, so it packs steps around the SNR≈1 crossover where denoising is hardest (the EDM family of schedules behaves this way). Separately, the resolution-shift trick rescales the whole schedule in log-SNR: SNR′ = SNR·shift², i.e. log-SNR += 2·ln(shift). Bigger images need more signal retained per step, so a shift>1 (used by SD-XL, simple-diffusion) keeps ᾱ higher for longer. This lab adds the sigmoid schedule and a live shift slider.',
      details: [
        { label: 'σ(logSNR)', text: 'ᾱ = σ(logSNR) = SNR/(1+SNR). Sweeping log-SNR linearly makes the schedule symmetric and steerable by just its endpoints.' },
        { label: 'Shift = 2·ln(s)', text: 'A constant offset in log-SNR space. shift>1 moves every curve up — more signal at every t; shift<1 destroys structure earlier.' },
        { label: 'Why resolution matters', text: 'Independent pixels average out noise, so high-res images look "less noisy" at the same ᾱ. Shifting the schedule compensates so denoising stays well-posed.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'METHODOLOGY', title: 'Why cosine beats linear', description: 'Linear schedules at common resolutions push ᾱₜ to near-zero too early, so the final steps add noise to what is already noise — wasted capacity.', recommendation: 'Prefer a cosine (or learned/EDM-style) schedule so the model spends steps where denoising is actually informative.' },
    { category: 'VERIFICATION', title: 'Steps vs quality trade-off', description: 'Fewer sampling steps approximate the reverse SDE/ODE more coarsely; too few and samples blur or lose detail.', recommendation: 'Sweep the step count against a quality metric (FID) and pick the knee; pair an efficient schedule with a fast solver.' },
    { category: 'METHODOLOGY', title: 'Shift the schedule with resolution', description: 'The same {βₜ} that works at 64×64 is too aggressive at 512×512: independent pixels make high-res images effectively less noisy at a given ᾱ, so the model wastes steps. Modern systems shift log-SNR by 2·ln(shift) instead of redesigning the schedule.', recommendation: 'Scale the log-SNR shift with image resolution (roughly log-linear); verify with the shift slider that ᾱ stays high enough through the chain.' },
  ],
};
