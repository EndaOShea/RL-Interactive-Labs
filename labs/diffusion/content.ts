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
      body: 'Because each step is Gaussian, the chain has a closed-form marginal: xₜ = √(ᾱₜ)·x₀ + √(1−ᾱₜ)·ε, with ᾱₜ = ∏ᵢ₌₁ᵗ (1−βᵢ) and ε ~ N(0,I). We never simulate the chain step-by-step — we sample t and noise the clean sample x₀ directly. The forward view draws every frame from this marginal with ONE fixed ε per point, so each frame is an exact sample of q(xₜ); scrubbing t rescales the same noise, so consecutive frames are not one Markov sample path.',
      details: [
        { label: 'ᾱₜ (alpha-bar)', text: 'Cumulative signal-retention. ᾱ₀ = 1 (all signal), ᾱ_T ≈ 0 (all noise). √(ᾱₜ) scales the data, √(1−ᾱₜ) scales the noise.' },
        { label: 'Reparam trick', text: 'Writing xₜ as a deterministic function of x₀ and ε makes the marginal sampleable and the loss differentiable.' },
        { label: 'The denoiser here', text: 'A real model has no x₀ at sample time; it trains a net εθ(xₜ,t) to PREDICT the noise. This lab uses the exact answer instead: taking the data distribution to be a Gaussian kernel density over the training points (bandwidth s = 0.05), E[x₀|xₜ] has a closed form — a softmax-weighted blend of the points — and ε̂ = (xₜ − √ᾱₜ·E[x₀|xₜ])/√(1−ᾱₜ). A perfectly trained network would converge to this; nothing is trained.' },
      ],
    },
    {
      heading: 'Samplers & guidance shape the reverse pass',
      body: 'The reverse pass starts from FRESH N(0, I) noise — not the forward endpoint — and generates new samples. DDPM samples ancestrally over all T steps: each removes the predicted noise and adds fresh noise σₜ·z, with σₜ² = β̃ₜ = (1−ᾱₜ₋₁)/(1−ᾱₜ)·βₜ. DDIM (η = 0) is deterministic and visits only S steps, t = round(k·T/S): it predicts x̂₀, then jumps straight to the next visited step. Classifier-free guidance steers every step with ε̂ = (1+w)·ε_cond − w·ε_uncond, where ε_cond comes from the sample\'s own class and ε_uncond from all the data. The grey dots are the training data; the off-data and coverage-gap numbers score the samples against it.',
      details: [
        { label: 'DDIM stride', text: 'With S steps each move skips ≈T/S chain steps: predict x̂₀, then re-noise straight to the next visited step t′ ≈ t − T/S. It is deterministic, so the same x_T always gives the same sample. The last step always lands on x̂₀; with too few steps it does so from a still-noisy t, where the posterior mean averages across nearby data: the samples thin onto the data\'s centre-lines and the coverage gap grows.' },
        { label: 'w · CFG scale', text: 'w = 0 is plain conditional sampling. Larger w extrapolates away from the unconditional prediction, which here pushes each sample away from the other classes: tighter clusters that avoid the regions near other classes, a larger coverage gap and, at high w, samples beyond the data.' },
        { label: 'Diversity trade', text: 'Few DDIM steps and high w are fast and class-typical but cover less of the data; enough steps and low w reproduce the whole density. Compare the off-data and coverage-gap numbers with the exact-KDE-draw reference shown alongside.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'ε-prediction vs score', description: 'The network usually predicts the added noise ε, which is equivalent (up to scaling) to the score ∇ₓ log q(xₜ) — the direction toward higher data density.', recommendation: 'Think of reverse sampling as repeatedly nudging the sample up the data-density gradient while removing a bit of noise each step.' },
    { category: 'DEPLOYMENT', title: 'DDPM vs DDIM sampling', description: 'DDPM is ancestral: each reverse step samples xₜ₋₁ ~ N(μθ, σₜ²) with fresh noise, and the original recipe walks all T (~1000) steps. DDIM is a deterministic sampler for the same trained model: it predicts x̂₀ = (xₜ−√(1−ᾱₜ)ε̂)/√ᾱₜ and re-noises to the next visited step with no added randomness, so it can skip most of the chain — the DDIM paper reports near-full-chain quality 10–50× faster.', recommendation: 'Use DDIM (or a higher-order ODE solver like DPM-Solver) when latency matters. Here, compare the coverage gap: DDIM with 20–30 steps comes close to DDPM\'s T steps, while 4–8 steps visibly thin the samples.' },
    { category: 'METHODOLOGY', title: 'Classifier-free guidance (CFG)', description: 'Train one network on both conditional and unconditional inputs (dropping the label sometimes). At sample time combine them: ε̂ = (1+w)·ε_cond − w·ε_uncond. The guidance scale w extrapolates the prediction away from the unconditional, sharpening class identity.', recommendation: 'Raise w for more class-typical samples, at the cost of diversity. Sweep w here and watch the coverage gap: samples crowd away from the other classes as w grows (on the Guided blobs preset the gap is several times the exact-draw value).' },
  ],
};

export const NOISE_SCHEDULE_CONTENT: LabContent = {
  sections: [
    {
      heading: 'The schedule controls how fast information dies',
      body: 'The noise schedule {βₜ} (or equivalently {ᾱₜ}) decides how quickly the forward process destroys signal. A linear β schedule ramps the per-step variance uniformly; a cosine schedule keeps ᾱₜ near 1 for longer, then drops it smoothly to 0. Every schedule here is built the same way: a curve ᾱ(t), then βₜ = min(0.999, 1 − ᾱₜ/ᾱₜ₋₁) and ᾱₜ = ∏(1−βᵢ), so β, α and ᾱ always agree.',
      details: [
        { label: 'β linear', text: 'DDPM\'s recipe raises βₜ linearly from 1e-4 to 0.02 over T = 1000 steps; here both ends are scaled by 1000/T (0.1/T to 20/T) so every T discretises the same curve. ᾱₜ collapses early: about a third of the steps have ᾱₜ < 0.01, where almost no signal is left to destroy.' },
        { label: 'ᾱₜ cosine', text: 'ᾱₜ = f(t)/f(0) with f(t) = cos²((t/T+s)/(1+s)·π/2), s = 0.008. Information is removed more evenly: SNR = 1 falls at mid-chain and only about 7% of the steps have ᾱₜ < 0.01.' },
        { label: 'αₜ', text: 'αₜ = 1−βₜ is the per-step signal-retention; ᾱₜ is its cumulative product.' },
      ],
    },
    {
      heading: 'Signal-to-noise ratio (SNR) ties it together',
      body: 'SNR(t) = ᾱₜ / (1−ᾱₜ) measures how much clean signal survives relative to noise at step t. It starts huge (pure signal), crosses 1 somewhere in the middle, and approaches 0 at t=T. Plotting log-SNR makes the schedule\'s behaviour legible: the loss weighting and the difficulty of denoising at each t both track SNR.',
      details: [
        { label: 'SNR = 1', text: 'The crossover where signal and noise have equal power (ᾱ = 0.5) — the transition between a recognisable sample and noise.' },
        { label: 'log-SNR', text: 'Many modern formulations parameterise the schedule directly in log-SNR space, which decouples it from a fixed T.' },
        { label: 'Loss weighting', text: 'How much each t contributes to the training loss is effectively a function of SNR; reweighting it changes sample quality.' },
      ],
    },
    {
      heading: 'Sigmoid, EDM and the resolution shift',
      body: 'Two more families. Chen (2023)\'s sigmoid schedule, ᾱ(u) = (σ(end/τ) − σ((u·(end−start)+start)/τ)) / (σ(end/τ) − σ(start/τ)) with u = t/T, is normalised to run from 1 to 0; with start −3, end 3, τ = 1 (used here) it tracks cosine closely. The EDM noise levels of Karras et al. (2022) space σ from 0.002 to 80 as (σ_min^{1/ρ} + u·(σ_max^{1/ρ} − σ_min^{1/ρ}))^ρ with ρ = 7, mapped here to ᾱ = 1/(1+σ²): about a fifth of the steps sit at ᾱ > 0.99, a third at ᾱ < 0.01, and only about 10% near SNR = 1. Separately, the resolution shift rescales any schedule in log-SNR: SNR′ = SNR/shift², i.e. ln SNR − 2·ln(shift). shift > 1 adds noise at every t, as higher resolutions need — simple diffusion (Hoogeboom et al., 2023) shifts a 64×64 schedule this way with shift = d/64 for a d×d image.',
      details: [
        { label: 'Sigmoid (Chen 2023)', text: 'A sigmoid in t, normalised so ᾱ runs exactly from 1 to 0; start, end and τ set where and how steeply it drops. With −3, 3, 1 it crosses SNR = 1 at t = T/2, next to cosine.' },
        { label: 'EDM (ρ = 7)', text: 'Karras et al. space the noise levels σ rather than t; ρ = 7 shortens the steps near σ_min (low noise), and σ_max = 80 makes the last steps very noisy.' },
        { label: 'Shift = 2·ln(s)', text: 'A constant offset in log-SNR: SNR′ = SNR/shift². shift > 1 moves every curve down — more noise at every t, so SNR = 1 comes earlier; shift < 1 keeps more signal.' },
        { label: 'Why resolution matters', text: 'Neighbouring pixels are redundant, so at the same ᾱ a high-resolution image is easier to recognise than a low-resolution one — the noise averages out over nearby pixels. Adding noise (shift > 1) compensates, so the coarse structure is still undecided mid-chain rather than settled in the first few steps.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'METHODOLOGY', title: 'Why cosine beats linear', description: 'With the same T, the linear schedule drives ᾱₜ to near zero early: about a third of its steps have ᾱₜ < 0.01, against about 7% for cosine — those steps add noise to what is already noise.', recommendation: 'Prefer a cosine (or another schedule designed in log-SNR) so the model spends steps where there is still signal to remove; compare the ᾱ < 0.01 share of each schedule here.' },
    { category: 'VERIFICATION', title: 'Steps vs quality trade-off', description: 'Fewer sampling steps approximate the reverse SDE/ODE more coarsely; too few and samples blur or lose detail.', recommendation: 'Sweep the step count against a quality metric (FID) and pick the knee; pair an efficient schedule with a fast solver.' },
    { category: 'METHODOLOGY', title: 'Shift the schedule with resolution', description: 'A schedule that suits 64×64 is too gentle at 512×512: redundant neighbouring pixels keep the image recognisable at a given ᾱ, so the global layout is settled in only a few steps. Shifting ln SNR down by 2·ln(shift) adds noise at every t instead of redesigning the schedule.', recommendation: 'Raise the shift with resolution (simple diffusion uses shift = d/64) and check with the shift slider that SNR = 1 moves earlier — for cosine at T = 260, shift 2 moves it from t≈129 to t≈75.' },
  ],
};
