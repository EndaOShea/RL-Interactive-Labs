import { LabContent } from '../../catalog/types';

// Co-located theory + lifecycle content for the Maths labs (the foundations
// behind ML), rendered in each lab's Context tab via LabContext.

export const GD_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Derivatives, gradients & optimisation',
      body: 'Almost all ML training is optimisation: minimise a loss J(θ) over parameters θ. The derivative f′(x) (in many dimensions, the gradient ∇J) is the local slope — it points in the direction of steepest increase. Gradient descent steps the opposite way, θ ← θ − α·∇J, so each step lowers the loss while the slope is non-zero.',
      details: [
        { label: "f'(x)", text: 'The slope of the tangent line. Where it is zero we are at a stationary point (min, max, or saddle).' },
        { label: 'Negative gradient', text: 'The downhill direction — GD follows it; the size of each move is α·|∇f|.' },
        { label: 'Convergence', text: 'Updates stop when the gradient ≈ 0; for a convex bowl that point is the global minimum.' },
      ],
    },
    {
      heading: 'Learning rate, momentum & local minima',
      body: 'The learning rate α sets the step size: too small and convergence crawls, too large and the iterate overshoots and diverges. Momentum keeps a running velocity (v ← βv − α∇f), smoothing the trajectory and carrying the point across shallow dips and flat plateaus. Convex functions have a single minimum; non-convex ones (double-well, wavy) have many, and where you land depends on the start point and momentum.',
      details: [
        { label: 'α too large', text: 'Steps overshoot the minimum and the value climbs — divergence.' },
        { label: 'Momentum β', text: 'Accumulated velocity helps escape shallow local minima and dampens zig-zagging in narrow valleys.' },
        { label: 'Convex vs non-convex', text: 'Convex → any descent finds the global min. Non-convex → GD only guarantees a local min.' },
      ],
    },
    {
      heading: 'Adaptive & second-order optimisers',
      body: 'Plain GD uses one global step size; modern optimisers adapt it. RMSProp divides the step by √(running mean of g²), so steep directions are damped and flat ones amplified. Adam combines this second moment with a momentum-like first moment (and bias-corrects both), giving the de-facto default for deep learning. Newton’s method goes further and uses curvature: x ← x − f′(x)/f″(x). On an exact quadratic that lands on the minimum in a single step, but it needs the Hessian f″ to be positive-definite — near a maximum (f″<0) Newton steps the wrong way.',
      details: [
        { label: 'RMSProp', text: 'Per-coordinate step α·g/√(s+ε) with s an EMA of g² — scale-free, good for ill-conditioned losses.' },
        { label: 'Adam', text: 'First moment m (momentum) + second moment v (RMSProp) with bias correction; robust default α≈1e-3.' },
        { label: 'Newton', text: 'Second-order: divides by curvature f″. One-step exact on quadratics; expensive and unsafe where f″≤0.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'METHODOLOGY', title: 'Tuning the learning rate', description: 'A single fixed α rarely suits the whole loss surface; the same step that is stable in one region diverges in another.', recommendation: 'Use learning-rate schedules or adaptive optimisers (Adam, RMSProp) and watch the loss curve for smooth decay.' },
    { category: 'CONCEPT', title: 'Local minima & initialisation', description: 'On non-convex losses the solution depends on where you start, so a single run can land in a poor basin.', recommendation: 'Use sensible initialisation, momentum, and multiple restarts; in deep nets, over-parameterisation makes most minima comparably good.' },
  ],
};

export const TAYLOR_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Taylor approximation',
      body: 'A Taylor series rewrites a smooth function as an infinite polynomial about a centre a: f(x) = Σ fⁿ(a)/n!·(x−a)ⁿ. Truncating at degree n gives a polynomial that matches f and its first n derivatives exactly at a, and hugs the curve more widely as n grows. Polynomials are cheap to evaluate, which is why approximations underpin so much numerical computing.',
      details: [
        { label: 'Coefficients', text: 'The n-th coefficient is fⁿ(a)/n! — the n-th derivative at the centre, scaled by 1/n!.' },
        { label: 'Degree n', text: 'Higher degree → tighter fit over a wider interval (inside the radius of convergence).' },
        { label: 'Centre a', text: 'The expansion is most accurate near a and degrades as you move away.' },
      ],
    },
    {
      heading: 'Why it underlies ML & radius of convergence',
      body: 'First- and second-order Taylor expansions justify the core of optimisation: gradient descent is the first-order model J(θ+Δ) ≈ J(θ) + ∇J·Δ, and Newton/quasi-Newton methods use the second-order (Hessian) term. Smooth activations like GELU and softplus are effectively polynomial/series approximations of gates. But series only converge within a radius: 1/(1−x) about a=0 converges only for |x|<1 and blows up at the pole x=1; ln(1+x) converges on (−1, 1]. The radius is the distance to the nearest singularity in the COMPLEX plane — so tanh (poles at ±iπ/2) and Runge’s 1/(1+25x²) (poles at ±i/5) have finite radii even though they are perfectly smooth and bounded on the real line.',
      details: [
        { label: 'Optimisation link', text: 'GD = linear Taylor model of the loss; Newton = quadratic model — faster but needs the Hessian.' },
        { label: 'Radius of convergence', text: 'Distance from a to the nearest (complex) singularity. Outside it, adding terms makes the error worse.' },
        { label: 'eˣ, sin, cos', text: 'Entire functions — their series converge on the whole real line.' },
      ],
    },
    {
      heading: 'Padé: rational approximation past the pole',
      body: 'A Padé approximant replaces the truncated polynomial with a ratio of polynomials R(x)=P_m(x)/Q_m(x) whose Taylor expansion matches f to order 2m. Because the denominator Q can vanish, Padé can model the function’s poles — so it often converges far past the radius where the raw Taylor series diverges (e.g. usefully approximating 1/(1−x) near x=1, or tanh out on its saturated plateau). For the same total work, Padé[m/m] is frequently orders of magnitude more accurate than the degree-2m polynomial, which is why it shows up in special-function libraries, control theory and model reduction.',
      details: [
        { label: 'Rational P/Q', text: 'Numerator and denominator polynomials fit to the Taylor coefficients; the denominator captures poles.' },
        { label: 'Beats the polynomial', text: 'Same coefficient data, but the rational form extends the accurate range past the Taylor radius.' },
        { label: 'Used in', text: 'Special-function evaluation, matrix-exponential scaling-and-squaring, and reduced-order modelling.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'VERIFICATION', title: 'Truncation error', description: 'A finite Taylor polynomial carries a remainder term; using it outside its accurate region gives silently wrong numbers.', recommendation: 'Bound the remainder (Lagrange form) or evaluate near the centre, and check error at the working point.' },
    { category: 'CONCEPT', title: 'Convergence is local', description: 'More terms only help inside the radius of convergence; beyond it (e.g. past a pole) the approximation diverges.', recommendation: 'Know where the singularities are; re-centre the expansion or switch to a different approximation (Padé, splines) when needed.' },
  ],
};

export const LINTRANSFORM_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Linear maps & the determinant',
      body: 'A 2×2 matrix M is a linear map of the plane: v ↦ Mv. Its columns are exactly where the basis vectors î and ĵ land, which fully determines the transform of every other vector. The determinant is the signed factor by which areas scale: |det| is the area of the transformed unit square, det<0 means orientation is flipped, and det=0 means the plane is squashed onto a line (the map is singular and non-invertible).',
      details: [
        { label: 'Columns = basis images', text: 'Column 1 is M·î, column 2 is M·ĵ. Linearity does the rest.' },
        { label: 'Determinant', text: 'Signed area scale. det=0 ⇒ rank-deficient, no inverse exists.' },
        { label: 'Composition', text: 'Applying two maps multiplies their matrices; dets multiply too.' },
      ],
    },
    {
      heading: 'Eigenvectors, eigenvalues & their role',
      body: 'An eigenvector v satisfies Mv = λv — the map only stretches it by λ without changing its direction. Real eigenvectors are the special axes of the transform; rotations have complex eigenvalues (no fixed real direction). These are the same axes PCA finds in a covariance matrix (directions of maximum variance), and the eigenvalues of a system matrix govern stability: a linear recurrence xₜ₊₁ = M xₜ grows if any |λ|>1 and decays if all |λ|<1.',
      details: [
        { label: 'Mv = λv', text: 'Eigen-directions are stretched, not rotated; λ is the stretch factor.' },
        { label: 'PCA', text: 'Principal components are the eigenvectors of the covariance matrix; explained variance ∝ eigenvalue.' },
        { label: 'Stability', text: '|λ|>1 ⇒ a direction blows up; spectral radius decides growth/decay in dynamics & RNNs.' },
      ],
    },
    {
      heading: 'SVD, singular values & conditioning',
      body: 'Every matrix factors as M = U Σ Vᵀ — a rotation (Vᵀ), an axis-aligned stretch by the singular values σ₁≥σ₂≥0 (Σ), then another rotation (U). Geometrically the unit circle maps to an ellipse whose semi-axis lengths are exactly the singular values, oriented along the columns of U. Unlike eigenvalues, singular values are always real and ≥0 and exist for any matrix — even rectangular ones, and even when the eigenvalues are complex (a pure rotation has complex λ but singular values all 1). The ratio κ = σ₁/σ₂ is the condition number: large κ means the map is nearly singular and solving with it amplifies noise. SVD is the engine behind PCA, low-rank/least-squares approximation (keep the top-k σ) and the Moore–Penrose pseudo-inverse.',
      details: [
        { label: 'M = U Σ Vᵀ', text: 'Rotate → stretch by σ → rotate. The σ are the ellipse semi-axes the unit circle maps to.' },
        { label: 'Condition number κ', text: 'κ = σ₁/σ₂. κ≈1 is well-conditioned (rotation/uniform scale); κ≫1 is near-singular and unstable.' },
        { label: 'Low-rank & pseudo-inverse', text: 'Truncating small σ gives the best low-rank fit (PCA); inverting the non-zero σ gives the pseudo-inverse.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'Singular & near-singular maps', description: 'When det ≈ 0 the matrix is (nearly) non-invertible; solving systems with it amplifies numerical error.', recommendation: 'Check the condition number, not just the determinant; regularise or use a pseudo-inverse (SVD) for ill-conditioned problems.' },
    { category: 'DEPLOYMENT', title: 'Eigenvalues & stability', description: 'In linear dynamics, RNNs and iterative solvers, eigenvalues with magnitude >1 cause exploding behaviour and <1 cause vanishing signals.', recommendation: 'Constrain the spectral radius (e.g. spectral normalisation, orthogonal init) to keep training and inference stable.' },
  ],
};


export const DERIVATIVES_CONTENT: LabContent = {
  sections: [
    {
      heading: 'The derivative as a tangent slope',
      body: "The derivative f′(x) measures how fast f changes at x — geometrically, the slope of the line that just touches the curve there (the tangent). To get it, take a nearby point x+dx, draw the SECANT through (x, f(x)) and (x+dx, f(x+dx)), and read off its slope [f(x+dx) − f(x)]/dx, the average rate of change over that interval. As dx shrinks the secant pivots about x and lines up with the tangent; its slope converges to f′(x). That limit is the definition of the derivative.",
      details: [
        { label: 'Secant slope', text: '[f(x+dx) − f(x)]/dx — the average rate of change over [x, x+dx]. An approximation to f′(x) for any finite dx.' },
        { label: 'Tangent slope', text: 'The limit of the secant slope as dx→0; the instantaneous rate of change f′(x) at the point.' },
        { label: 'Limit definition', text: "f′(x) = lim₍dx→0₎ [f(x+dx) − f(x)]/dx. The whole of differential calculus follows from this one shrinking-secant idea." },
      ],
    },
    {
      heading: 'Known derivatives & convergence',
      body: "Each function here has a closed-form derivative, so the exact tangent slope is known and we can watch the numeric secant chase it: x² → 2x, x³−x → 3x²−1, sin x → cos x, eˣ → eˣ. The forward-difference error is governed by a Taylor expansion: f(x+dx) = f(x) + f′(x)·dx + ½f″(x)·dx² + …, so the secant slope overshoots the true derivative by about ½·f″(x)·dx. That means the error is roughly LINEAR in dx — halve dx and the error roughly halves — which is exactly what the |error| chip shows as you drive dx toward 0.",
      details: [
        { label: 'Closed forms', text: 'Power rule (xⁿ → n·xⁿ⁻¹), the trig pair (sin↔cos), and eˣ — the self-derivative — cover the four pickers.' },
        { label: 'Truncation error', text: 'Forward difference error ≈ ½·f″(x)·dx (first-order). A centred difference [f(x+dx)−f(x−dx)]/2dx is second-order (∝ dx²) and far more accurate.' },
        { label: 'Why it matters', text: 'Gradient descent, backprop and autodiff are all built on derivatives; finite differences are the limit definition turned into a numerical recipe.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'Differentiability is local & not guaranteed', description: 'The tangent slope only exists where the curve is smooth; at a kink or jump the left and right secants disagree and no single derivative exists (e.g. |x| at 0, or a ReLU at 0).', recommendation: 'Check smoothness before differentiating; for non-smooth functions use subgradients or smooth surrogates (softplus for ReLU) where a true derivative is required.' },
    { category: 'METHODOLOGY', title: 'Finite-difference step size', description: 'A finite-difference derivative trades truncation error (∝ dx, too large a step) against floating-point round-off (∝ 1/dx, too small a step), so error bottoms out at an intermediate dx rather than at dx→0.', recommendation: 'Use a centred difference and a step near √ε·|x| (≈1e-6 in double precision), or prefer exact analytic / automatic differentiation when available.' },
  ],
};

export const CHAINRULE_CONTENT: LabContent = {
  sections: [
    {
      heading: 'The chain rule: derivative as a product',
      body: 'A composite function feeds its output through a chain of simpler maps: x → u → … → y. The chain rule says the derivative of the whole thing is the PRODUCT of the local derivatives along that path. For y(u(x)) it is dy/dx = (dy/du)·(du/dx); for a three-link chain y(v(u(x))) it is dy/dx = (dy/dv)·(dv/du)·(du/dx). Each link is differentiated on its own — treating its input as the variable — and then everything multiplies.',
      details: [
        { label: 'Local derivative', text: 'Differentiate one link in isolation, e.g. d(sin u)/du = cos u, evaluated at that link’s input value.' },
        { label: 'Multiply the path', text: 'Walk x → y and multiply every local derivative you cross; order does not matter for the product, only that you include each link once.' },
        { label: 'Evaluate at a point', text: 'Each factor is a number once you fix x₀: the forward pass fills in u, v, …, then the local derivatives are read off and multiplied.' },
      ],
    },
    {
      heading: 'Forward pass then backward product',
      body: 'Computing dy/dx is a two-sweep process. The forward sweep evaluates each node’s value (u = x², v = −u, …). The backward sweep evaluates each link’s local derivative at the value that flowed into it, then multiplies them. This is exactly what reverse-mode automatic differentiation (backpropagation) does: forward to get activations, backward to multiply local Jacobians. A finite-difference check, [f(x+h) − f(x−h)]/2h on the whole composite, must agree with the product — and in this lab it does, to within tiny discretisation error.',
      details: [
        { label: 'Forward = values', text: 'Each node stores its numeric output; later links need these inputs to evaluate their local slopes.' },
        { label: 'Backward = derivatives', text: 'Each edge carries d(out)/d(in) at its input value; the running product is dy/dx.' },
        { label: 'Numeric cross-check', text: 'A central finite difference of the full function confirms the analytic product — realism, not a mocked number.' },
      ],
    },
    {
      heading: 'Why it powers backprop',
      body: 'A neural network is one giant composite: loss(softmax(W₂·act(W₁·x))). Training needs ∂loss/∂each-weight, and the chain rule supplies every one of them as a product of local derivatives. Reverse-mode autodiff multiplies those local Jacobians from the loss back to the parameters in a single sweep, reusing shared sub-paths. The same product structure explains vanishing and exploding gradients: multiply many factors below 1 and the gradient decays to nothing; multiply many above 1 and it blows up.',
      details: [
        { label: 'Backpropagation', text: 'The chain rule applied to a computation graph; each layer contributes one local Jacobian factor.' },
        { label: 'Vanishing / exploding', text: 'dy/dx is a product over depth — repeated small factors vanish, repeated large factors explode.' },
        { label: 'Reuse', text: 'Reverse mode computes all input derivatives in one pass by sharing the multiplied prefixes along the path.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'Evaluate each factor at the right point', description: 'A link’s local derivative must be evaluated at the value that actually flows into it, not at x. Plugging x into an inner-layer derivative is a classic chain-rule mistake.', recommendation: 'Do the forward pass first to record every node’s value, then evaluate each local derivative at its own input before multiplying.' },
    { category: 'METHODOLOGY', title: 'Gradients are products over depth', description: 'Because dy/dx multiplies one factor per link, deep compositions are prone to vanishing or exploding gradients when factors are consistently below or above 1.', recommendation: 'Use variance-preserving initialisation, normalisation, and skip connections so the per-link factors stay near 1 and the product stays well-scaled.' },
  ],
};

export const MATMUL_CONTENT: LabContent = {
  sections: [
    {
      heading: 'The dot product',
      body: 'The dot product collapses two vectors into one number: a·b = a₁b₁ + a₂b₂ (sum the element-wise products). It has a twin geometric form, a·b = |a||b|cos θ, so the same number measures both algebraic agreement and the angle between the vectors. This is the single most important operation in ML — the pre-activation of every neuron is the dot product of its weight vector with its input.',
      details: [
        { label: 'a·b = Σ aᵢbᵢ', text: 'Multiply matching entries and add. Order does not matter: a·b = b·a.' },
        { label: '|a||b|cos θ', text: 'cos θ > 0 ⇒ aligned (positive), = 0 ⇒ orthogonal (a·b = 0), < 0 ⇒ opposed (negative).' },
        { label: 'Cosine similarity', text: 'Normalising gives cos θ = a·b / (|a||b|) — the similarity score used in retrieval and embeddings.' },
      ],
    },
    {
      heading: 'Projection: the shadow of one vector on another',
      body: 'The dot product also gives projection. The signed length of the shadow a casts on b is comp_b(a) = a·b / |b|, and the projection vector itself is proj_b(a) = (a·b / |b|²) b. Projection is the geometric heart of least squares: the best linear fit is the projection of the targets onto the span of the features, and Gram–Schmidt / QR build orthogonal bases by repeatedly subtracting projections.',
      details: [
        { label: 'Scalar projection', text: 'comp_b(a) = a·b / |b| — how far a reaches along b. Zero when a ⟂ b.' },
        { label: 'Vector projection', text: 'proj_b(a) = (a·b / |b|²) b — the component of a lying along b; a − proj_b(a) is the orthogonal residual.' },
        { label: 'Least squares', text: 'Fitting = projecting the target onto the column space of the design matrix; the residual is orthogonal to it.' },
      ],
    },
    {
      heading: 'Matrix · vector = a dense layer',
      body: 'A matrix·vector product stacks dot products: y = A x, where each output entry yᵢ is row i of A dotted with x. That is exactly what a fully-connected (dense) layer computes — A is the weight matrix, x the input, y the pre-activation. Read the columns instead and you get the transform view: the columns of A are where the basis vectors ê₁, ê₂ land, and y is just x₁·col₁ + x₂·col₂ — a weighted sum of column vectors.',
      details: [
        { label: 'Row view', text: 'yᵢ = Σⱼ Aᵢⱼ xⱼ — every output is a row·input dot product. m×n matrix needs n-length x and gives m-length y.' },
        { label: 'Column view', text: 'Columns of A are the images of the basis; y = Σⱼ xⱼ·(column j). Same answer, geometric reading.' },
        { label: 'det(A)', text: 'For a square map, |det| is the area/volume scale; det<0 flips orientation; det=0 collapses dimensions (singular).' },
      ],
    },
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'Shape compatibility', description: 'A (m×n)·(n) works but (m×n)·(k) with k≠n is undefined; mismatched inner dimensions are the most common matrix bug.', recommendation: 'Track shapes explicitly: (m×n)·(n×p) → (m×p). Print or assert tensor shapes before every matmul in a model.' },
    { category: 'DEPLOYMENT', title: 'Matmul is where the FLOPs go', description: 'Dense and attention layers are dominated by matrix multiplies; their cost and numerical conditioning drive both speed and stability.', recommendation: 'Batch products into single GEMM calls, use the right precision (bf16/fp16 with care), and watch for ill-conditioned weights that amplify error.' },
  ],
};

export const CONVEX_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Convex vs non-convex loss surfaces',
      body: 'A function is convex if a straight line between any two points on its graph never dips below the curve — equivalently, its curvature f″ is everywhere ≥ 0. A convex loss has exactly one basin and therefore a single global minimum: gradient descent, x ← x − α·f′(x), slides to that minimum from any starting point. Non-convex losses (here f(x)=0.15x²+2·sin 3x) ripple into several local minima separated by humps and near-flat saddle regions, so the slope can vanish at many different points.',
      details: [
        { label: 'Convex f', text: 'One global minimum; any descent path reaches it. f″ ≥ 0 everywhere (e.g. f(x)=x², f″=2).' },
        { label: 'Non-convex f', text: 'Multiple local minima and maxima; f″ changes sign. Descent only guarantees a local minimum.' },
        { label: 'Basin of attraction', text: 'The set of starts that flow to a given minimum. Convex = one basin; non-convex = many.' },
      ],
    },
    {
      heading: 'Why initialisation decides the answer',
      body: 'Gradient descent is a purely local method: it can only follow the slope downhill from where it stands, so it settles at the bottom of whatever basin it starts in. On the convex bowl every runner converges to the same point — the start is irrelevant. On the rippled surface, runners launched from different x split apart and settle in different minima, and only some of those basins contain the global minimum (x≈−0.515, f≈−1.96 here). This is the core reason training deep networks is sensitive to weight initialisation, random seeds and restarts.',
      details: [
        { label: 'Local, not global', text: 'GD stops wherever f′≈0; on a non-convex surface that is usually a local, not the global, minimum.' },
        { label: 'Multi-start', text: 'Running many starts and keeping the best is a standard way to hedge against a bad basin.' },
        { label: 'Learning rate α', text: 'A larger α can hop between ripples (sometimes escaping a poor basin) but too large overshoots and diverges.' },
      ],
    },
    {
      heading: 'Coping with non-convexity in practice',
      body: 'Real ML losses are overwhelmingly non-convex, yet deep networks train well. Several factors help: careful initialisation (Xavier/He) places parameters in good basins; stochastic gradients add noise that can jolt the iterate out of sharp local traps; momentum and learning-rate schedules carry the point across shallow dips; and crucially, in very high dimensions most critical points are saddles, not bad minima, and the many minima that exist tend to be of similar quality. So while convexity guarantees a unique optimum, modern practice manages non-convexity rather than eliminating it.',
      details: [
        { label: 'Smart init', text: 'He/Xavier scaling keeps activations and gradients well-conditioned, biasing toward good basins.' },
        { label: 'SGD noise', text: 'Mini-batch gradient noise acts like annealing, helping escape narrow/sharp local minima.' },
        { label: 'High-dimensional geometry', text: 'In large nets, saddles dominate over poor local minima, and most minima generalise comparably.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'Initialisation & seeds matter', description: 'On non-convex losses the solution found depends on where optimisation starts, so a single run can land in a poor basin and a different seed can change results.', recommendation: 'Use principled initialisation (He/Xavier), fix and report seeds, and run multiple restarts when the surface is rugged.' },
    { category: 'METHODOLOGY', title: 'Don’t assume the global optimum', description: 'Gradient descent only certifies a local minimum (f′≈0, f″>0); claiming a global optimum without convexity is unjustified.', recommendation: 'Compare several starts, use momentum/SGD noise and schedules to escape shallow traps, and judge by held-out performance rather than training loss alone.' },
  ],
};

export const EIGENSVD_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Eigenvalues & the characteristic equation',
      body: 'An eigenvector of a square matrix A is a direction the map only stretches: A·v = λ·v, with the scalar λ its eigenvalue. For a 2×2 matrix the eigenvalues are the roots of the characteristic equation λ² − t·λ + det = 0, where t = trace = a+d and det = ad−bc. The discriminant disc = t² − 4·det decides everything: disc ≥ 0 gives two real eigenvalues λ = (t ± √disc)/2, each with a real eigenvector found by solving (A − λI)v = 0; disc < 0 gives a complex conjugate pair, meaning A rotates the plane and has no real invariant axis.',
      details: [
        { label: 'A v = λ v', text: 'Eigen-directions are scaled, not rotated; λ is the stretch (negative λ flips, |λ|>1 grows).' },
        { label: 'trace & det', text: 'Eigenvalues sum to the trace (a+d) and multiply to the determinant (ad−bc).' },
        { label: 'disc < 0 ⇒ rotation', text: 'A complex pair means no real fixed direction — every vector turns. A pure rotation is the cleanest example.' },
      ],
    },
    {
      heading: 'The SVD: rotate → scale → rotate',
      body: 'Every matrix — even one with complex eigenvalues — factors as A = U Σ Vᵀ. Read right-to-left this is a rotation Vᵀ, then an axis-aligned stretch by the singular values σ₁ ≥ σ₂ ≥ 0 in Σ, then another rotation U. Concretely the unit circle maps to an ellipse whose semi-axis lengths are exactly σ₁ and σ₂, oriented along the columns of U. The σ are the square roots of the eigenvalues of the symmetric matrix AᵀA, whose eigenvectors are the columns of V; each left vector is uᵢ = A·vᵢ / σᵢ. Singular values are always real and non-negative and exist for any matrix, which is why the SVD is more universal than the eigen-decomposition.',
      details: [
        { label: 'A = U Σ Vᵀ', text: 'Vᵀ rotates the orthonormal pre-image axes onto the standard axes, Σ scales by σ, U rotates onto the ellipse.' },
        { label: 'σᵢ = √eig(AᵀA)', text: 'Singular values come from the symmetric, positive-semidefinite Gram matrix AᵀA — so they are always real and ≥ 0.' },
        { label: 'Circle → ellipse', text: 'The image of the unit circle is an ellipse; its semi-axes are σ₁ (long) and σ₂ (short).' },
      ],
    },
    {
      heading: 'Conditioning, PCA & low-rank',
      body: 'The ratio κ = σ₁/σ₂ is the condition number. κ ≈ 1 is a near-rotation or uniform scale (well-conditioned); κ ≫ 1 means A is nearly singular — the ellipse is a thin sliver, det → 0, and solving systems with A blows up tiny errors. The SVD is the engine under much of ML: PCA diagonalises a covariance matrix, so its principal components are eigenvectors and the explained variance is proportional to the eigenvalue; keeping only the top-k singular values gives the best low-rank approximation (image/embedding compression); and inverting only the non-zero σ gives the Moore–Penrose pseudo-inverse used in least squares.',
      details: [
        { label: 'κ = σ₁/σ₂', text: 'Condition number. Large κ (or σ₂ → 0) ⇒ near-singular, numerically unstable to invert.' },
        { label: 'PCA', text: 'Principal axes are the eigenvectors of the covariance matrix; variance along each ∝ its eigenvalue.' },
        { label: 'Low-rank & pseudo-inverse', text: 'Truncate small σ for the best rank-k fit; invert the kept σ for the least-squares pseudo-inverse.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'Singular & near-singular maps', description: 'When det ≈ 0 (σ₂ → 0) the matrix is nearly non-invertible; the image ellipse collapses toward a line and solving with it amplifies noise.', recommendation: 'Watch the condition number κ = σ₁/σ₂, not just the determinant; regularise or use an SVD pseudo-inverse for ill-conditioned problems.' },
    { category: 'METHODOLOGY', title: 'Eigen vs singular values', description: 'Eigenvalues can be complex and only exist for square matrices, while singular values are always real, non-negative and defined for any matrix — a pure rotation has complex λ but all σ = 1.', recommendation: 'Use the eigen-decomposition for invariant directions and stability (spectral radius); reach for the SVD for stretch, rank, conditioning and least-squares.' },
  ],
};