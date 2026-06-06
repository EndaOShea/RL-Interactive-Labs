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
      body: 'First- and second-order Taylor expansions justify the core of optimisation: gradient descent is the first-order model J(θ+Δ) ≈ J(θ) + ∇J·Δ, and Newton/quasi-Newton methods use the second-order (Hessian) term. Smooth activations like GELU and softplus are effectively polynomial/series approximations of gates. But series only converge within a radius: 1/(1−x) about a=0 converges only for |x|<1 and blows up at the pole x=1; ln(1+x) converges on (−1, 1].',
      details: [
        { label: 'Optimisation link', text: 'GD = linear Taylor model of the loss; Newton = quadratic model — faster but needs the Hessian.' },
        { label: 'Radius of convergence', text: 'Distance from a to the nearest singularity. Outside it, adding terms makes the error worse.' },
        { label: 'eˣ, sin, cos', text: 'Entire functions — their series converge on the whole real line.' },
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
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'Singular & near-singular maps', description: 'When det ≈ 0 the matrix is (nearly) non-invertible; solving systems with it amplifies numerical error.', recommendation: 'Check the condition number, not just the determinant; regularise or use a pseudo-inverse (SVD) for ill-conditioned problems.' },
    { category: 'DEPLOYMENT', title: 'Eigenvalues & stability', description: 'In linear dynamics, RNNs and iterative solvers, eigenvalues with magnitude >1 cause exploding behaviour and <1 cause vanishing signals.', recommendation: 'Constrain the spectral radius (e.g. spectral normalisation, orthogonal init) to keep training and inference stable.' },
  ],
};
