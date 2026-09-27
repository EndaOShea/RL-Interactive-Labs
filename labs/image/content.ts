import { LabContent } from '../../catalog/types';

// Co-located theory + lifecycle content for the Image Classification labs
// (rendered in each lab's Context tab via LabContext).

export const CONV_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Convolution as Feature Extraction',
      body: 'A convolution layer slides a small kernel (here 3×3) across the image and, at each position, computes a weighted sum of the underlying pixels. The kernel weights decide WHAT is detected: an edge-detect kernel fires on intensity changes, a blur kernel averages, an emboss kernel reacts to a directional gradient. One kernel = one learned (or hand-picked) feature detector reused everywhere on the image. Strictly, what CNN layers compute — and what this lab computes by default — is cross-correlation: the kernel is laid on the image as-is. True convolution rotates the kernel 180° first (the Operation toggle).',
      details: [
        { label: 'The operation', text: '(I⋆K)(i,j) = ΣΣ I(i+m, j+n)·K(m,n) — cross-correlation: multiply each kernel weight by the pixel under it, then sum. True convolution (I∗K)(i,j) = ΣΣ I(i−m, j−n)·K(m,n) is the same with K rotated 180°; the two differ only for asymmetric kernels (Sobel changes sign, emboss changes).' },
        { label: 'Edges', text: 'Edge kernels (edge-detect, Sobel, Laplacian) have weights summing to 0, so flat regions give exactly 0 and only intensity changes respond — with a sign, which is why the output is drawn on a zero-centred diverging scale.' },
        { label: 'Receptive field', text: 'Each output pixel sees only a small local patch — the kernel’s footprint. Stacking convolutions grows this field.' },
      ],
    },
    {
      heading: 'Padding, Stride & Output Size',
      body: 'Without padding, a kernel cannot be centred on the border, so the output shrinks. Padding by P = 1 lets the 3×3 kernel sit on border pixels and keeps a stride-1 output the same size; this lab offers three border modes for the padded ring. Stride is how far the window jumps each step; stride > 1 downsamples. The output side length is ⌊(W − F + 2P)/S⌋ + 1 for input W, kernel F, padding P, stride S.',
      details: [
        { label: 'Padding P', text: 'P = (F−1)/2 with stride 1 keeps the output the same size ("same" padding). This lab uses P = 1 with a choice of border mode.' },
        { label: 'Border mode', text: 'Zero treats off-image pixels as 0, so ink that touches the border looks like an edge: the default edge-detect gives 5 at (0,6) with zero padding but 3 — the interior bar value — with replicate. Replicate repeats the nearest edge pixel; reflect mirrors the image without repeating the edge pixel.' },
        { label: 'Stride S', text: 'S = 1 visits every position; S = 2 halves the output (14×14 → 7×7) and is a cheap downsampler that replaces a separate pooling layer in many modern nets.' },
        { label: 'Translation equivariance', text: 'The same kernel runs everywhere, so a shifted feature produces a shifted response — the network does not relearn it per location.' },
      ],
    },
    {
      heading: 'A Zoo of Kernels',
      body: 'The kernel weights alone decide the feature. First-derivative kernels (Sobel-X/Y) respond to directional edges; the second-derivative Laplacian responds to intensity curvature — negative on a bright ridge, positive just beside it, crossing zero at each edge. Smoothing kernels (box blur, Gaussian) average a neighbourhood — the Gaussian weights the centre more, giving a softer blur. Emboss combines a diagonal gradient with the pixel itself (its weights sum to 1) for a relief look; sharpen is the identity MINUS this −4-centre Laplacian, boosting local contrast.',
      details: [
        { label: 'Sobel vs Laplacian', text: 'Sobel = smoothed first derivative: a positive and a negative lobe, antisymmetric, so its sign gives the edge direction (and flips under true convolution). Laplacian = sum of second derivatives: isotropic and symmetric, its sign flips across an edge.' },
        { label: 'Box vs Gaussian blur', text: 'Box blur weights all 9 pixels equally; Gaussian uses 1-2-1 weights so the centre dominates — less ringing, the standard low-pass pre-filter.' },
        { label: 'Sharpen = I − Laplacian', text: 'With this Laplacian (centre −4), I − L = [0 −1 0; −1 5 −1; 0 −1 0] — exactly the sharpen kernel. Subtracting the Laplacian adds back the high-frequency detail it isolates.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'Weight sharing', description: 'A convolution reuses one tiny kernel across the whole image instead of a separate weight per pixel-pair, drastically cutting parameters versus a dense layer.', recommendation: 'Think of each kernel as a single reusable pattern matcher; depth (many kernels) gives breadth of features.' },
    { category: 'METHODOLOGY', title: 'Border handling matters', description: 'Different padding choices (zero, reflect, replicate) create different artefacts at the image edge.', recommendation: 'Pick padding deliberately; zero-pad is simplest but can darken borders for blur-like kernels — reflect/replicate avoid the dark halo.' },
    { category: 'METHODOLOGY', title: 'Strided convolution', description: 'A stride-2 convolution both filters and downsamples in one op, shrinking the feature map and growing the effective receptive field of later layers.', recommendation: 'Use strided convs (or pooling) to build a coarse-to-fine pyramid; check the output-size formula so dimensions stay integral.' },
  ],
};

export const FEATUREMAPS_CONTENT: LabContent = {
  sections: [
    {
      heading: 'A Tiny CNN Pipeline',
      body: 'This lab runs the classic convolutional stack on a small, optionally perturbed glyph: input → 3 fixed filters (cross-correlation, as in every CNN layer) → ReLU → 2×2 pooling (max or average) → flatten → classify. Each filter produces a signed feature map; ReLU keeps only its positive part; pooling halves each map. All three maps are drawn on one shared colour scale, so their strengths compare honestly.',
      details: [
        { label: 'Fixed filters', text: 'Vertical edge [−1 0 1], horizontal edge (its transpose) and a 3×3 blur whose weights sum to 1 — hand-picked, NOT trained. On every clean glyph both edge filters peak at 3 and the blur at 0.9. A real CNN learns its filters by backprop.' },
        { label: 'ReLU', text: 'max(0, x). On a signed edge filter it keeps ONE polarity: the vertical-edge map keeps the left side of each stroke (dark→bright) and zeroes the right side; the horizontal-edge map keeps only top edges.' },
        { label: 'Pooling 2×2', text: 'Downsamples each map by halving both dimensions. Max-pool keeps the strongest activation per block (the classic choice); average-pool blends all four (smoother, used in modern nets and as global-average-pool before the classifier).' },
        { label: 'Max vs average', text: 'A 2×2 pool only absorbs a move that keeps a feature inside its block, so neither choice makes this network shift-invariant: a 1-px shift lowers the own-class cosine from 1 to 0.76–0.84 with max pooling and 0.77–0.81 with average pooling. The choice can flip a decision — T shifted 2 px right is still T with max pooling (p = 0.54) but O with average pooling (p = 0.62).' },
      ],
    },
    {
      heading: 'Classification & Honesty',
      body: 'After pooling, the maps are flattened into one 108-number vector and compared by cosine similarity with the vector of each CLEAN glyph (the templates); softmax(8·cos) turns the similarities into a class distribution. A clean input is its own template (cos = 1), so the perturbations — shifts, thicker strokes, noise — are what make the match a test. This is template matching on top of fixed features — it illustrates the data flow of a CNN without any learning. A trained CNN would replace both the filters AND the final matcher with learned weights.',
      details: [
        { label: 'Cosine similarity', text: 'cos(a,b) = (a·b)/(‖a‖‖b‖) — measures pattern overlap regardless of overall brightness.' },
        { label: 'Softmax', text: 'p_i = e^{8·cos_i}/Σ_j e^{8·cos_j} — converts the cosines into a probability distribution; the factor 8 sets how peaked it is.' },
        { label: 'Why CNNs beat dense nets on images', text: 'Local receptive fields + weight sharing + pooling encode translation tolerance and slash parameters versus a fully-connected net.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'Hierarchy of features', description: 'Early layers detect edges/blobs; deeper layers compose them into parts and objects. This lab shows only the first layer.', recommendation: 'Read the feature maps as "where does this simple pattern occur" — depth is what builds semantics.' },
    { category: 'VERIFICATION', title: 'Fixed vs learned', description: 'With hand-picked filters and template matching, accuracy is limited and brittle: a 2-px shift cuts the margin over the next class below 0.08 for every glyph and flips T to O under average pooling.', recommendation: 'Treat the result as a wiring diagram, not a performant classifier; training (and data augmentation with shifted inputs) is what makes CNNs work.' },
  ],
};
