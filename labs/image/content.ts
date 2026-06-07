import { LabContent } from '../../catalog/types';

// Co-located theory + lifecycle content for the Image Classification labs
// (rendered in each lab's Context tab via LabContext).

export const CONV_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Convolution as Feature Extraction',
      body: 'A convolution slides a small kernel (here 3×3) across the image and, at each position, computes a weighted sum of the underlying pixels. The kernel weights decide WHAT is detected: an edge-detect kernel fires on intensity changes, a blur kernel averages, an emboss kernel reacts to directional gradients. One kernel = one learned (or hand-picked) feature detector reused everywhere on the image.',
      details: [
        { label: 'The operation', text: '(I∗K)(i,j) = ΣΣ I(i+m, j+n)·K(m,n) — multiply each kernel weight by the pixel under it, then sum.' },
        { label: 'Edges', text: 'Edge kernels have a positive and a negative lobe, so flat regions sum to ~0 and only intensity changes survive.' },
        { label: 'Receptive field', text: 'Each output pixel sees only a small local patch — the kernel’s footprint. Stacking convolutions grows this field.' },
      ],
    },
    {
      heading: 'Padding, Stride & Output Size',
      body: 'Without padding, a kernel cannot be centred on the border, so the output shrinks. Zero-padding (used here) keeps the output the same size by surrounding the image with zeros. Stride is how far the window jumps each step; stride > 1 downsamples. The output side length is ⌊(W − F + 2P)/S⌋ + 1 for input W, kernel F, padding P, stride S.',
      details: [
        { label: 'Padding P', text: 'P = (F−1)/2 with stride 1 keeps the output the same size ("same" padding). This lab uses P = 1 with a choice of border mode.' },
        { label: 'Border mode', text: 'Zero treats off-image pixels as 0 (can darken/halo borders); replicate repeats the nearest edge pixel; reflect mirrors the image across its edge for the smoothest border.' },
        { label: 'Stride S', text: 'S = 1 visits every position; S = 2 halves the output (14×14 → 7×7) and is a cheap downsampler that replaces a separate pooling layer in many modern nets.' },
        { label: 'Translation equivariance', text: 'The same kernel runs everywhere, so a shifted feature produces a shifted response — the network does not relearn it per location.' },
      ],
    },
    {
      heading: 'A Zoo of Kernels',
      body: 'The kernel weights alone decide the feature. First-derivative kernels (Sobel-X/Y) respond to directional edges; the second-derivative Laplacian responds to curvature and fires on both sides of an edge. Smoothing kernels (box blur, Gaussian) average a neighbourhood — the Gaussian weights the centre more, giving a softer, artefact-free blur. Emboss combines a gradient with a bias for a 3-D relief look; sharpen is the identity plus a Laplacian to boost local contrast.',
      details: [
        { label: 'Sobel vs Laplacian', text: 'Sobel = first derivative (one signed lobe → edge direction); Laplacian = sum of second derivatives (isotropic, sign flips across an edge).' },
        { label: 'Box vs Gaussian blur', text: 'Box blur weights all 9 pixels equally; Gaussian uses 1-2-1 weights so the centre dominates — less ringing, the standard low-pass pre-filter.' },
        { label: 'Sharpen = I + Laplacian', text: 'Adding a scaled Laplacian back to the original amplifies exactly the high-frequency detail the Laplacian isolates.' },
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
      body: 'This lab runs the classic convolutional stack on a small glyph: input → convolution with several filters → ReLU non-linearity → max-pooling → flatten → classify. Each filter produces a feature map highlighting where its pattern occurs; ReLU keeps only positive (present) responses; pooling shrinks the maps while keeping the strongest activations.',
      details: [
        { label: 'Fixed filters', text: 'The conv filters here are hand-picked (vertical edge, horizontal edge, blob) — NOT trained. A real CNN learns these by backprop.' },
        { label: 'ReLU', text: 'max(0, x) — discards negative responses, so each map shows only where its feature is positively present.' },
        { label: 'Pooling 2×2', text: 'Downsamples each map by halving both dimensions. Max-pool keeps the strongest activation per block (peak-preserving, the classic choice); average-pool blends all four (smoother, used in modern nets and as global-average-pool before the classifier).' },
        { label: 'Max vs average', text: 'Max is invariant to where in the block the peak sits, so it is robust to small shifts; average retains more of the overall energy and is gentler on noise. Swapping pooling can change which template the feature vector lands closest to.' },
      ],
    },
    {
      heading: 'Classification & Honesty',
      body: 'After pooling, the maps are flattened into one vector and compared to a stored template vector per class by cosine similarity; softmax turns the similarities into a class distribution. This is template matching on top of fixed features — it illustrates the data flow of a CNN without any learning. A trained CNN would replace both the filters AND the final matcher with learned weights.',
      details: [
        { label: 'Cosine similarity', text: 'cos(a,b) = (a·b)/(‖a‖‖b‖) — measures pattern overlap regardless of overall brightness.' },
        { label: 'Softmax', text: 'σ(z)_i = e^{z_i}/Σ e^{z_j} — converts scores into a probability distribution over classes.' },
        { label: 'Why CNNs beat dense nets on images', text: 'Local receptive fields + weight sharing + pooling encode translation tolerance and slash parameters versus a fully-connected net.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'Hierarchy of features', description: 'Early layers detect edges/blobs; deeper layers compose them into parts and objects. This lab shows only the first layer.', recommendation: 'Read the feature maps as "where does this simple pattern occur" — depth is what builds semantics.' },
    { category: 'VERIFICATION', title: 'Fixed vs learned', description: 'With hand-picked filters and template matching, accuracy is limited and brittle to new glyphs.', recommendation: 'Treat the result as a wiring diagram, not a performant classifier; training is what makes CNNs work.' },
  ],
};
