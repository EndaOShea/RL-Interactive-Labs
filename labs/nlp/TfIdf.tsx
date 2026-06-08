import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import Heatmap from '../../components/labkit/viz/Heatmap';
import DistributionBars from '../../components/labkit/viz/DistributionBars';
import { AlgoPill, RunControls, Legend, MonoLabel } from '../../components/stage/primitives';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { tfidfPython } from './python';
import { tfidf, cosine, TFIDF_DOCS } from './shared';

const ACCENT = '#14b8a6';

const TfIdfLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const docs = TFIDF_DOCS;
  const [selA, setSelA] = useState(0);
  const [selB, setSelB] = useState(1);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const model = useMemo(() => tfidf(docs), [docs]);
  const simAB = cosine(model.tfidf[selA], model.tfidf[selB]);

  // Transpose: rows = vocab, cols = docs (so word labels are legible on rows)
  const matrixT = model.vocab.map((_, j) => model.tfidf.map((row) => row[j]));

  // Shared-term contributions to cosine(dA, dB)
  const rawContribs = model.vocab.map((w, j) => ({
    label: w,
    value: model.tfidf[selA][j] * model.tfidf[selB][j],
  }));
  const positiveContribs = rawContribs.filter((b) => b.value > 0).sort((a, b) => b.value - a.value).slice(0, 6);
  const topSharedTerm = positiveContribs.length > 0 ? positiveContribs[0].label : '—';
  const contribBars = positiveContribs.length > 0
    ? positiveContribs
    : [{ label: '(no shared terms)', value: 0, muted: true }];

  // Words with idf ≈ 0 (present in all docs)
  const nearZeroIdfWords = model.vocab
    .filter((_, j) => model.idf[j] < 0.01)
    .join(', ') || '(none)';

  const compare = () => {
    narration.narratePhase(
      `tfidf:${selA}:${selB}`,
      `Document ${selA} and document ${selB} have a cosine similarity of ${simAB.toFixed(3)}. ${
        simAB > 0.3
          ? `They cover the same topic, sharing key terms like "${topSharedTerm}" that carry high TF-IDF weight.`
          : `They cover different topics; any words they share are common low-IDF words that barely contribute to the score.`
      } TF-IDF zeros out ubiquitous words like "${nearZeroIdfWords}" and boosts rare informative ones, so cosine similarity reflects genuine topic overlap.`,
    );
    setLastLog({
      algorithm: 'TF-IDF · bag-of-words similarity',
      stepDescription: `Compare document ${selA} and document ${selB} using TF-IDF vectors and cosine similarity`,
      formula: 'tf·idf,  idf = ln(N/df);  sim = cos(dᵢ, dⱼ)',
      variables: {
        i: selA,
        j: selB,
        'cos': +simAB.toFixed(3),
        vocab: model.vocab.length,
        topTerm: topSharedTerm,
      },
      result: `d${selA} vs d${selB}: cos = ${simAB.toFixed(3)} (${simAB > 0.3 ? 'similar' : 'different'} topics)`,
      mathDetails: {
        params: [
          {
            label: 'idf kills ubiquitous words',
            info: `Words appearing in all ${docs.length} documents get idf = ln(${docs.length}/${docs.length}) = 0. Current zero-idf words: ${nearZeroIdfWords}. They contribute nothing to cosine.`,
          },
          {
            label: 'shared terms drive similarity',
            info: positiveContribs.length > 0
              ? `The top shared terms are: ${positiveContribs.map((b) => `"${b.label}" (${b.value.toFixed(3)})`).join(', ')}. Each contributes tfidf[i]·tfidf[j] to the dot product.`
              : `Documents ${selA} and ${selB} share no vocabulary with non-zero TF-IDF weight — they are on entirely different topics.`,
          },
          {
            label: 'cosine ignores length',
            info: 'Cosine divides the dot product by both vector norms, so a longer document that simply repeats the same words scores identically to a shorter one with the same proportions.',
          },
        ],
        implication: `TF-IDF + cosine is the classical search baseline (refined as BM25) that powered information retrieval before dense neural embeddings. The Semantic Search lab shows how learned dense vectors capture synonyms and paraphrases that bag-of-words misses entirely.`,
      },
    });
  };

  return (
    <LabStage
      descriptor={descriptor}
      running={false}
      narration={narration}
      stats={[
        { label: 'vocab', value: model.vocab.length },
        { label: 'docs', value: docs.length },
        { label: 'cos(dA,dB)', value: simAB.toFixed(3), color: ACCENT },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, tfidfPython(docs))}
      grid={(
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14, alignItems: 'center' }}>
          <MonoLabel>TF-IDF weight · rows = vocabulary, cols = documents</MonoLabel>
          <Heatmap
            matrix={matrixT}
            mode="diverging"
            min={0}
            rowLabels={model.vocab.map((w) => (w.length > 7 ? w.slice(0, 6) + '…' : w))}
            colLabels={docs.map((_, i) => 'd' + i)}
            cell={18}
            accent={ACCENT}
          />
          <MonoLabel style={{ marginTop: 4 }}>shared-term contribution to cos(dA,dB)</MonoLabel>
          <DistributionBars
            bars={contribBars}
            width={360}
            accent={ACCENT}
          />
        </div>
      )}
      controls={(
        <RunControls
          isPlaying={false}
          onPlay={compare}
          onReset={() => { setLastLog(null); narration.cancel(); }}
        />
      )}
      legend={(
        <Legend
          title="HEATMAP"
          items={[
            { color: ACCENT, label: 'high tf-idf' },
            { color: '#1a2335', label: '~0 (e.g. "the")' },
          ]}
        />
      )}
      lastLog={lastLog}
      contextInsight={`d${selA} vs d${selB}: cosine = ${simAB.toFixed(3)} (${simAB > 0.3 ? 'same topic' : 'different topics'}). ${positiveContribs.length > 0 ? `Primary drivers: ${positiveContribs.slice(0, 3).map((b) => `"${b.label}"`).join(', ')} — rare, high-idf terms shared by both documents.` : 'No shared non-zero TF-IDF terms: the docs cover different vocabulary entirely.'}`}
      params={(
        <ParamsWrap>
          <ParamsHead
            title="TF-IDF & Similarity"
            hint="Weighted bag-of-words vectors and cosine document similarity."
          />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Comparison presets</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              <AlgoPill accent={ACCENT} onClick={() => { setSelA(0); setSelB(1); setLastLog(null); }}>
                cat/dog story (0 vs 1)
              </AlgoPill>
              <AlgoPill accent={ACCENT} onClick={() => { setSelA(2); setSelB(3); setLastLog(null); }}>
                market pair (2 vs 3)
              </AlgoPill>
              <AlgoPill accent={ACCENT} onClick={() => { setSelA(0); setSelB(2); setLastLog(null); }}>
                cross-topic (0 vs 2)
              </AlgoPill>
            </div>
          </div>
          <DocSelect
            label="Document A"
            value={selA}
            docs={docs}
            onChange={(v) => { setSelA(v); setLastLog(null); }}
          />
          <DocSelect
            label="Document B"
            value={selB}
            docs={docs}
            onChange={(v) => { setSelB(v); setLastLog(null); }}
          />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{
        topic: 'TF-IDF and document similarity',
        docA: selA,
        docB: selB,
        cosine: +simAB.toFixed(3),
        vocabSize: model.vocab.length,
      }}
      apiPanel={apiPanel}
    />
  );
};

const DocSelect: React.FC<{
  label: string;
  value: number;
  docs: string[];
  onChange: (v: number) => void;
}> = ({ label, value, docs, onChange }) => (
  <div style={{ marginTop: 10 }}>
    <MonoLabel style={{ marginBottom: 6 }}>{label}</MonoLabel>
    <select
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      style={{
        width: '100%',
        padding: '6px 8px',
        background: 'var(--bg0)',
        color: 'var(--t0)',
        border: '1px solid var(--border)',
        borderRadius: 6,
        fontFamily: 'var(--mono)',
        fontSize: 12,
      }}
    >
      {docs.map((d, i) => (
        <option key={i} value={i}>
          d{i}: {d.length > 28 ? d.slice(0, 28) + '…' : d}
        </option>
      ))}
    </select>
  </div>
);

export default TfIdfLab;
