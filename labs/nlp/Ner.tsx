import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import Heatmap from '../../components/labkit/viz/Heatmap';
import { AlgoPill, RunControls, Legend, MonoLabel, GOOD } from '../../components/stage/primitives';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { nerPython } from './python';
import { viterbi, NER_TAGS, NER_SENTENCES, emission } from './shared';
import type { NerTag } from './shared';

const ACCENT = '#14b8a6';

const TAG_COLOR: Record<NerTag, string> = {
  O: '#6b7494',
  PER: '#14b8a6',
  LOC: '#fbbf24',
  ORG: '#a855f7',
};

const NerLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const [sentIdx, setSentIdx] = useState(0);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const sentence = NER_SENTENCES[sentIdx];
  const { tags, score } = useMemo(() => viterbi(sentence), [sentIdx]);

  const emissMatrix = NER_TAGS.map((tag) => sentence.map((w) => emission(w, tag)));

  const entityCount = tags.filter((t) => t !== 'O').length;

  // Compute entity spans for display
  const spansText = (() => {
    const spans: string[] = [];
    let i = 0;
    while (i < tags.length) {
      if (tags[i] !== 'O') {
        spans.push(`${sentence[i]}→${tags[i]}`);
      }
      i++;
    }
    return spans.join(', ') || '(none)';
  })();

  const tag = () => {
    const entityParts: string[] = [];
    tags.forEach((t, i) => {
      if (t !== 'O') entityParts.push(`${sentence[i]} (${t})`);
    });
    const entityStr = entityParts.join(', ') || 'none';

    narration.narratePhase(
      `ner:${sentIdx}`,
      `The Viterbi decoder labeled ${sentence.length} tokens in the sentence "${sentence.join(' ')}". ${
        entityCount > 0
          ? `It found ${entityCount} entit${entityCount === 1 ? 'y' : 'ies'}: ${entityStr}. Each token's emission score from the lexicon and capitalisation prior is combined with transition scores between neighbouring tags, and Viterbi finds the globally best labeling.`
          : 'No named entities were found — all tokens scored highest as Outside. Lowercase common words receive a strong O emission score of +2.0.'
      }`,
    );

    setLastLog({
      algorithm: 'NER · Viterbi sequence labeling',
      stepDescription: `Tag each token in "${sentence.join(' ')}" with PER/LOC/ORG/O`,
      formula: 'argmax_y  Σₜ emission(xₜ, yₜ) + transition(yₜ₋₁, yₜ)',
      variables: {
        tokens: sentence.length,
        entities: entityCount,
        score: +score.toFixed(2),
        spans: spansText,
      },
      result: entityCount > 0
        ? `Entities: ${entityStr}`
        : 'No entities found (all tokens tagged O)',
      mathDetails: {
        params: [
          {
            label: 'emission = lexicon + shape',
            info: 'Known names (Alice, Google, Berlin…) get strong lexicon scores. Unseen capitalised words get a mild entity boost (+0.4) and a mild O penalty (−0.5). Lowercase words strongly prefer O (+2.0 vs −2.0).',
          },
          {
            label: 'transition couples neighbours',
            info: 'O→O earns +0.5 (fluent non-entity runs); same-tag continuation earns +0.3. Viterbi uses these jointly, so the globally best sequence may differ from picking the best tag at each position independently.',
          },
          {
            label: 'token→tag assignments',
            info: sentence.map((w, i) => `${w}→${tags[i]}`).join('  ·  '),
          },
        ],
        implication: `Viterbi guarantees the globally optimal tag sequence (argmax over all Sᵀ paths) in O(T·S²) time. Modern taggers replace these hand-crafted scores with a BiLSTM or Transformer encoder feeding a CRF layer, but keep the same Viterbi decode step.`,
      },
    });
  };

  const sentenceLabels = NER_SENTENCES.map((s) => s.join(' '));

  return (
    <LabStage
      descriptor={descriptor}
      running={false}
      narration={narration}
      stats={[
        { label: 'tokens', value: sentence.length },
        { label: 'entities', value: entityCount, color: ACCENT },
        { label: 'score', value: score.toFixed(2), color: GOOD },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, nerPython())}
      grid={(
        <div style={{ display: 'flex', flexDirection: 'column', gap: 18, alignItems: 'center' }}>
          {/* Token timeline */}
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'center' }}>
            {sentence.map((word, i) => {
              const t = tags[i];
              const col = TAG_COLOR[t];
              const isEntity = t !== 'O';
              return (
                <div
                  key={i}
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: 4,
                    padding: '8px 14px',
                    borderRadius: 8,
                    border: `1.5px solid ${col}`,
                    background: isEntity
                      ? `color-mix(in srgb, ${col} 12%, transparent)`
                      : 'rgba(107,116,148,0.08)',
                    minWidth: 48,
                  }}
                >
                  <span style={{
                    fontFamily: 'var(--mono)',
                    fontSize: 13,
                    fontWeight: 600,
                    color: isEntity ? col : 'var(--t1)',
                  }}>
                    {word}
                  </span>
                  <span style={{
                    fontFamily: 'var(--mono)',
                    fontSize: 10,
                    color: col,
                    letterSpacing: '.06em',
                  }}>
                    {t}
                  </span>
                </div>
              );
            })}
          </div>

          {/* Emission heatmap */}
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
            <Heatmap
              matrix={emissMatrix}
              mode="diverging"
              min={-2.5}
              max={3}
              rowLabels={[...NER_TAGS]}
              colLabels={sentence}
              cell={26}
              accent={ACCENT}
              showValues
            />
            <MonoLabel style={{ fontSize: 10, color: 'var(--t2)' }}>
              emission score: word → tag affinity (Viterbi also adds transition scores)
            </MonoLabel>
          </div>
        </div>
      )}
      controls={(
        <RunControls
          isPlaying={false}
          onPlay={tag}
          onReset={() => { setLastLog(null); narration.cancel(); }}
        />
      )}
      legend={(
        <Legend
          title="TAGS"
          items={[
            { color: TAG_COLOR.PER, label: 'PER' },
            { color: TAG_COLOR.LOC, label: 'LOC' },
            { color: TAG_COLOR.ORG, label: 'ORG' },
            { color: TAG_COLOR.O, label: 'O (outside)' },
          ]}
        />
      )}
      lastLog={lastLog}
      contextInsight={`Sentence: "${sentence.join(' ')}". Decoded: ${
        entityCount > 0
          ? `${entityCount} entit${entityCount === 1 ? 'y' : 'ies'} — ${spansText}. `
          : 'no entities found. '
      }Emission scores highlight which tag each word fits (lexicon + capitalisation); transition scores couple neighbours so the Viterbi sequence may differ from picking the best tag per token in isolation.`}
      params={(
        <ParamsWrap>
          <ParamsHead
            title="Named Entity Recognition"
            hint="Tag each token PER/LOC/ORG/O; Viterbi finds the best whole-sentence labeling."
          />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Sentences</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {NER_SENTENCES.map((s, i) => {
                const label = sentenceLabels[i];
                return (
                  <AlgoPill
                    key={i}
                    accent={ACCENT}
                    active={sentIdx === i}
                    onClick={() => { setSentIdx(i); setLastLog(null); narration.cancel(); }}
                  >
                    {label.length > 30 ? label.slice(0, 29) + '…' : label}
                  </AlgoPill>
                );
              })}
            </div>
          </div>
          <div style={{ marginTop: 12 }}>
            <MonoLabel style={{ marginBottom: 6, fontSize: 10 }}>Tag legend</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              {(Object.entries(TAG_COLOR) as [NerTag, string][]).map(([tag, col]) => (
                <span key={tag} style={{ fontFamily: 'var(--mono)', fontSize: 11, color: col }}>
                  {tag} &mdash; {tag === 'PER' ? 'Person' : tag === 'LOC' ? 'Location' : tag === 'ORG' ? 'Organisation' : 'Outside'}
                </span>
              ))}
            </div>
          </div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{
        topic: 'Named entity recognition (Viterbi sequence labeling)',
        sentence: sentence.join(' '),
        tags: tags.join(','),
        entities: entityCount,
        score: +score.toFixed(2),
      }}
      apiPanel={apiPanel}
    />
  );
};

export default NerLab;
