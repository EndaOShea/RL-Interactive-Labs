// Label wrapping for node boxes: whole words, at most `maxLines` lines; the last
// line ends with an ellipsis only when words remain (the full label is in the
// node's tooltip, its accessible name and the detail panel).
export function wrapLabel(text: string, maxChars = 27, maxLines = 3): string[] {
  const words = text.split(/\s+/).filter(Boolean);
  const lines: string[] = [];
  let line = '';
  let i = 0;
  for (; i < words.length; i++) {
    const word = words[i]!;
    const next = line ? `${line} ${word}` : word;
    if (next.length <= maxChars) { line = next; continue; }
    if (line) { lines.push(line); line = ''; if (lines.length === maxLines) break; }
    line = word.length > maxChars ? `${word.slice(0, maxChars - 1)}…` : word;
  }
  if (line && lines.length < maxLines) lines.push(line);
  if (i < words.length && lines.length) {
    const last = lines[lines.length - 1]!;
    lines[lines.length - 1] = last.length >= maxChars ? `${last.slice(0, maxChars - 1)}…` : `${last}…`;
  }
  return lines;
}
