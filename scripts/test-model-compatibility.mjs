import assert from 'node:assert/strict';
import { build } from 'esbuild';

const compiled = await build({stdin:{contents:"export { callLlm } from './services/llmClient'; export { PROVIDERS } from './services/providers';",resolveDir:process.cwd()},
  bundle:true,write:false,platform:'node',format:'cjs',plugins:[{name:'mock-google',setup(b){
    b.onResolve({filter:/^@google\/genai$/},() => ({path:'google',namespace:'mock'}));
    b.onLoad({filter:/.*/,namespace:'mock'},() => ({contents:'export const ThinkingLevel = {LOW: \"LOW\"}; export class GoogleGenAI { models = {generateContent: async args => {globalThis.captured = args; return {text:"ok"}}}; }'}));
  }}]});
const module={exports:{}};
new Function('module','exports',compiled.outputFiles[0].text)(module,module.exports);
const {callLlm,PROVIDERS}=module.exports;
globalThis.fetch=async (_url,args) => {globalThis.captured=JSON.parse(args.body);return {ok:true,json:async () => ({choices:[{message:{content:'ok'}}],content:[{type:'thinking',text:'private'},{type:'text',text:'ok'}]})};};
for (const provider of Object.values(PROVIDERS)) {
  assert(provider.models.some(m => m.id === provider.defaultModel));
  for (const model of provider.models) {
    assert.equal(await callLlm(provider.id,model.id,'test','test-key'),'ok');
    const body=globalThis.captured;
    if(provider.id==='google') assert.deepEqual(body.config.thinkingConfig,{thinkingLevel:'LOW'});
    if(provider.id==='deepseek') assert.equal(body.reasoning_effort,'low');
    if(provider.id==='anthropic') {
      assert(!('temperature' in body));
      if(model.id.startsWith('claude-haiku')) assert(body.max_tokens > body.thinking.budget_tokens);
      else {assert.deepEqual(body.thinking,{type:'adaptive'});assert.equal(body.output_config.effort,'low');}
    }
  }
}
console.log('All 17 curated model calls passed with mocked providers.');
