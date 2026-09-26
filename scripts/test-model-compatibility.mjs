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
    if(provider.id==='openai') {assert(!('temperature' in body));assert(!('tools' in body));}
    if(provider.id==='deepseek') assert.equal(body.reasoning_effort,'low');
    if(provider.id==='anthropic') {
      assert(!('temperature' in body));
      if(model.id.startsWith('claude-haiku')) assert(body.max_tokens > body.thinking.budget_tokens);
      else {assert.deepEqual(body.thinking,{type:'adaptive'});assert.equal(body.output_config.effort,'low');}
    }
  }
}
console.log(`All ${Object.values(PROVIDERS).reduce((n,p) => n+p.models.length,0)} curated model calls passed with mocked providers.`);

// Render the actual picker and exercise its model-change handler for new releases.
const pickerBuild = await build({stdin:{contents:"export { default as Panel } from './components/stage/ApiKeyPanel';",resolveDir:process.cwd()},bundle:true,write:false,platform:'node',format:'cjs',packages:'external'});
const { createRequire } = await import('node:module');
const require = createRequire(import.meta.url);
const React = require('react');
const { renderToStaticMarkup } = require('react-dom/server');
const pickerModule = {exports:{}};
new Function('module','exports','require',pickerBuild.outputFiles[0].text)(pickerModule,pickerModule.exports,require);
const {Panel}=pickerModule.exports;
function selects(node) {
  if (!node || typeof node !== 'object') return [];
  if (Array.isArray(node)) return node.flatMap(selects);
  return [...(node.type === 'select' ? [node] : []), ...selects(node.props?.children)];
}
for (const [provider, ids] of Object.entries({openai:['gpt-6-astra','gpt-6-sol','gpt-6-luna'],anthropic:['claude-opus-5-5','claude-fable-5-1'],google:['gemini-3.8-flash']})) {
  let model=PROVIDERS[provider].defaultModel;
  const props={provider,providerConfig:PROVIDERS[provider],keyInput:'',manualKey:'',hasKey:false,
    onProviderChange:()=>{},onModelChange:value=>{model=value},setKeyInput:()=>{},onActivateKey:()=>{},onClearKey:()=>{}};
  for (const id of ids) {
    const tree=Panel({...props,model});
    const html=renderToStaticMarkup(tree);
    assert(html.includes(`value="${id}"`),`Picker omits ${id}`);
    selects(tree)[1].props.onChange({target:{value:id}});
    assert.equal(model,id);
    assert.match(renderToStaticMarkup(React.createElement(Panel,{...props,model})),new RegExp(`value="${id}"[^>]*selected`));
  }
}
console.log('Rendered picker exposes and selects current OpenAI, Anthropic and Google models.');
