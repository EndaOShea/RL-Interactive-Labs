// Small re-export barrel so the LLM labs import their right-column param helpers
// from one place. ParamsWrap/ParamsHead come from the Classic ML shared module;
// ParamSlider is a stage primitive.
export { ParamsWrap, ParamsHead } from '../classic-ml/shared';
export { ParamSlider } from '../../components/stage/primitives';
