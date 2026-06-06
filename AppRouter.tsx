import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import App from './App';
import HomeCatalog from './catalog/HomeCatalog';
import AreaHost from './components/labkit/AreaHost';

// Root of the multi-area platform. The existing RL app mounts unchanged at /rl;
// the catalog home is the hub. New areas get a generic AreaHost keyed by
// category. nginx already serves index.html for unknown paths (SPA fallback),
// so deep links work in the Docker build.
const AppRouter: React.FC = () => (
  <BrowserRouter>
    <Routes>
      <Route path="/" element={<HomeCatalog />} />
      <Route path="/rl" element={<App />} />
      <Route path="/classic-ml/:labId?" element={<AreaHost category="classic-ml" />} />
      <Route path="/search/:labId?" element={<AreaHost category="search" />} />
      <Route path="/unsupervised/:labId?" element={<AreaHost category="unsupervised" />} />
      <Route path="/supervised/:labId?" element={<AreaHost category="supervised" />} />
      <Route path="/logic/:labId?" element={<AreaHost category="logic" />} />
      <Route path="/neural/:labId?" element={<AreaHost category="neural" />} />
      <Route path="/model-checking/:labId?" element={<AreaHost category="model-checking" />} />
      <Route path="/image/:labId?" element={<AreaHost category="image" />} />
      <Route path="/audio/:labId?" element={<AreaHost category="audio" />} />
      <Route path="/llm/:labId?" element={<AreaHost category="llm" />} />
      <Route path="/diffusion/:labId?" element={<AreaHost category="diffusion" />} />
      <Route path="/math/:labId?" element={<AreaHost category="math" />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  </BrowserRouter>
);

export default AppRouter;
