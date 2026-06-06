import React from 'react';
import { LabDescriptor } from '../../catalog/types';
import { PATHFINDING_CONTENT, GRAPH_SEARCH_CONTENT } from './content';

const ACCENT = '#38bdf8';

export const SEARCH_LABS: LabDescriptor[] = [
  {
    id: 'pathfinding',
    category: 'search',
    title: 'Grid Pathfinding',
    subtitle: 'BFS · DFS · Dijkstra · Greedy · A* on a grid',
    blurb: 'Draw walls and race the classic search algorithms — watch the frontier spread and the path snap into place.',
    icon: 'M4 4h6v6H4zM14 14h6v6h-6zM10 7h6M7 10v4M16 14v-4h-2',
    accent: ACCENT,
    codeFile: 'pathfinding.py',
    content: PATHFINDING_CONTENT,
    component: React.lazy(() => import('./Pathfinding')),
  },
  {
    id: 'graph-search',
    category: 'search',
    title: 'Weighted Graph Search',
    subtitle: 'Cheapest-path search on a node graph',
    blurb: 'See why hops ≠ cost: BFS, Dijkstra and A* on a weighted graph, with g/h/f and the path lit up.',
    icon: 'M6 7a2 2 0 1 0 0-.01M18 7a2 2 0 1 0 0-.01M12 18a2 2 0 1 0 0-.01M7.5 7.5 11 16M16.5 7.5 13 16M8 6h8',
    accent: ACCENT,
    codeFile: 'graph_search.py',
    content: GRAPH_SEARCH_CONTENT,
    component: React.lazy(() => import('./GraphSearch')),
  },
];
