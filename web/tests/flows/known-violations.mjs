// WCAG A/AA debt by view and rule ID. Each selector allows one failing
// node in each listed project. Remove the selector when a fix lands.
const runningJobStatus = {
  'color-contrast': {
    projects: ['1280-light', '390-light'],
    nodes: [
      'section[aria-label="Running jobs"] .what > .badge.info',
      'section[aria-label="Running jobs"] .what > .badge.warn',
    ],
  },
};

export const knownViolations = {
  library: runningJobStatus,
  queue: runningJobStatus,
  transfers: runningJobStatus,
  compare: runningJobStatus,
  convert: runningJobStatus,
  evaluate: runningJobStatus,
  benchmarks: runningJobStatus,
  calibration: runningJobStatus,
  settings: runningJobStatus,
  profiles: {
    ...runningJobStatus,
    'label-content-name-mismatch': {
      projects: ['1280-light', '1280-dark', '390-light', '390-dark'],
      nodes: ['button[aria-label^="Use Fixture HGN profile revision "]'],
    },
  },
  weights: runningJobStatus,
  job: {
    ...runningJobStatus,
    'scrollable-region-focusable': {
      projects: ['390-light', '390-dark'],
      nodes: ['main section[aria-labelledby="res-h"] .table-wrap'],
    },
  },
};
