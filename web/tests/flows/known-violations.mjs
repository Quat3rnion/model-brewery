// WCAG A/AA debt by view and rule ID. Each selector allows one failing
// node in each listed project. Remove the selector when a fix lands.
export const knownViolations = {
  profiles: {
    'label-content-name-mismatch': {
      projects: ['1280-light', '1280-dark', '390-light', '390-dark'],
      nodes: ['button[aria-label^="Use Fixture HGN profile revision "]'],
    },
  },
  job: {
    'scrollable-region-focusable': {
      projects: ['390-light', '390-dark'],
      nodes: ['main section[aria-labelledby="res-h"] .table-wrap'],
    },
  },
};
