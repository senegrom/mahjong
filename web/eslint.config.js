import js from '@eslint/js';
import globals from 'globals';

export default [
  // runtime/** is generated, minified ONNX Runtime vendor output. It is built
  // from our model-specific operator list and exercised by the browser tests;
  // lint the source that selects/builds it, not generated third-party code.
  { ignores: ['node_modules/**', 'dist/**', 'public/**', 'runtime/**', 'src/wasm/**', 'test-results/**', '.tile-effects-*/**'] },
  {
    ...js.configs.recommended,
    files: ['**/*.{js,mjs}'],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: 'module',
      // Regression scripts also contain callbacks evaluated inside Chrome.
      globals: { ...globals.node, ...globals.browser },
    },
    rules: {
      ...js.configs.recommended.rules,
      'no-unused-vars': ['error', { argsIgnorePattern: '^_', varsIgnorePattern: '^_', caughtErrorsIgnorePattern: '^_' }],
    },
  },
];
