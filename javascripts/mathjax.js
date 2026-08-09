window.MathJax = {
  tex: {
    inlineMath: [["$", "$"], ["\\(", "\\)"]],
    displayMath: [["$$", "$$"], ["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    skipHtmlTags: [
      "script",
      "noscript",
      "style",
      "textarea",
      "pre",
      "code"
    ]
  }
};

window.addEventListener("load", function () {
  if (window.MathJax) {
    MathJax.typesetPromise();
  }
});