// The page works without JavaScript. Only swap a pre-rendered GIF and poster;
// no model, simulation, data fetch, or inference runs in the browser.
const image = document.getElementById('process-animation');
const button = document.getElementById('motion-toggle');
const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
const poster = image.getAttribute('src');
let playing = false;

function setMotion(enabled) {
  playing = enabled;
  image.src = enabled ? image.dataset.animation : poster;
  button.textContent = enabled ? 'Dừng ảnh động' : 'Phát ảnh động';
  button.setAttribute('aria-pressed', String(enabled));
}
button.hidden = false;
button.addEventListener('click', () => setMotion(!playing));
preference.addEventListener('change', event => setMotion(!event.matches));
image.addEventListener('error', () => {
  if (playing) {
    setMotion(false);
    button.hidden = true;
  }
});
setMotion(!preference.matches);
