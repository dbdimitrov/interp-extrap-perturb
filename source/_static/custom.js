document.addEventListener('DOMContentLoaded', () => {
  const links   = Array.from(document.querySelectorAll('.task-link'));
  const entries = Array.from(document.querySelectorAll('.method-entry'));

  function applyFilter(task) {
    entries.forEach(div => {
      const tasks = div.dataset.tasks.split(/\s+/);
      div.style.display = (task === 'all' || tasks.includes(task))
        ? ''
        : 'none';
    });
    // highlight active link
    links.forEach(a => {
      a.classList.toggle('active', a.dataset.task === task);
    });
  }

  links.forEach(a => {
    a.addEventListener('click', ev => {
      ev.preventDefault();
      applyFilter(a.dataset.task);
    });
  });

  // initialize
  applyFilter('all');
});