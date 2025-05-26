document.addEventListener('DOMContentLoaded', () => {
  const links   = Array.from(document.querySelectorAll('.task-link'));
  const entries = Array.from(document.querySelectorAll('.method-entry'));

  function applyFilter(task) {
    const filterVal = task.toLowerCase();
    entries.forEach(div => {
      const tasks = div.dataset.tasks
        .split(',')
        .map(t => t.trim().toLowerCase())
        .filter(Boolean);

      div.style.display = (filterVal === 'all' || tasks.includes(filterVal))
        ? ''
        : 'none';
    });

    links.forEach(a => {
      a.classList.toggle('active',
        a.dataset.task.toLowerCase() === filterVal
      );
    });
  }

  links.forEach(a => {
    a.addEventListener('click', ev => {
      ev.preventDefault();
      applyFilter(a.dataset.task);
    });
  });

  // initialize on “all”
  applyFilter('all');
});